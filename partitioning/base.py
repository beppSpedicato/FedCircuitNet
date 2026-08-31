from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd




class DatasetPartitioner(ABC):
    """Abstract base for CircuitNet-N28 dataset partitioning strategies.

    Subclasses implement :meth:`partition` to split a metadata DataFrame
    (produced by parsing sample filenames) into ``n_partitions`` subsets.
    Each returned DataFrame carries the same columns as the input, including
    the ``filename`` column that maps rows back to the actual ``.npy`` files.

    Expected input DataFrame columns (from the filename parsing convention):
        design_name, macro_count, clock_ns, utilization,
        macro_placement, power_mesh, filler_insertion, filename
    """

    def __init__(self, n_partitions: int) -> None:
        if n_partitions < 1:
            raise ValueError(f"n_partitions must be >= 1, got {n_partitions}")
        self.n_partitions = n_partitions

    @abstractmethod
    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split *df* into :attr:`n_partitions` non-overlapping subsets.

        Args:
            df: Metadata DataFrame with at least a ``filename`` column.

        Returns:
            List of ``n_partitions`` DataFrames whose union equals *df*.
        """
        ...

    def _validate(self, df: pd.DataFrame) -> None:
        if "filename" not in df.columns:
            raise ValueError("DataFrame must contain a 'filename' column.")
        if len(df) < self.n_partitions:
            raise ValueError(
                f"Dataset has {len(df)} samples but {self.n_partitions} "
                "partitions were requested."
            )


class FedChipPartitioner(DatasetPartitioner):
    """Group assignment followed by FedChip-style Dirichlet spillover.

    Subclasses decide only *how* each row is mapped to one of the
    ``n_partitions`` groups (:meth:`_assign_groups`).  This class owns the
    shared second stage, taken from the FedChip recipe (Ashkboos et al.,
    2024): group ``c`` keeps ``cluster_share`` (default 80 %) of its rows
    on its owner client ``c``, and the remaining ``1 - cluster_share`` is
    redistributed over all ``n_partitions`` clients through a fresh
    ``Dirichlet(dirichlet_alpha)`` draw.

    The output is still a proper non-overlapping partition of the input
    rows, but no client is a pure single-group silo: every client sees a
    small, randomly-weighted tail of the other groups' distributions.
    Smaller ``dirichlet_alpha`` concentrates that tail on fewer clients.

    Args:
        n_partitions: Number of clients (== number of groups).
        cluster_share: Fraction of each group kept by its owner client.
            Must lie in ``[0, 1]``.  ``1.0`` disables the spillover and
            recovers a hard group-per-client split.  Default ``0.8``.
        dirichlet_alpha: Concentration of the Dirichlet used to
            redistribute the leftover ``1 - cluster_share`` fraction.
            Smaller => more skew.  Default ``0.5``.
        random_state: Seed for the spillover RNG (and, in subclasses that
            need one, for the group-assignment step).
    """

    def __init__(
        self,
        n_partitions: int,
        cluster_share: float = 0.8,
        dirichlet_alpha: float = 0.5,
        random_state: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if not 0.0 <= cluster_share <= 1.0:
            raise ValueError(
                f"cluster_share must be in [0, 1], got {cluster_share}"
            )
        if dirichlet_alpha <= 0:
            raise ValueError(
                f"dirichlet_alpha must be > 0, got {dirichlet_alpha}"
            )
        self.cluster_share = float(cluster_share)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.random_state = int(random_state)

    @abstractmethod
    def _assign_groups(self, df: pd.DataFrame) -> np.ndarray:
        """Return an integer group id in ``[0, n_partitions)`` per row.

        Rows that no group claims may be marked ``-1``; they are dealt to
        random clients by :meth:`partition`.
        """
        ...

    @staticmethod
    def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
        """Split *total* items over *weights* without losing/creating items."""
        raw = weights * total
        floors = np.floor(raw).astype(int)
        remainder = total - int(floors.sum())
        if remainder > 0:
            frac = raw - floors
            order = np.argsort(-frac, kind="stable")
            floors[order[:remainder]] += 1
        return floors

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        df = df.copy().reset_index(drop=True)

        group_ids = self._assign_groups(df)
        K = self.n_partitions
        rng = np.random.default_rng(self.random_state)
        assignment = np.full(len(df), -1, dtype=int)

        for c in range(K):
            positions = np.where(group_ids == c)[0]
            if len(positions) == 0:
                continue
            rng.shuffle(positions)

            n_own = int(round(self.cluster_share * len(positions)))
            n_own = min(n_own, len(positions))
            own = positions[:n_own]
            leftover = positions[n_own:]

            assignment[own] = c

            if len(leftover) > 0:
                weights = rng.dirichlet([self.dirichlet_alpha] * K)
                counts = self._largest_remainder(len(leftover), weights)
                cursor = 0
                for target, take in enumerate(counts):
                    if take == 0:
                        continue
                    assignment[leftover[cursor : cursor + take]] = target
                    cursor += take

        stray = np.where(assignment == -1)[0]
        if len(stray) > 0:
            assignment[stray] = rng.integers(0, K, size=len(stray))

        return [df[assignment == i].reset_index(drop=True) for i in range(K)]


