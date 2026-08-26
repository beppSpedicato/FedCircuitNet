from typing import List, Sequence

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class DirichletPartitioner(DatasetPartitioner):
    """NIID-Bench-style Dirichlet partitioning on a metadata composite label.

    Given an ordered list of metadata columns, build the composite categorical
    key ``feat1|feat2|...`` for every sample. For every distinct composite
    level, draw a fresh ``Dirichlet(alpha)`` over the ``n_partitions``
    clients and split that level's samples across clients using the
    largest-remainder method.

    Smaller ``alpha`` -> stronger per-level skew (a level lands almost
    entirely on one client). Larger ``alpha`` -> distribution flattens toward
    uniform across clients.

    Args:
        n_partitions: Number of clients.
        features: Ordered list of metadata columns forming the composite
            categorical key. The order affects only the human-readable label
            layout, not the partition semantics.
        alpha: Dirichlet concentration parameter. Default 0.5
            (moderate skew, matches NIID-Bench convention).
        random_state: Seed for the Dirichlet RNG.
    """

    def __init__(
        self,
        n_partitions: int,
        features: Sequence[str],
        alpha: float = 0.5,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(n_partitions)
        if not features:
            raise ValueError("features must contain at least one column name")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.features = list(features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.stratify_cols: List[str] = list(self.features)

    @staticmethod
    def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
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

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"features missing from DataFrame: {missing}. "
                f"Available: {list(df.columns)}"
            )

        composite = df[self.features].astype(str).agg("|".join, axis=1)
        K = self.n_partitions
        rng = np.random.default_rng(self.random_state)
        assignment = np.full(len(df), -1, dtype=int)

        for _, group_positions in df.groupby(composite).groups.items():
            positions = np.array(group_positions)
            rng.shuffle(positions)
            weights = rng.dirichlet([self.alpha] * K)
            counts = self._largest_remainder(len(positions), weights)
            cursor = 0
            for target, take in enumerate(counts):
                if take == 0:
                    continue
                assignment[positions[cursor : cursor + take]] = target
                cursor += take

        stray = np.where(assignment == -1)[0]
        if len(stray) > 0:
            assignment[stray] = rng.integers(0, K, size=len(stray))

        return [
            df[assignment == i].reset_index(drop=True) for i in range(K)
        ]
