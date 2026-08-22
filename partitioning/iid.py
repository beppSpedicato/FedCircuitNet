from typing import List, Optional

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class IIDPartitioner(DatasetPartitioner):
    """Partitions the CircuitNet-N28 dataset into IID subsets.

    Each partition mirrors the global distribution of design-configuration
    parameters (and optionally label severity tiers).  Within every stratum
    defined by a composite categorical key, samples are shuffled and then
    split as evenly as possible across the ``n_partitions`` clients; when a
    stratum size is not a multiple of ``n_partitions`` the leftover samples
    go to the partitions that are currently the smallest, so overall
    partition sizes stay balanced.

    Args:
        n_partitions: Number of subsets to produce.
        mode: Stratification scope:
            - ``"features"`` (default): stratify on design-configuration
              columns only (design_name, utilization, clock).
            - ``"features_label"``: also include the label tier column so
              each partition contains a proportional mix of DRC severity
              levels.  Requires the DataFrame to carry a ``tier`` column,
              produced by :class:`~partitioning.label_tier.LabelTierAssigner`.
        stratify_cols: Override the default set of categorical columns used
            for stratification.  Columns absent from the input DataFrame are
            silently skipped.
        label_tier_col: Name of the integer tier column.  Only used when
            ``mode="features_label"``.  Defaults to ``"tier"``.
    """

    _DEFAULT_STRATIFY_COLS: List[str] = [
        "design_name",
        "utilization",
        "clock",
    ]

    def __init__(
        self,
        n_partitions: int,
        mode: str = "features",
        stratify_cols: Optional[List[str]] = None,
        label_tier_col: str = "tier",
    ) -> None:
        super().__init__(n_partitions)
        if mode not in ("features", "features_label"):
            raise ValueError(f"mode must be 'features' or 'features_label', got {mode!r}")
        self.mode = mode
        self.stratify_cols = stratify_cols or self._DEFAULT_STRATIFY_COLS
        self.label_tier_col = label_tier_col

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Return ``n_partitions`` IID subsets of *df*.

        Args:
            df: Metadata DataFrame produced by parsing CircuitNet-N28 filenames.
                When ``mode="features_label"``, must also contain a tier column
                (see :class:`~partitioning.label_tier.LabelTierAssigner`).

        Returns:
            List of DataFrames, each with a reset integer index.
        """
        self._validate(df)

        if self.mode == "features_label" and self.label_tier_col not in df.columns:
            raise ValueError(
                f"mode='features_label' requires column '{self.label_tier_col}' in df. "
                "Run LabelTierAssigner.process_csv() and merge the result first."
            )

        df = df.copy().reset_index(drop=True)

        stratify = list(self.stratify_cols)
        if self.mode == "features_label":
            stratify.append(self.label_tier_col)

        available = [c for c in stratify if c in df.columns]
        strat_key = (
            df[available].astype(str).agg("-".join, axis=1)
            if available
            else pd.Series(["_all_"] * len(df), index=df.index)
        )

        assignment = np.empty(len(df), dtype=int)
        # Within each stratum, split shuffled samples into `n_partitions` chunks
        # of near-equal size (n // K each, remainder r spread one-per-partition).
        # The r "extra" samples are handed to whichever partitions are globally
        # smallest so partition sizes stay balanced overall -- avoids the bias
        # of always awarding the extras to partitions [0, ..., r-1].
        sizes = np.zeros(self.n_partitions, dtype=int)
        for group_positions in df.groupby(strat_key).groups.values():
            positions = np.array(group_positions)
            np.random.shuffle(positions)

            n = len(positions)
            base, remainder = divmod(n, self.n_partitions)
            per_partition = np.full(self.n_partitions, base, dtype=int)
            if remainder:
                extras = np.argsort(sizes, kind="stable")[:remainder]
                per_partition[extras] += 1

            cursor = 0
            for p in range(self.n_partitions):
                take = per_partition[p]
                assignment[positions[cursor : cursor + take]] = p
                cursor += take
            sizes += per_partition

        return [
            df[assignment == i].reset_index(drop=True)
            for i in range(self.n_partitions)
        ]
