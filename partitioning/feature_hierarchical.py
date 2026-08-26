from typing import List, Sequence

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class FeatureHierarchicalPartitioner(DatasetPartitioner):
    """Top-down hierarchical partitioning driven by an ordered feature list.

    Recursively splits the input DataFrame using the ``features`` list in
    order. At every level the current subset is grouped by the next feature;
    the ``n_partitions`` budget is distributed across the resulting groups
    proportionally to group size, and each group recurses with its share of
    the budget. Recursion stops when a subset only needs one partition
    (return as-is) or when the feature list is exhausted (return as-is).

    When a level has more distinct values than the remaining budget, adjacent
    values (in ``sort=True`` order) are merged into contiguous bins to hit
    the budget exactly. This keeps neighbouring feature values on the same
    client, which is the natural reading of "hierarchical by importance".

    Args:
        n_partitions: Number of clients.
        features: Ordered list of columns; earlier entries dominate the
            top-level splits.
    """

    def __init__(
        self,
        n_partitions: int,
        features: Sequence[str],
        **kwargs,
    ) -> None:
        super().__init__(n_partitions)
        if not features:
            raise ValueError("features must contain at least one column name")
        self.features = list(features)
        self.stratify_cols: List[str] = list(self.features)

    @staticmethod
    def _allocate_budget(sizes: np.ndarray, k: int) -> np.ndarray:
        raw = sizes / sizes.sum() * k
        alloc = np.floor(raw).astype(int)
        alloc = np.maximum(alloc, 1)
        while alloc.sum() > k:
            alloc[np.argmax(alloc)] -= 1
        while alloc.sum() < k:
            frac = raw - alloc
            alloc[int(np.argmax(frac))] += 1
        return alloc

    def _split(
        self,
        sub: pd.DataFrame,
        remaining_features: List[str],
        k: int,
    ) -> List[pd.DataFrame]:
        if k <= 1 or not remaining_features or len(sub) == 0:
            return [sub.reset_index(drop=True)]

        feature = remaining_features[0]
        rest = remaining_features[1:]

        groups = [g for _, g in sub.groupby(feature, sort=True)]
        m = len(groups)

        if m == 1:
            return self._split(sub, rest, k)

        if m >= k:
            bins = np.array_split(np.arange(m), k)
            return [
                pd.concat([groups[j] for j in bin_idx]).reset_index(drop=True)
                for bin_idx in bins
                if len(bin_idx) > 0
            ]

        sizes = np.array([len(g) for g in groups], dtype=float)
        alloc = self._allocate_budget(sizes, k)
        out: List[pd.DataFrame] = []
        for group, k_sub in zip(groups, alloc):
            out.extend(self._split(group, rest, int(k_sub)))
        return out

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        df = df.copy().reset_index(drop=True)

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"features missing from DataFrame: {missing}. "
                f"Available: {list(df.columns)}"
            )

        parts = self._split(df, list(self.features), self.n_partitions)

        if len(parts) != self.n_partitions:
            parts = sorted(parts, key=len, reverse=True)
            while len(parts) > self.n_partitions:
                smallest = parts.pop()
                parts[-1] = pd.concat([parts[-1], smallest]).reset_index(drop=True)
            while len(parts) < self.n_partitions:
                parts.append(df.iloc[0:0].reset_index(drop=True))

        return [p.reset_index(drop=True) for p in parts]
