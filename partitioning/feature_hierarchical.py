import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd

from .base import FedChipPartitioner


class FeatureHierarchicalPartitioner(FedChipPartitioner):
    """Top-down hierarchical partitioning driven by an ordered feature list.

    This is the metadata-space counterpart of the distance-based clustering
    partitioners in :mod:`partitioning.clustering`.  Rather than embedding
    the metadata (one-hot -> quantile -> scale -> PCA) and letting a
    clustering algorithm discover groups, the split is taken *directly* on
    the raw metadata columns, in an order supplied by configuration:

        ``features: [design_name, clock, utilization]``

    The first entry produces the top-level cut, the second subdivides
    within each top-level bucket, and so on -- so feature order encodes
    design-knob importance (the Phase 2 ranking) and every client is
    describable in plain EDA terms ("RISCY-FPU-a at 500/200 MHz") instead
    of by a cluster centroid.  Keeping the columns raw is deliberate: the
    K-means preprocessing chain is meant to make a Euclidean metric
    meaningful, and there is no metric here -- only exact grouping -- so
    whitening categorical knobs would distort the factor structure without
    buying anything.  It also sidesteps the circularity of feeding
    clustering labels back in as client labels.

    Recursion.  At every level the current subset is grouped by the next
    feature; the ``n_partitions`` budget is distributed across the
    resulting groups proportionally to group size, and each group recurses
    with its share of the budget.  Recursion stops when a subset only
    needs one partition, or when the feature list is exhausted.

    When a level has more distinct values than the remaining budget,
    adjacent values (in ``sort=True`` order) are merged into contiguous
    bins to hit the budget exactly.  This keeps neighbouring feature
    values on the same client, which is the natural reading of
    "hierarchical by importance".

    The leaves define one group per client; the FedChip ownership +
    Dirichlet spillover stage inherited from
    :class:`~partitioning.base.FedChipPartitioner` then leaks
    ``1 - cluster_share`` of every group across the other clients, so the
    feature boundaries stay soft (as in the K-means / clustering schemes)
    rather than producing perfectly disjoint silos.

    Args:
        n_partitions: Number of clients.
        features: Ordered list of columns; earlier entries dominate the
            top-level splits.  Combined cardinality must be large enough
            to reach ``n_partitions`` non-empty leaves.
        cluster_share: Fraction of each feature group kept by its owner
            client.  ``1.0`` disables the spillover and gives a hard,
            fully deterministic feature split.  Default ``0.8``.
        dirichlet_alpha: Concentration of the Dirichlet redistributing the
            leftover ``1 - cluster_share``.  Default ``0.5``.
        random_state: Seed for the spillover RNG.
    """

    def __init__(
        self,
        n_partitions: int,
        features: Sequence[str],
        cluster_share: float = 0.8,
        dirichlet_alpha: float = 0.5,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            n_partitions,
            cluster_share=cluster_share,
            dirichlet_alpha=dirichlet_alpha,
            random_state=random_state,
        )
        if not features:
            raise ValueError("features must contain at least one column name")
        self.features = list(features)
        self.stratify_cols: List[str] = list(self.features)

    @staticmethod
    def _allocate_budget(sizes: np.ndarray, k: int) -> np.ndarray:
        """Hand each group at least one partition, k in total, size-weighted."""
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
    ) -> List[np.ndarray]:
        """Return one array of row positions per leaf of the split tree.

        *sub* must keep the positional index of the DataFrame passed to
        :meth:`_assign_groups`, so leaves can be mapped back to rows.
        """
        if k <= 1 or not remaining_features or len(sub) == 0:
            return [sub.index.to_numpy()]

        feature = remaining_features[0]
        rest = remaining_features[1:]

        groups = [g for _, g in sub.groupby(feature, sort=True)]
        m = len(groups)

        # Constant column at this level: it carries no information here,
        # so drop straight to the next feature without spending budget.
        if m == 1:
            return self._split(sub, rest, k)

        # More distinct values than budget: pack adjacent values into k
        # contiguous bins instead of recursing any further.
        if m >= k:
            bins = np.array_split(np.arange(m), k)
            return [
                np.concatenate([groups[j].index.to_numpy() for j in bin_idx])
                for bin_idx in bins
                if len(bin_idx) > 0
            ]

        sizes = np.array([len(g) for g in groups], dtype=float)
        alloc = self._allocate_budget(sizes, k)
        out: List[np.ndarray] = []
        for group, k_sub in zip(groups, alloc):
            out.extend(self._split(group, rest, int(k_sub)))
        return out

    def _reconcile(self, leaves: List[np.ndarray]) -> List[np.ndarray]:
        """Force exactly ``n_partitions`` leaves.

        The recursion under-produces when the feature list runs out before
        the budget does (a leaf owed >1 partition has nothing left to split
        on), and cannot over-produce -- but both directions are handled so
        the group ids stay in ``[0, n_partitions)``.
        """
        leaves = sorted(leaves, key=len, reverse=True)
        while len(leaves) > self.n_partitions:
            smallest = leaves.pop()
            leaves[-1] = np.concatenate([leaves[-1], smallest])
            leaves = sorted(leaves, key=len, reverse=True)

        if len(leaves) < self.n_partitions:
            warnings.warn(
                f"features={self.features} only yielded {len(leaves)} distinct "
                f"groups for n_partitions={self.n_partitions}; the remaining "
                f"{self.n_partitions - len(leaves)} client(s) start empty and are "
                "filled by the Dirichlet spillover alone. Add a feature, or lower "
                "n_partitions, to give every client an owned group "
                "(with cluster_share=1.0 they would stay empty).",
                stacklevel=4,  # _reconcile -> _assign_groups -> partition -> caller
            )
            while len(leaves) < self.n_partitions:
                leaves.append(np.empty(0, dtype=int))
        return leaves

    def _assign_groups(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"features missing from DataFrame: {missing}. "
                f"Available: {list(df.columns)}"
            )

        leaves = self._reconcile(self._split(df, list(self.features), self.n_partitions))

        group_ids = np.full(len(df), -1, dtype=int)
        for gid, positions in enumerate(leaves):
            group_ids[positions] = gid
        return group_ids
