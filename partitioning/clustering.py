from abc import abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .base import DatasetPartitioner


class _ClusteringPartitioner(DatasetPartitioner):
    """Shared preprocessing + PCA + FedChip-style Dirichlet spillover.

    Preprocessing pipeline matches ``feature_axis_validation_N28.ipynb``:

    1. One-hot encode categorical (non-numeric) columns via
       :func:`pd.get_dummies` (``dtype='uint8'``).
    2. Map every column to a Gaussian marginal with
       :class:`~sklearn.preprocessing.QuantileTransformer`
       (``output_distribution='normal'``).
    3. Re-center / rescale with :class:`~sklearn.preprocessing.StandardScaler`.
    4. Project onto all principal components (no whitening) with
       :class:`~sklearn.decomposition.PCA`; clustering then operates on the
       full PC matrix ``Z`` (the reference notebook's ``Z3 = Z``).

    Partitioning follows the FedChip recipe (Ashkboos et al., 2024):
    each cluster ``c`` gives ``cluster_share`` (default 80 %) of its samples
    to its owner client ``c``; the remaining ``1 - cluster_share`` (default
    20 %) is redistributed across all ``n_partitions`` clients according to a
    fresh :class:`Dirichlet(dirichlet_alpha)` draw. The final assignment is a
    proper non-overlapping partition of the input rows.

    Args:
        n_partitions: Number of clients (== number of clusters).
        features: Columns of the input DataFrame used to build the
            clustering matrix. ``None`` (default) uses every column except
            the identifiers ``filename`` and ``sample_id``.
        cluster_share: Fraction of each cluster kept by its owner. Must lie
            in ``[0, 1]``. Default ``0.8``.
        dirichlet_alpha: Concentration of the Dirichlet used to redistribute
            the leftover ``1 - cluster_share`` fraction. Smaller ⇒ more
            skew. Default ``0.5``.
        random_state: Seed for PCA/K-means/QuantileTransformer and the
            Dirichlet RNG.
    """

    _IDENTIFIER_COLS = ("filename", "sample_id")

    def __init__(
        self,
        n_partitions: int,
        features: Optional[Sequence[str]] = None,
        cluster_share: float = 0.8,
        dirichlet_alpha: float = 0.5,
        random_state: int = 42,
        **kwargs,
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
        self.features = list(features) if features is not None else None
        self.cluster_share = cluster_share
        self.dirichlet_alpha = dirichlet_alpha
        self.random_state = random_state
        self.stratify_cols: List[str] = list(self.features or [])

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        if self.features is not None:
            missing = [c for c in self.features if c not in df.columns]
            if missing:
                raise ValueError(
                    f"features missing from DataFrame: {missing}. "
                    f"Available: {list(df.columns)}"
                )
            cols = list(self.features)
        else:
            cols = [c for c in df.columns if c not in self._IDENTIFIER_COLS]
        if not cols:
            raise ValueError("No feature columns available for clustering.")
        return cols

    def _prepare_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = self._select_features(df)
        self.stratify_cols = list(cols)
        sub = df[cols].copy()

        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(sub[c])]
        if cat_cols:
            sub = pd.get_dummies(sub, columns=cat_cols, dtype="uint8")

        X = sub.to_numpy(dtype=float)

        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(X)),
            random_state=self.random_state,
        )
        X = qt.fit_transform(X)

        X = StandardScaler().fit_transform(X)

        n_comp = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        return pca.fit_transform(X)

    @abstractmethod
    def _cluster(self, X: np.ndarray) -> np.ndarray:
        """Return an integer cluster id in ``[0, n_partitions)`` per row."""

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

        X = self._prepare_matrix(df)
        cluster_ids = self._cluster(X)
        K = self.n_partitions
        rng = np.random.default_rng(self.random_state)
        assignment = np.full(len(df), -1, dtype=int)

        for c in range(K):
            positions = np.where(cluster_ids == c)[0]
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

        return [
            df[assignment == i].reset_index(drop=True) for i in range(K)
        ]


class KMeansClustering(_ClusteringPartitioner):
    """FedChip partitioning with K-means clustering (Lloyd's algorithm)."""

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        model = KMeans(
            n_clusters=self.n_partitions,
            n_init=10,
            random_state=self.random_state,
        )
        return model.fit_predict(X)


class HierarchicalClustering(_ClusteringPartitioner):
    """FedChip partitioning with Ward-linkage hierarchical clustering."""

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        Z_link = linkage(X, method="ward")
        labels = fcluster(Z_link, t=self.n_partitions, criterion="maxclust")
        return labels - 1
