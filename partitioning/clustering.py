from abc import abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .base import FedChipPartitioner


class _ClusteringPartitioner(FedChipPartitioner):
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

    The resulting cluster ids are then handed to
    :meth:`~partitioning.base.FedChipPartitioner.partition`, which applies
    the FedChip ownership + Dirichlet-spillover stage shared with the other
    partitioners.

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
        super().__init__(
            n_partitions,
            cluster_share=cluster_share,
            dirichlet_alpha=dirichlet_alpha,
            random_state=random_state,
        )
        self.features = list(features) if features is not None else None
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

    def _assign_groups(self, df: pd.DataFrame) -> np.ndarray:
        return self._cluster(self._prepare_matrix(df))


class KMeansClustering(_ClusteringPartitioner):
    """FedChip partitioning with K-means clustering (Lloyd's algorithm)."""

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        model = KMeans(
            n_clusters=self.n_partitions,
            n_init=10,
            random_state=self.random_state,
        )
        return model.fit_predict(X)
