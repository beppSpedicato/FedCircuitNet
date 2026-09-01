from typing import Any, Dict

from .base import DatasetPartitioner, FedChipPartitioner
from .clustering import KMeansClustering
from .dirichlet import DirichletPartitioner
from .feature_hierarchical import FeatureHierarchicalPartitioner
from .iid import IIDPartitioner
from .preprocessing import (
    PREPROCESSOR_REGISTRY,
    MetadataPreprocessor,
    OrdinalRank,
    UtilizationRank,
    build_preprocessors,
)

PARTITIONER_REGISTRY: Dict[str, type] = {
    "iid": IIDPartitioner,
    "kmeans": KMeansClustering,
    "dirichlet": DirichletPartitioner,
    "feature_hierarchical": FeatureHierarchicalPartitioner,
}

def _build_partitioner(part_cfg: Dict[str, Any]):
    kwargs = dict(part_cfg)
    ptype = kwargs.pop("type")
    if ptype not in PARTITIONER_REGISTRY:
        raise ValueError(
            f"Unknown partitioning.type={ptype!r}; known: {list(PARTITIONER_REGISTRY)}"
        )
    return PARTITIONER_REGISTRY[ptype](**kwargs)

__all__ = [
    "_build_partitioner",
    "build_preprocessors",
    "DatasetPartitioner",
    "DirichletPartitioner",
    "FedChipPartitioner",
    "FeatureHierarchicalPartitioner",
    "IIDPartitioner",
    "KMeansClustering",
    "MetadataPreprocessor",
    "OrdinalRank",
    "PREPROCESSOR_REGISTRY",
    "UtilizationRank",
]
