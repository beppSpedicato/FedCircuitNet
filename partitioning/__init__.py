from typing import Any, Dict

from .base import DatasetPartitioner
from .clustering import HierarchicalClustering, KMeansClustering
from .iid import IIDPartitioner

PARTITIONER_REGISTRY: Dict[str, type] = {
    "iid": IIDPartitioner,
    "kmeans": KMeansClustering,
    "hierarchical": HierarchicalClustering,
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
    "DatasetPartitioner",
    "HierarchicalClustering",
    "IIDPartitioner",
    "KMeansClustering",
]
