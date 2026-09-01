"""Metadata transforms applied to the partitioner's *view* of the dataset.

A partitioner reads the metadata table twice over: once to decide which
client every row belongs to, and once to produce the DataFrames it hands
back.  Everything downstream reports on the second one -- the per-partition
factor histograms in ``train_aim.py``, the JS-divergence / coverage tables
in ``partitioning_analysis.ipynb``.  The transforms here separate the two:
they rewrite the column used to *form* the groups while the returned rows
keep their raw values, so a client can be built on "high utilization" and
still be reported at 0.85 / 0.90.

Set ``keep_in_output: true`` on a transform to make it visible in the
returned partitions as well.

Selected through the ``partitioning.preprocessing`` config block, which
accepts a single entry or a list::

    partitioning:
      type: feature_hierarchical
      features: [design_name, clock, utilization]
      preprocessing:
        - type: rank_utilization

Contract for subclasses: :meth:`MetadataPreprocessor.apply` must preserve
row count *and* row order.  The working frame and the output frame are
aligned positionally by :class:`~partitioning.base.DatasetPartitioner`, so
a transform that reorders or drops rows would scramble the assignment.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping as _MappingABC
from collections.abc import Sequence as _SequenceABC
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class MetadataPreprocessor(ABC):
    """Base for a row-preserving transform of the metadata table.

    Args:
        keep_in_output: When ``False`` (default) the transform only shapes
            the grouping decision and the returned partitions carry the
            untouched original columns.  When ``True`` it is applied to the
            returned partitions too, so the transformed value is what every
            downstream report sees.
    """

    def __init__(self, keep_in_output: bool = False) -> None:
        self.keep_in_output = bool(keep_in_output)

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a transformed copy of *df* with the same rows, same order."""
        ...

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.apply(df)
        if len(out) != len(df):
            raise RuntimeError(
                f"{type(self).__name__} changed the row count "
                f"({len(df)} -> {len(out)}); preprocessors must preserve rows."
            )
        return out


class OrdinalRank(MetadataPreprocessor):
    """Collapse a discrete factor onto fewer, ordered levels, in place.

    The column keeps its name and gains an *ordered* ``Categorical`` dtype
    whose category order is the order the ranks are declared in.  That
    ordering is what the consumers rely on:

    - :class:`~partitioning.feature_hierarchical.FeatureHierarchicalPartitioner`
      groups with ``sort=True`` and packs adjacent values into contiguous
      bins, so the ranks must sort low -> high rather than alphabetically
      (which would give ``high, low, medium``).
    - :class:`~partitioning.clustering.KMeansClustering` maps ordered
      categoricals to their integer codes instead of one-hot columns, so a
      ranked factor keeps the single-column weight the raw numeric factor
      had -- one-hot would triple its pull on the PCA/K-means geometry.

    Values are matched exactly, after rounding floats to *decimals* places,
    so ``0.7`` and ``0.70`` collide.  A value no rank claims is an error
    rather than a silent ``NaN`` bucket: it means the level list and the
    corpus disagree, which would quietly misgroup clients.

    Args:
        column: Column to rewrite.  Must exist in the DataFrame.
        levels: Ordered mapping ``rank_name -> [raw values]``.  Declaration
            order is the rank order.  Every value present in the column
            must appear in exactly one rank.
        keep_in_output: See :class:`MetadataPreprocessor`.
        decimals: Rounding applied to numeric values before matching.
    """

    def __init__(
        self,
        column: str,
        levels: Mapping[str, Sequence[Any]],
        keep_in_output: bool = False,
        decimals: int = 6,
    ) -> None:
        super().__init__(keep_in_output=keep_in_output)
        if not levels:
            raise ValueError("levels must map at least one rank name to its values")
        self.column = str(column)
        self.decimals = int(decimals)
        # A one-value rank reads better as `medium: 0.80` than `medium: [0.80]`,
        # so accept the scalar form too.
        self.levels: Dict[str, List[Any]] = {
            str(rank): list(values)
            if isinstance(values, (list, tuple, set, _SequenceABC))
            and not isinstance(values, (str, bytes))
            else [values]
            for rank, values in levels.items()
        }

        self._lookup: Dict[Any, str] = {}
        for rank, values in self.levels.items():
            if not values:
                raise ValueError(f"rank {rank!r} lists no values")
            for value in values:
                key = self._match_key(value)
                if key in self._lookup:
                    raise ValueError(
                        f"value {value!r} is claimed by both "
                        f"{self._lookup[key]!r} and {rank!r}"
                    )
                self._lookup[key] = rank

    def _match_key(self, value: Any) -> Any:
        """Hashable key that unifies ``70``, ``70.0`` and ``0.7`` vs ``0.70``."""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return round(float(value), self.decimals)
        return str(value)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column not in df.columns:
            raise ValueError(
                f"preprocessing column {self.column!r} missing from DataFrame. "
                f"Available: {list(df.columns)}"
            )

        keys = df[self.column].map(self._match_key)
        unknown = sorted(
            {str(v) for v, k in zip(df[self.column], keys) if k not in self._lookup}
        )
        if unknown:
            raise ValueError(
                f"column {self.column!r} holds value(s) {unknown} that no rank in "
                f"{list(self.levels)} covers. Extend `levels` (N14 stores "
                "utilization as whole percents, N28 as a fraction) or fix the "
                "metadata CSV."
            )

        out = df.copy()
        out[self.column] = pd.Categorical(
            keys.map(self._lookup),
            categories=list(self.levels),
            ordered=True,
        )
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(column={self.column!r}, "
            f"levels={self.levels!r}, keep_in_output={self.keep_in_output})"
        )


class UtilizationRank(OrdinalRank):
    """The utilization ranking fixed in ``PARTITIONING_ANALYSIS.md`` §4.1.

    N28 utilization has five levels; ranking them low / medium / high cuts
    the ``design_name x utilization`` grid from 30 cells to 18, which is
    what makes a 20-client hierarchical split land on describable EDA
    personas ("area-driven = high utilization") instead of on five
    near-duplicate neighbours.

    Defaults are the N28 fractions.  N14 stores utilization as whole
    percents (50-75), so a cross-node run must pass its own ``levels``.
    """

    _DEFAULT_LEVELS: Dict[str, List[float]] = {
        "low": [0.70, 0.75],
        "medium": [0.80],
        "high": [0.85, 0.90],
    }

    def __init__(
        self,
        column: str = "utilization",
        levels: Optional[Mapping[str, Sequence[Any]]] = None,
        keep_in_output: bool = False,
        decimals: int = 6,
    ) -> None:
        super().__init__(
            column=column,
            levels=self._DEFAULT_LEVELS if levels is None else levels,
            keep_in_output=keep_in_output,
            decimals=decimals,
        )


PREPROCESSOR_REGISTRY: Dict[str, type] = {
    "ordinal_rank": OrdinalRank,
    "rank_utilization": UtilizationRank,
}


def build_preprocessors(cfg: Any) -> List[MetadataPreprocessor]:
    """Instantiate a ``partitioning.preprocessing`` block.

    Accepts ``None`` (no preprocessing), a single entry, or a list of
    entries.  An entry is a mapping whose ``type`` names a key of
    :data:`PREPROCESSOR_REGISTRY` and whose remaining keys are forwarded as
    kwargs, a bare string as shorthand for ``{"type": <string>}``, or an
    already-built :class:`MetadataPreprocessor`.
    """
    if cfg is None:
        return []
    if isinstance(cfg, (str, _MappingABC, MetadataPreprocessor)):
        cfg = [cfg]

    built: List[MetadataPreprocessor] = []
    for entry in cfg:
        if isinstance(entry, MetadataPreprocessor):
            built.append(entry)
            continue
        if isinstance(entry, str):
            entry = {"type": entry}
        if not isinstance(entry, _MappingABC):
            raise TypeError(
                "each preprocessing entry must be a mapping or a type name, "
                f"got {type(entry).__name__}"
            )
        kwargs = dict(entry)
        ptype = kwargs.pop("type", None)
        if ptype not in PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing.type={ptype!r}; "
                f"known: {list(PREPROCESSOR_REGISTRY)}"
            )
        built.append(PREPROCESSOR_REGISTRY[ptype](**kwargs))
    return built
