from abc import ABC, abstractmethod
from typing import List

from addict import Dict
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



