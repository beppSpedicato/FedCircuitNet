from turtle import pd
from typing import Any, Callable, Dict
import pandas as pd
from .drc_dataset import DRCDataset

def _build_dataset_fn(data_cfg: Dict[str, Any]) -> Callable[[pd.DataFrame], DRCDataset]:
    feature_dir = data_cfg["feature_dir"]
    label_dir = data_cfg["label_dir"]

    def _make(part_df: pd.DataFrame) -> DRCDataset:
        return DRCDataset(part_df, feature_dir=feature_dir, label_dir=label_dir)

    return _make

__all__ = ["DRCDataset", "_build_dataset_fn"]
