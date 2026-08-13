"""Torch ``Dataset`` bridging a partition metadata DataFrame to CircuitNet-N28
feature/label ``.npy`` tensors on disk.

Mirrors the layout of the upstream ``code_examples/CircuitNet`` loader
(9 x H x W feature stack, single-channel DRC label) but consumes a
DataFrame of filenames instead of a two-column CSV so it composes
directly with the :mod:`partitioning` module.
"""

from __future__ import annotations

import os.path as osp
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DRCDataset(Dataset):
    """CircuitNet-N28 DRC-hotspot dataset backed by a partition DataFrame.

    Args:
        df: DataFrame with at least a ``filename`` column, one row per sample.
        feature_dir: Directory containing per-sample feature ``.npy`` files.
        label_dir: Directory containing per-sample label ``.npy`` files.
        filename_col: Column of *df* to read sample identifiers from.
        extension: File suffix appended to each identifier to form the
            on-disk path (default ``.npy``).  Set to ``""`` if the column
            already contains full filenames.
        return_path: If True, ``__getitem__`` returns ``(feature, label,
            label_path)``; otherwise just ``(feature, label)``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: str,
        label_dir: str,
        filename_col: str = "filename",
        extension: str = ".npy",
        return_path: bool = False,
    ) -> None:
        if filename_col not in df.columns:
            raise ValueError(
                f"DataFrame missing required column {filename_col!r}."
            )

        self.filenames = df[filename_col].astype(str).tolist()
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.extension = extension
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.filenames)

    def _paths(self, idx: int) -> tuple[str, str]:
        name = self.filenames[idx] + self.extension
        return osp.join(self.feature_dir, name), osp.join(self.label_dir, name)

    def __getitem__(self, idx: int):
        feature_path, label_path = self._paths(idx)
        feature = np.load(feature_path).transpose(2, 0, 1).astype(np.float32)
        label = np.load(label_path).transpose(2, 0, 1).astype(np.float32)

        feature_t = torch.from_numpy(feature)
        label_t = torch.from_numpy(label)

        if self.return_path:
            return feature_t, label_t, label_path
        return feature_t, label_t
