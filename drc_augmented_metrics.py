"""Additional test-set metrics for DRC violation prediction.

Port of ``code_examples/CircuitNet/drc_prediction/drc_augmented_metrics.py``:
the metrics and their Aim / CSV output are identical.  Only the config
plumbing (nested Hydra layout, as in test_aim.py) and the sample-loader
source (a metadata DataFrame instead of an ann_file CSV) differ.

For each test sample the trained model predicts a DRC map and four metrics are
computed against the label:

- NRMS:         RMSE over all cells / (label.max() - label.min()).
- NRMS_nonzero: RMSE over the cells with label > 0 only, same normalization as
                NRMS so the two are directly comparable.
- MAE:          mean |pred - label| over all cells, in DRC violations per cell.
                generate_training_set.py clips the DRC counts to 200 and
                divides by 200, so pred and label are multiplied by label_scale.
- MAE_nonzero:  as MAE, restricted to the cells with label > 0.

NRMS, NRMS_nonzero and MAE_nonzero are undefined on samples whose label has no
violation: those samples are excluded from the averages and counted instead.
Unlike the NRMS of test_aim.py, maps are not quantized to uint8 before computing it.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
from aim import Run
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.drc_dataset import DRCDataset  # noqa: E402
from models import load_model  # noqa: E402
from utils.device import resolve_device  # noqa: E402

METRICS = ['NRMS', 'NRMS_design_with_violations', 'NRMS_nonzero', 'MAE', 'MAE_design_with_violations', 'MAE_nonzero']


def drc_metrics(label, pred, label_scale):
    label = label.astype(np.float64).ravel()
    pred = pred.astype(np.float64).ravel()
    err = pred - label
    nonzero = label > 0
    label_range = label.max() - label.min()
    has_violations = label_range > 0 and nonzero.any()

    nan = float('nan')
    return {
        'NRMS': float(np.sqrt(np.mean(err ** 2)) / label_range),
        'NRMS_design_with_violations': float(np.sqrt(np.mean(err ** 2)) / label_range) if has_violations else nan,
        'NRMS_nonzero': float(np.sqrt(np.mean(err[nonzero] ** 2)) / label_range) if has_violations else nan,
        'MAE': float(np.mean(np.abs(err)) * label_scale),
        'MAE_design_with_violations': float(np.mean(np.abs(err)) * label_scale) if has_violations else nan,
        'MAE_nonzero': float(np.mean(np.abs(err[nonzero])) * label_scale) if has_violations else nan,
    }


@hydra.main(version_base=None, config_path="./config", config_name="fedavg_augmented_metrics_iid")
def evaluate(CFG: omegaconf.DictConfig) -> None:
    resolved: Dict[str, Any] = omegaconf.OmegaConf.to_container(CFG, resolve=True)  # type: ignore[assignment]

    run = Run(experiment=resolved.get("experiment", "fedavg_drc_augmented_metrics"))
    run['hparams'] = resolved
    if resolved.get('tag'):
        run.add_tag(resolved['tag'])

    data_cfg = resolved["data"]
    model_cfg = resolved["model"]
    runtime_cfg = resolved.get("runtime", {})
    evaluation_cfg = resolved.get("evaluation") or {}
    save_path = evaluation_cfg["save_path"]
    label_scale = float(evaluation_cfg.get('label_scale', 200))

    device = resolve_device(runtime_cfg)
    print(f"===> Using device: {device}")

    print('===> Loading datasets')
    metadata_df = pd.read_csv(data_cfg["metadata_csv"])
    dataset = DRCDataset(
        metadata_df,
        feature_dir=data_cfg["feature_dir"],
        label_dir=data_cfg["label_dir"],
        return_path=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
    )

    print(f"===> Loading checkpoint {resolved['checkpoint']}")
    model = load_model(model_cfg, resolved["checkpoint"], device)
    model.eval()

    rows = []
    with torch.no_grad():
        for feature, label, label_path in tqdm(loader):
            feature = feature.to(device)
            prediction = model(feature).cpu().numpy()
            label = label.numpy()

            values = drc_metrics(label, prediction, label_scale)
            for name, value in values.items():
                if not np.isnan(value):
                    run.track(
                        value=value,
                        name=f"Test {name} (dist)",
                        context={'subset': 'test', 'aggregation': 'distribution'},
                    )
            rows.append({
                'sample': osp.splitext(osp.basename(label_path[0]))[0],
                'violating_cells': int((label > 0).sum()),
                **values,
            })

    df = pd.DataFrame(rows)
    os.makedirs(save_path, exist_ok=True)
    csv_path = osp.join(save_path, 'augmented_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"===> Per-sample metrics saved to {csv_path}")

    n_with_violations = int(df['NRMS'].notna().sum())
    print(f"===> Samples with at least one violation: {n_with_violations}/{len(df)}")
    run.track(len(df), name='Test samples', context={'subset': 'test'})
    run.track(n_with_violations, name='Test samples with violations', context={'subset': 'test'})

    for name in METRICS:
        avg = float(df[name].mean())  # NaN (undefined) samples are skipped
        print("===> Avg. {}: {:.4f} (over {} samples)".format(name, avg, df[name].notna().sum()))
        run.track(avg, name=f"Test Avg {name}", context={'subset': 'test'})


if __name__ == "__main__":
    evaluate()
