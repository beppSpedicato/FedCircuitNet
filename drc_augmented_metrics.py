
from __future__ import annotations

import os
import os.path as osp
import sys
from functools import partial
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
from utils import build_metric  # noqa: E402
from utils.device import features_to_device, resolve_device  # noqa: E402

METRICS = ['NRMS', 'NRMS_design_with_violations', 'NRMS_nonzero', 'MAE', 'MAE_design_with_violations', 'MAE_nonzero']
# Reported in DRC violations per cell instead of normalized label units.
SCALED_METRICS = ['MAE', 'MAE_design_with_violations', 'MAE_nonzero']


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

    metrics = {name: build_metric(name) for name in METRICS}
    for name in SCALED_METRICS:
        metrics[name] = partial(metrics[name], label_scale=label_scale)
    avg_metrics: Dict[str, float] = {name: 0.0 for name in METRICS}
    n_defined: Dict[str, int] = {name: 0 for name in METRICS}

    print(f"===> Iterating {len(dataset)} samples")
    rows = []
    with torch.no_grad():
        with tqdm(total=len(loader), desc="test") as bar:
            for feature, label, label_path in loader:
                feature, label = features_to_device(feature, label, runtime_cfg)

                prediction = model(feature)

                row = {
                    'sample': osp.splitext(osp.basename(label_path[0]))[0],
                    'violating_cells': int((label > 0).sum()),
                }
                for metric_name, metric_fn in metrics.items():
                    metric_v = float(metric_fn(label.cpu(), prediction.squeeze(1).cpu()))
                    row[metric_name] = metric_v
                    # NaN = undefined on this sample: kept in the CSV, left out of the average
                    if not np.isnan(metric_v):
                        avg_metrics[metric_name] += metric_v
                        n_defined[metric_name] += 1
                        run.track(
                            value=metric_v,
                            name=f"Test {metric_name} (dist)",
                            context={'subset': 'test', 'aggregation': 'distribution'},
                        )
                rows.append(row)

                bar.update(1)

    df = pd.DataFrame(rows)
    os.makedirs(save_path, exist_ok=True)
    csv_path = osp.join(save_path, 'augmented_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"===> Per-sample metrics saved to {csv_path}")

    n_with_violations = int((df['violating_cells'] > 0).sum())
    print(f"===> Samples with at least one violation: {n_with_violations}/{len(df)}")
    run.track(len(df), name='Test samples', context={'subset': 'test'})
    run.track(n_with_violations, name='Test samples with violations', context={'subset': 'test'})

    for metric_name, total in avg_metrics.items():
        n = n_defined[metric_name]
        avg = total / n if n else float("nan")
        print(f"===> Avg. {metric_name}: {avg:.4f} (over {n} samples)")
        run.track(avg, name=f"Test Avg {metric_name}", context={'subset': 'test'})


if __name__ == "__main__":
    evaluate()
