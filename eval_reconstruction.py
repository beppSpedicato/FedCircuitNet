from __future__ import annotations

import argparse
import math
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.drc_dataset import DRCDataset  # noqa: E402
from models import load_model  # noqa: E402
from utils import build_metric  # noqa: E402
from utils.device import features_to_device, resolve_device  # noqa: E402

RUN_IDS = [f"{scheme}_c{k}" for scheme in ("iid") for k in (5)]
ROUNDS = (400, 800, 1200, 1600)
METRICS = ("NRMS", "SSIM")
DATA_KEYS = ("metadata_csv", "feature_dir", "label_dir")


def _abs(path: str) -> str:
    """Config paths are relative to the repo root, not to the caller's cwd."""
    return path if osp.isabs(path) else osp.normpath(osp.join(_HERE, path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(_HERE / "20261012_configs"))
    parser.add_argument("--output", default=str(_HERE / "reconstruction_res.csv"))
    args = parser.parse_args()

    cfgs = {}
    for run_id in RUN_IDS:
        with open(osp.join(args.config_dir, f"fedavg_{run_id}_test.yaml")) as fh:
            cfgs[run_id] = yaml.safe_load(fh)

    data_cfg = cfgs[RUN_IDS[0]]["data"]
    for run_id, cfg in cfgs.items():
        if any(cfg["data"][k] != data_cfg[k] for k in DATA_KEYS):
            raise ValueError(
                f"{run_id} evaluates on a different test set than {RUN_IDS[0]}; "
                f"the single shared pass needs identical data.{DATA_KEYS}"
            )

    runtime_cfg = dict(cfgs[RUN_IDS[0]].get("runtime") or {})
    device = resolve_device(runtime_cfg)
    print(f"===> Using device: {device}")

    models: Dict[Tuple[str, int], torch.nn.Module] = {}
    for run_id, cfg in cfgs.items():
        ckpt_dir = osp.dirname(_abs(cfg["checkpoint"]))
        for round_idx in ROUNDS:
            path = osp.join(ckpt_dir, f"model_round_{round_idx:04d}.pth")
            if not osp.isfile(path):
                print(f"WARNING: missing checkpoint {path} -- cell left empty")
                continue
            model = load_model(cfg["model"], path, device)
            model.eval()
            models[(run_id, round_idx)] = model
    if not models:
        raise SystemExit("No checkpoint found for any run; nothing to evaluate.")
    print(f"===> Loaded {len(models)}/{len(RUN_IDS) * len(ROUNDS)} checkpoints")

    metadata_df = pd.read_csv(_abs(data_cfg["metadata_csv"]))
    dataset = DRCDataset(
        metadata_df,
        feature_dir=_abs(data_cfg["feature_dir"]),
        label_dir=_abs(data_cfg["label_dir"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
    )
    print(f"===> {len(dataset)} test samples from {data_cfg['metadata_csv']}")

    metric_fns = {name: build_metric(name) for name in METRICS}
    totals = {key: {name: 0.0 for name in METRICS} for key in models}

    with torch.no_grad():
        for feature, label in tqdm(loader, desc="test"):
            feature, label = features_to_device(feature, label, runtime_cfg)
            label_cpu = label.cpu()
            for key, model in models.items():
                prediction = model(feature).squeeze(1).cpu()
                for name, metric_fn in metric_fns.items():
                    metric_v = metric_fn(label_cpu, prediction)
                    # Same sentinel skip as test_aim.py; the denominator below
                    # stays the full sample count, as there.
                    if metric_v != 1:
                        totals[key][name] += float(metric_v)

    rows = []
    for run_id in RUN_IDS:
        row = {"run name": run_id}
        for name in METRICS:
            for round_idx in ROUNDS:
                key = (run_id, round_idx)
                row[f"{name} {round_idx}"] = (
                    totals[key][name] / len(dataset) if key in totals else math.nan
                )
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"===> Results saved to {args.output}")


if __name__ == "__main__":
    main()
