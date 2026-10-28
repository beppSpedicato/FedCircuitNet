"""Test metrics of every periodic checkpoint of the 20261012 client-sweep runs.

One-shot script, nothing is logged to Aim.  For each {iid, FH} x K = 5 / 10 /
20 run, the paired ``20261012_configs/fedavg_{ID}_test.yaml`` supplies the
model, the test set and -- through ``checkpoint`` -- the run's work dir.  Each
of its ``model_round_{0400,0800,1200,1600}.pth`` checkpoints goes through the
``test_aim.py`` evaluation, written to ``{dest}/{ID}/round_{NNNN}/``:

1. central pass over the whole test CSV: NRMS / SSIM averaged exactly as
   ``test_aim.py``'s ``Avg NRMS`` / ``Avg SSIM``, and every prediction saved
   to ``test_result/``;
2. ``build_roc_prc_metric`` on those predictions -> ``roc_prc.csv``;
3. ``test.ipynb``'s reduction of that ``roc_prc.csv``: Accuracy (pooled over
   every threshold), ROC-AUC and PR-AUC.  TPR @ FPR=5% interpolates the same
   ROC curve at 0.05, as ``show_sweep_clients_result.ipynb`` does.

The prediction maps (~0.8 GB per checkpoint) are removed once ``roc_prc.csv``
is written unless ``--keep-predictions`` is given.  The table is rewritten
after every checkpoint, so an interrupted sweep keeps what was finished.  A
missing checkpoint is reported and leaves its cells empty.

Usage:
    python test_round_sweep.py
    python test_round_sweep.py --dest ./work_dirs/20261012/round_sweep \
        --output reconstruction_res.csv
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.drc_dataset import DRCDataset  # noqa: E402
from models import load_model  # noqa: E402
from utils import build_metric  # noqa: E402
from utils.device import features_to_device, resolve_device  # noqa: E402
from utils.metrics import build_roc_prc_metric  # noqa: E402

RUN_IDS = [f"{scheme}_c{k}" for scheme in ("iid") for k in (5)]
ROUNDS = (400, 800, 1200, 1600)
IMAGE_METRICS = ("NRMS", "SSIM")
ROC_METRICS = ("Accuracy", "ROC-AUC", "TPR @ FPR=5%", "PR-AUC")
TARGET_FPR = 0.05


def _abs(path: str) -> str:
    """Config paths are relative to the repo root, not to the caller's cwd."""
    return path if osp.isabs(path) else osp.normpath(osp.join(_HERE, path))


@contextlib.contextmanager
def _quiet_stdout():
    """Silence fd 1, which multi_process_score's pool workers inherit under
    any start method (redirecting sys.stdout only reaches forked ones)."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def predict(
    model: torch.nn.Module,
    data_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    result_dir: str,
    desc: str,
) -> Dict[str, float]:
    """test_aim.py's test loop: Avg NRMS / SSIM, predictions saved to result_dir."""
    dataset = DRCDataset(
        pd.read_csv(_abs(data_cfg["metadata_csv"])),
        feature_dir=_abs(data_cfg["feature_dir"]),
        label_dir=_abs(data_cfg["label_dir"]),
        return_path=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
    )
    metric_fns = {name: build_metric(name) for name in IMAGE_METRICS}
    avg_metrics = {name: 0.0 for name in IMAGE_METRICS}

    with torch.no_grad():
        for feature, label, label_path in tqdm(loader, desc=desc):
            feature, label = features_to_device(feature, label, runtime_cfg)

            prediction = model(feature)

            for name, metric_fn in metric_fns.items():
                metric_v = metric_fn(label.cpu(), prediction.squeeze(1).cpu())
                # Same sentinel skip as test_aim.py; the denominator below
                # stays the full sample count, as there.
                if metric_v != 1:
                    avg_metrics[name] += float(metric_v)

            file_name = osp.splitext(osp.basename(label_path[0]))[0]
            out_file = osp.join(result_dir, f"{file_name}.npy")
            np.save(out_file, prediction.float().detach().cpu().numpy())

    return {name: total / len(dataset) for name, total in avg_metrics.items()}


def roc_metrics(csv_file: str) -> Dict[str, float]:
    """test.ipynb's reduction of one roc_prc.csv (cell 0, plus cell 1's PR-AUC)."""
    df = pd.read_csv(csv_file, header=None, names=["threshold", "id", "tn", "fp", "fn", "tp"])
    no_negatives = (df["fp"] == 0) & (df["tn"] == 0)
    no_positives = (df["tp"] == 0) & (df["fn"] == 0)
    no_pred_positives = (df["tp"] == 0) & (df["fp"] == 0)
    df = df[~(no_negatives | no_positives | no_pred_positives)].copy()
    if df.empty:
        return {name: math.nan for name in ROC_METRICS}

    tp, tn, fp, fn = (df[col].sum() for col in ("tp", "tn", "fp", "fn"))
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    df["tpr"] = df["tp"] / (df["tp"] + df["fn"])
    df["fpr"] = df["fp"] / (df["fp"] + df["tn"])
    df["precision"] = df["tp"] / (df["tp"] + df["fp"])
    macro = df.groupby("threshold")[["fpr", "tpr", "precision"]].mean()

    # ROC: thresholds high -> low, anchored at (0, 0) and (1, 1).
    roc = macro.sort_index(ascending=False)
    fpr = np.r_[0.0, roc["fpr"].to_numpy(), 1.0]
    tpr = np.r_[0.0, roc["tpr"].to_numpy(), 1.0]
    roc_auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    # The macro fpr need not be monotone in the threshold; np.interp needs
    # it sorted.
    order = np.argsort(fpr, kind="stable")
    tpr_at_fpr = np.interp(TARGET_FPR, fpr[order], tpr[order])

    # PR (recall = tpr): first threshold of each distinct recall, by
    # increasing recall.
    pts = macro.drop_duplicates("tpr").sort_values("tpr")
    if len(pts) > 3:  # the cubic spline needs at least 4 points
        x = np.linspace(0, 1, 25)
        y = make_interp_spline(pts["tpr"].to_numpy(), pts["precision"].to_numpy(), k=3)(x)
        pr_auc = np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)
    else:
        pr_auc = math.nan

    return {
        "Accuracy": float(accuracy),
        "ROC-AUC": float(roc_auc),
        "TPR @ FPR=5%": float(tpr_at_fpr),
        "PR-AUC": float(pr_auc),
    }


def write_table(results: Dict[Tuple[str, int], Dict[str, float]], output: str) -> pd.DataFrame:
    rows = []
    for run_id in RUN_IDS:
        row = {"run name": run_id}
        for name in IMAGE_METRICS + ROC_METRICS:
            for round_idx in ROUNDS:
                row[f"{name} {round_idx}"] = results.get((run_id, round_idx), {}).get(name, math.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-dir", default=str(_HERE / "20261012_configs"))
    parser.add_argument(
        "--dest",
        default=str(_HERE / "work_dirs" / "20261012" / "round_sweep"),
        help="roc_prc.csv of each checkpoint lands in {dest}/{run}/round_{NNNN}/",
    )
    parser.add_argument("--output", default=str(_HERE / "reconstruction_res.csv"))
    parser.add_argument("--keep-predictions", action="store_true",
                        help="keep the per-sample prediction maps under test_result/")
    parser.add_argument("--cpu", action="store_true", help="override runtime.cpu")
    args = parser.parse_args()

    cfgs = {}
    for run_id in RUN_IDS:
        with open(osp.join(args.config_dir, f"fedavg_{run_id}_test.yaml")) as fh:
            cfgs[run_id] = yaml.safe_load(fh)

    results: Dict[Tuple[str, int], Dict[str, float]] = {}
    for run_id, cfg in cfgs.items():
        data_cfg = cfg["data"]
        runtime_cfg = dict(cfg.get("runtime") or {})
        if args.cpu:
            runtime_cfg["cpu"] = True
        device = resolve_device(runtime_cfg)
        threshold = float((cfg.get("evaluation") or {}).get("threshold", 0.1))
        ckpt_dir = osp.dirname(_abs(cfg["checkpoint"]))

        for round_idx in ROUNDS:
            checkpoint = osp.join(ckpt_dir, f"model_round_{round_idx:04d}.pth")
            if not osp.isfile(checkpoint):
                print(f"WARNING: missing checkpoint {checkpoint} -- cells left empty")
                continue
            print(f"===> {run_id} round {round_idx}: {checkpoint} ({device})")

            save_path = osp.join(_abs(args.dest), run_id, f"round_{round_idx:04d}")
            result_dir = osp.join(save_path, "test_result")
            csv_file = osp.join(save_path, "roc_prc.csv")
            # multi_process_score sweeps every .npy under test_result, so maps
            # left by an earlier invocation must not survive; and it copies
            # roc_prc.csv into place with an unchecked `cp`, so a stale one
            # must not either.
            shutil.rmtree(result_dir, ignore_errors=True)
            if osp.exists(csv_file):
                os.remove(csv_file)
            os.makedirs(result_dir)

            model = load_model(cfg["model"], checkpoint, device)
            model.eval()
            values = predict(model, data_cfg, runtime_cfg, result_dir,
                             desc=f"{run_id} r{round_idx}")

            # Same call as test_aim.py; its per-threshold progress prints are
            # silenced.
            with _quiet_stdout():
                build_roc_prc_metric(
                    threshold=threshold,
                    dataroot=osp.dirname(_abs(data_cfg["feature_dir"])),
                    ann_file=_abs(data_cfg["ann_file"]),
                    save_path=save_path,
                )
            values.update(roc_metrics(csv_file))
            if not args.keep_predictions:
                shutil.rmtree(result_dir)

            results[(run_id, round_idx)] = values
            print("     " + ", ".join(f"{k} {v:.4f}" for k, v in values.items()))
            print(f"     roc_prc.csv -> {csv_file}")
            write_table(results, args.output)

    if not results:
        raise SystemExit("No checkpoint found for any run; nothing to evaluate.")
    print(write_table(results, args.output).to_string(index=False))
    print(f"===> Results saved to {args.output}")


if __name__ == "__main__":
    main()
