"""FedAvg checkpoint evaluation with Hydra config + Aim tracking.

1:1 port of ``code_examples/CircuitNet/drc_prediction/test.py``: the
metrics, ROC / PR-AUC sweep and confusion counts are computed by the
same code, shared via the :mod:`test_utils` package.  Only the config
plumbing (nested Hydra layout) and the sample-loader source (a
partition metadata DataFrame instead of an ann_file CSV) differ.

Aim ``Run`` receives:
    - ``hparams`` -- the full resolved Hydra config
    - per-sample ``Test <MetricName> (dist)`` scalars for every metric
      in ``evaluation.eval_metric`` (skipping the sentinel value 1.0,
      same as the reference)
    - aggregate ``Avg <MetricName>`` scalars, and -- when
      ``evaluation.plot_roc`` is set -- ROC-AUC, PR-AUC (via the
      multi-threshold CSV sweep), TP/TN/FP/FN, accuracy and per-i
      ROC_TPR / ROC_FPR curve points.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from utils.metrics import build_roc_prc_metric
import numpy as np
import omegaconf
import pandas as pd
import torch
from aim import Run
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.routenet_groupnorm import RouteNetGroupNorm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.drc_dataset import DRCDataset  # noqa: E402
from utils.device import features_to_device, resolve_device  # noqa: E402
from models.routenet import RouteNet  # noqa: E402
from utils import build_metric, roc_prc, multi_process_score, set_random_seed  # noqa: E402


MODEL_REGISTRY: Dict[str, type] = {
    "RouteNet": RouteNet,
    "RouteNetGroupNorm": RouteNetGroupNorm
}


def _load_model(
    model_cfg: Dict[str, Any],
    checkpoint: str,
    device: torch.device,
) -> torch.nn.Module:
    mtype = model_cfg["type"]
    if mtype not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.type={mtype!r}; known: {list(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[mtype]
    model = cls(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="./config", config_name="fedavg_test")
def test(CFG: omegaconf.DictConfig) -> None:
    resolved: Dict[str, Any] = omegaconf.OmegaConf.to_container(CFG, resolve=True)  # type: ignore[assignment]

    run = Run(experiment=resolved.get("experiment", "fedavg_test"))
    run["hparams"] = resolved

    data_cfg = resolved["data"]
    model_cfg = resolved["model"]
    runtime_cfg = resolved.get("runtime", {})
    evaluation_cfg = resolved.get("evaluation") or {}
    strategy_tag = (resolved.get("strategy") or {}).get("type")
    checkpoint = resolved["checkpoint"]

    device = resolve_device(runtime_cfg)
    print(f"===> Using device: {device}")
    threshold = float(evaluation_cfg.get("threshold", 0.1))
    plot_roc = bool(evaluation_cfg.get("plot_roc", False))
    save_path = evaluation_cfg["save_path"]
    metric_names = list(evaluation_cfg.get("eval_metric", []))

    seed = runtime_cfg.get("seed")
    if seed is not None:
        set_random_seed(int(seed))

    os.makedirs(save_path, exist_ok=True)

    print("===> Loading metadata")
    metadata_df = pd.read_csv(data_cfg["metadata_csv"])
    print(f"     {len(metadata_df)} test samples loaded from {data_cfg['metadata_csv']}")

    # Reference test.py forces batch_size=1 + shuffle=False and returns the
    # label path per sample; mirror that here.
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

    print(f"===> Loading checkpoint {checkpoint}")
    model = _load_model(model_cfg, checkpoint, device)

    metrics = {name: build_metric(name) for name in metric_names}
    avg_metrics: Dict[str, float] = {name: 0.0 for name in metric_names}

    print(f"===> Iterating {len(dataset)} samples")
    with torch.no_grad():
        with tqdm(total=len(loader), desc="test") as bar:
            for feature, label, label_path in loader:
                feature, label = features_to_device(feature, label, runtime_cfg)

                prediction = model(feature)

                for metric_name, metric_fn in metrics.items():
                    metric_v = metric_fn(label.cpu(), prediction.squeeze(1).cpu())
                    if metric_v != 1:
                        metric_v = float(metric_v)
                        avg_metrics[metric_name] += metric_v
                        run.track(
                            value=metric_v,
                            name=f"Test {metric_name} (dist)",
                            context={"subset": "test", "aggregation": "distribution"},
                        )

                if plot_roc:
                    result_dir = osp.join(save_path, "test_result")
                    os.makedirs(result_dir, exist_ok=True)
                    file_name = osp.splitext(osp.basename(label_path[0]))[0]
                    out_file = osp.join(result_dir, f"{file_name}.npy")
                    output_final = prediction.float().detach().cpu().numpy()
                    np.save(out_file, output_final)

                bar.update(1)

    summary: Dict[str, Any] = {
        "strategy": strategy_tag,
        "num_samples": len(dataset),
        "threshold": threshold,
    }

    for metric_name, total in avg_metrics.items():
        avg = total / len(dataset) if len(dataset) else float("nan")
        print(f"===> Avg. {metric_name}: {avg:.4f}")
        summary[f"avg_{metric_name.lower()}"] = avg
        run.track(avg, name=f"Avg {metric_name}", context={"subset": "test"})

    if plot_roc:
        parent_dir = data_cfg["feature_dir"].rstrip("/").rsplit("/", 1)[0]
        roc_metric, prc_numerator = build_roc_prc_metric(threshold=threshold, dataroot=parent_dir, ann_file=data_cfg["ann_file"], save_path=save_path)
        csv_file = osp.join(save_path, 'roc_prc.csv')
        df = pd.read_csv(csv_file, header=None, names=["threshold", "id", "tn", "fp", "fn", "tp"])
        t = df
        no_negatives      = (t["fp"] == 0) & (t["tn"] == 0)
        no_positives      = (t["tp"] == 0) & (t["fn"] == 0)
        no_pred_positives = (t["tp"] == 0) & (t["fp"] == 0)
        df = t[~(no_negatives | no_positives | no_pred_positives)]
        df["pos"] = df["tp"] + df["fn"]
        df["neg"] = df["fp"] + df["tn"]
        valid = df[(df["pos"] > 0) & (df["neg"] > 0)].copy()
        valid["tpr"] = valid["tp"] / valid["pos"]
        valid["fpr"] = valid["fp"] / valid["neg"]
        macro = valid.groupby("threshold")[["fpr", "tpr"]].mean()
        df_filtered = valid[valid["threshold"] == 0.1].copy()
        accuracy = (df_filtered['tp'].sum() + df_filtered['tn'].sum()) / (df_filtered['tp'].sum() + df_filtered['tn'].sum() + df_filtered['fp'].sum() + df_filtered['fn'].sum())
        precision = df_filtered['tp'].sum() / (df_filtered['tp'].sum() + df_filtered['fp'].sum())

        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))
        print("===> Precision: {:.4f}".format(precision))
        print(f"===> Accuracy @ score>={CFG['threshold']}: {accuracy:.4f}")
        print("===> PRC numerator: {:.4f}".format(prc_numerator))

        run.track(accuracy, name='Test Accuracy', context={'subset': 'test'})
        run.track(precision, name='Test Precision', context={'subset': 'test'})
        run.track(df_filtered['tp'].sum(), name='Confusion TP', context={'subset': 'test'})
        run.track(df_filtered['tn'].sum(), name='Confusion TN', context={'subset': 'test'})
        run.track(df_filtered['fp'].sum(), name='Confusion FP', context={'subset': 'test'})
        run.track(df_filtered['fn'].sum(), name='Confusion FN', context={'subset': 'test'})

        for i in range(len(macro['tpr'])):
            run.track(macro['tpr'].iloc[i], name='ROC_TPR', step=i, context={'type': 'curve'})
        for i in range(len(macro['fpr'])):
            run.track(macro['fpr'].iloc[i], name='ROC_FPR', step=i, context={'type': 'curve'})
        
    run["summary"] = summary

    metrics_path = osp.join(save_path, "test_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    run.log_info(f"Metrics saved to {metrics_path}")
    print(f"===> Metrics saved to {metrics_path}")


if __name__ == "__main__":
    test()
