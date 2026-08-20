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
from models.routenet import RouteNet  # noqa: E402
from test_utils import build_metric, roc_prc, multi_process_score, set_random_seed  # noqa: E402


MODEL_REGISTRY: Dict[str, type] = {
    "RouteNet": RouteNet,
}


def _load_model(model_cfg: Dict[str, Any], checkpoint: str, device: str) -> torch.nn.Module:
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

    device = runtime_cfg.get("device", "cpu")
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
    """ if "filename" not in metadata_df.columns:
        raise SystemExit(
            "data.metadata_csv must contain a 'filename' column; "
            f"got {list(metadata_df.columns)}"
        ) """
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
                feature = feature.to(device)
                label = label.to(device)

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
        # Multi-threshold ROC / PR-AUC sweep -- identical to the reference
        # build_roc_prc_metric plumbing, but with the label directory taken
        # straight from the config (we don't ship an ann_file).
        print("===> Running ROC / PR-AUC sweep (multi_process_score)")
        multi_process_score(
            out_name="roc_prc.csv",
            threshold=threshold,
            label_path=data_cfg["label_dir"],
            save_path=save_path,
        )
        (
            roc_auc,
            prc_numerator,
            tpr_list,
            fpr_list,
            precision_mean,
            accuracy_mean,
            tp,
            tn,
            fp,
            fn,
        ) = roc_prc(save_path)

        print(f"===> AUC of ROC. {roc_auc:.4f}")
        print(f"===> PRC numerator: {prc_numerator:.4f}")
        print(f"===> TPR mean: {sum(tpr_list)/len(tpr_list):.4f}")
        print(f"===> FPR mean: {sum(fpr_list)/len(fpr_list):.4f}")
        print(f"===> Precision: {precision_mean:.4f}")
        print(f"===> Accuracy: {accuracy_mean:.4f}")
        print(f"===> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        summary.update(
            {
                "roc_auc": float(roc_auc),
                "prc_numerator": float(prc_numerator),
                "precision_mean": float(precision_mean),
                "accuracy_mean": float(accuracy_mean),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

        run.track(accuracy_mean, name="Test Accuracy", context={"subset": "test"})
        run.track(precision_mean, name="Test Precision", context={"subset": "test"})
        run.track(float(roc_auc), name="Test ROC AUC", context={"subset": "test"})
        run.track(float(prc_numerator), name="Test PR AUC", context={"subset": "test"})
        run.track(int(tp), name="Confusion TP", context={"subset": "test"})
        run.track(int(tn), name="Confusion TN", context={"subset": "test"})
        run.track(int(fp), name="Confusion FP", context={"subset": "test"})
        run.track(int(fn), name="Confusion FN", context={"subset": "test"})

        for i in range(len(tpr_list)):
            run.track(tpr_list[i], name="ROC_TPR", step=i, context={"type": "curve"})
        for i in range(len(fpr_list)):
            run.track(fpr_list[i], name="ROC_FPR", step=i, context={"type": "curve"})

    run["summary"] = summary

    metrics_path = osp.join(save_path, "test_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    run.log_info(f"Metrics saved to {metrics_path}")
    print(f"===> Metrics saved to {metrics_path}")


if __name__ == "__main__":
    test()
