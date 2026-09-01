"""FedAvg (and future strategies) training with Hydra config + Aim tracking.

The Hydra config is grouped into ``data`` / ``partitioning`` / ``model``
/ ``strategy`` / ``runtime`` / ``training`` / ``evaluation`` sections.
The ``strategy.type`` field selects an entry in
:data:`federated.strategies.STRATEGY_REGISTRY` (mirroring the
``federated/strategies/<id>/`` package layout); the rest of that block
is forwarded to the corresponding factory.  Adding a new strategy
requires no changes to this script -- register the factory in
``federated/strategies/__init__.py`` and switch ``strategy.type``.

The Aim ``Run`` receives:
    - ``hparams`` -- the full resolved Hydra config
    - per-round scalars: mean client train loss, round wall-clock,
      aggregation wall-clock
    - per-client scalars: local train loss (tagged with ``client_id``)
    - optional per-round validation scalars when ``evaluation.metadata_csv``
      is set: pixel MSE, ROC-AUC, PR-AUC

Usage
-----
    python train_aim.py --config-name=fedavg_train \\
        data.metadata_csv=/abs/path/train.csv \\
        data.feature_dir=/abs/path/feature \\
        data.label_dir=/abs/path/label \\
        strategy.local_epochs=2

Any nested field is Hydra-overridable via dotted syntax.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from aim import Distribution, Run
from torch.utils.data import DataLoader
from federated.base import StateDict
from federated.statistics import RoundStats
from models import _build_model_fn
from partitioning import _build_partitioner
from utils.metrics import build_metric
from utils.seed import set_random_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets import _build_dataset_fn
from utils.device import features_to_device, resolve_device
from federated.strategies import build_strategy
from utils.losses import build_loss


def _track_partition_distribution(
    run: Run,
    partitioner,
    metadata_df: pd.DataFrame,
) -> None:
    partitions = partitioner.partition(metadata_df)
    partitions_len = [len(p) for p in partitions]
    run.track(
        Distribution.from_histogram(partitions_len, (0, len(partitions))),
        name="partitions",
        step=0,
    )

    stratify_cols = getattr(partitioner, "stratify_cols", None) or [
        c for c in metadata_df.columns if c != "filename"
    ]
    for col in stratify_cols:
        if col not in metadata_df.columns:
            continue
        unique_vals = sorted(
            {v for part in partitions for v in part[col].astype(str).unique()}
        )
        n_vals = len(unique_vals)
        if n_vals == 0:
            continue
        for i, part in enumerate(partitions):
            counts = (
                part[col]
                .astype(str)
                .value_counts()
                .reindex(unique_vals, fill_value=0)
                .to_numpy()
            )
            run.track(
                Distribution.from_histogram(counts, (0, n_vals)),
                name=f"partitions/{col}",
                context={"partition_id": int(i)},
                step=0,
            )

def _evaluate_global_model(
    model: nn.Module,
    loader: DataLoader,
    runtime_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any]
) -> Dict[str, float]:
    """Return pixel MSE + optional ROC-AUC / PR-AUC on the eval loader."""

    model.eval()
    loss_fn = build_loss(strategy_cfg)
    metrics = {k: build_metric(k) for k in strategy_cfg['eval_metric']}
    avg_metrics = {k: 0.0 for k in metrics.keys()}
    avg_loss = 0.0
    n = 0

    with torch.no_grad():
        for feature, label in loader:
            input, target = features_to_device(feature, label, runtime_cfg)

            prediction = model(input)
            avg_loss += loss_fn(prediction, target).item()

            pred_cpu = prediction.squeeze(1).detach().cpu()
            tgt_cpu = target.cpu()
            for name, fn in metrics.items():
                v = fn(tgt_cpu, pred_cpu)
                if v != 1:
                    avg_metrics[name] += float(v)

            n += 1
    if n > 0:
        avg_loss /= n
        for k in avg_metrics:
            avg_metrics[k] /= n
    
    model.train()
    return avg_loss, avg_metrics

def _build_round_callback(
    run: Run,
    eval_loader: Optional[DataLoader],
    eval_model: Optional[nn.Module],
    eval_freq: int,
    threshold: float,
    runtime_cfg: Dict[str, Any],
):
    """Compose the on_round_end callback: Aim tracking + optional eval.

    Periodic checkpointing lives in the strategy itself (``save_every`` /
    ``save_dir``) -- keep it out of the callback.
    """

    def on_round_end(stats: RoundStats, global_state: StateDict):
        for c in stats.client_stats:
            run.track(
                value=c.train_loss,
                name="Client Train Loss",
                context={"subset": "train", "client_id": int(c.client_id)},
                step=stats.round_idx,
            )
            run.track(
                value=c.local_update_time_s,
                name="Client Update Time (s)",
                context={"subset": "train", "client_id": int(c.client_id)},
                step=stats.round_idx,
            )

        losses = [c.train_loss for c in stats.client_stats]
        mean_loss = sum(losses) / len(losses) if losses else float("nan")
        run.track(
            mean_loss,
            name="Mean Client Train Loss",
            context={"subset": "train"},
            step=stats.round_idx,
        )
        run.track(
            stats.round_time_s,
            name="Round Time (s)",
            context={"subset": "train"},
            step=stats.round_idx,
        )
        run.track(
            stats.aggregation_time_s,
            name="Aggregation Time (s)",
            context={"subset": "train"},
            step=stats.round_idx,
        )

        if (
            eval_loader is not None
            and eval_model is not None
            and eval_freq > 0
            and stats.round_idx % eval_freq == 0
        ):
            eval_model.load_state_dict(global_state)
            val_loss, val_metrics = _evaluate_global_model(eval_model, eval_loader, runtime_cfg, threshold)
            stats.global_metrics = val_metrics
            run.track(val_loss, name=f"Val avg Loss", context={"subset": "val"}, step=stats.round_idx)
            for k, v in val_metrics.items():
                run.track(
                    value=v,
                    name=f"Val {k}",
                    context={'subset': 'val'},
                    step=stats.round_idx,
                )
            print(
                f"[val round {stats.round_idx}] "
                + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

        losses_str = f"mean_loss={mean_loss:.4f}" if losses else "no_updates"
        print(
            f"[round {stats.round_idx:04d}] "
            f"selected={stats.selected_client_ids} "
            f"{losses_str} "
            f"round={stats.round_time_s:.1f}s "
            f"agg={stats.aggregation_time_s:.2f}s"
        )

    return on_round_end


@hydra.main(version_base=None, config_path="./config", config_name="fedavg_train")
def train(CFG: omegaconf.DictConfig) -> None:
    resolved: Dict[str, Any] = omegaconf.OmegaConf.to_container(CFG, resolve=True)  # type: ignore[assignment]

    run = Run(experiment=resolved.get("experiment", "fedavg_train"))
    run["hparams"] = resolved

    data_cfg = resolved["data"]
    model_cfg = resolved["model"]
    partitioning_cfg = resolved["partitioning"]
    strategy_cfg = resolved["strategy"]
    runtime_cfg = resolved.get("runtime", {})
    training_cfg = resolved["training"]
    evaluation_cfg = resolved.get("evaluation") or {}

    set_random_seed(int(runtime_cfg.get("seed", 42)))

    device = resolve_device(runtime_cfg)
    print(f"===> Using device: {device}")

    save_path = training_cfg["save_path"]
    os.makedirs(save_path, exist_ok=True)

    print("===> Loading metadata")
    metadata_df = pd.read_csv(data_cfg["metadata_csv"])
    if "filename" not in metadata_df.columns:
        raise SystemExit(
            "data.metadata_csv must contain a 'filename' column; "
            f"got {list(metadata_df.columns)}"\
        )
    print(f"     {len(metadata_df)} samples loaded from {data_cfg['metadata_csv']}")

    partitioner = _build_partitioner(partitioning_cfg)
    print(
        f"===> Partitioner: {partitioner.__class__.__name__} "
        f"(n_partitions={partitioner.n_partitions})"
    )
    _track_partition_distribution(run, partitioner, metadata_df)

    print(f"===> Building strategy: {strategy_cfg['type']}")
    strategy = build_strategy(strategy_cfg, runtime_cfg)

    # Optional in-training evaluation loader
    eval_loader: Optional[DataLoader] = None
    eval_model: Optional[nn.Module] = None
    eval_csv = evaluation_cfg.get("metadata_csv")
    if eval_csv:
        print(f"===> Building eval loader from {eval_csv}")
        eval_df = pd.read_csv(eval_csv)
        eval_ds = _build_dataset_fn(data_cfg)(eval_df)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=int(evaluation_cfg.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
        )
        eval_model = _build_model_fn(model_cfg)().to(device)

    on_round_end = _build_round_callback(
        run=run,
        eval_loader=eval_loader,
        eval_model=eval_model,
        eval_freq=int(evaluation_cfg.get("freq_rounds", 0) or 0),
        threshold=float(evaluation_cfg.get("threshold", 0.1)),
        runtime_cfg=runtime_cfg,
    )

    num_rounds = int(training_cfg["num_rounds"])
    save_every = int(training_cfg.get("save_freq_rounds", 0) or 0)
    print(f"===> Running {num_rounds} rounds (save_every={save_every})")
    partition_sizes, stats = strategy.train(
        partitioner=partitioner,
        metadata_df=metadata_df,
        model_fn=_build_model_fn(model_cfg),
        dataset_fn=_build_dataset_fn(data_cfg),
        num_rounds=num_rounds,
        on_round_end=on_round_end,
        save_every=save_every,
        save_dir=save_path if save_every > 0 else None,
    )

    # Final artefacts
    final_ckpt = os.path.join(save_path, "global_model_final.pth")
    torch.save({"state_dict": strategy.global_state}, final_ckpt)
    run.log_info(f"Final model saved to {final_ckpt}")

    run["summary"] = {
        "strategy": strategy_cfg["type"],
        "partition_sizes": list(partition_sizes),
        "final_mean_train_loss": (
            float(np.mean([c.train_loss for c in stats.client_stats]))
        ),
    }

    print(f"===> Done. Partition sizes: {partition_sizes}, final mean train loss: {run['summary']['final_mean_train_loss']:.4f}")

if __name__ == "__main__":
    train()
