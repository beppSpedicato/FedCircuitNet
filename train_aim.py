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
from typing import Any, Callable, Dict, Optional

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from aim import Distribution, Run
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.drc_dataset import DRCDataset  # noqa: E402
from federated.device import features_to_device, resolve_device  # noqa: E402
from federated.strategies import build_strategy  # noqa: E402
from models.routenet import RouteNet  # noqa: E402
from partitioning.iid import IIDPartitioner  # noqa: E402


PARTITIONER_REGISTRY: Dict[str, type] = {
    "iid": IIDPartitioner,
}

MODEL_REGISTRY: Dict[str, type] = {
    "RouteNet": RouteNet,
}


def _build_model_fn(model_cfg: Dict[str, Any]) -> Callable[[], nn.Module]:
    mtype = model_cfg["type"]
    if mtype not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.type={mtype!r}; known: {list(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[mtype]
    in_channels = int(model_cfg["in_channels"])
    out_channels = int(model_cfg["out_channels"])

    def _make() -> nn.Module:
        model = cls(in_channels=in_channels, out_channels=out_channels)
        model.init_weights()
        return model

    return _make


def _build_dataset_fn(data_cfg: Dict[str, Any]) -> Callable[[pd.DataFrame], DRCDataset]:
    feature_dir = data_cfg["feature_dir"]
    label_dir = data_cfg["label_dir"]

    def _make(part_df: pd.DataFrame) -> DRCDataset:
        return DRCDataset(part_df, feature_dir=feature_dir, label_dir=label_dir)

    return _make


def _build_partitioner(part_cfg: Dict[str, Any]):
    kwargs = dict(part_cfg)
    ptype = kwargs.pop("type")
    if ptype not in PARTITIONER_REGISTRY:
        raise ValueError(
            f"Unknown partitioning.type={ptype!r}; known: {list(PARTITIONER_REGISTRY)}"
        )
    return PARTITIONER_REGISTRY[ptype](**kwargs)


def _track_partition_distribution(
    run: Run,
    partitioner,
    metadata_df: pd.DataFrame,
) -> None:
    """Log per-partition sample counts and per-feature value distributions.

    Emitted metrics (step=0):
      * ``partitions``                 -- histogram of samples per partition.
      * ``partitions/<feature>``       -- one Distribution per partition (via
        ``context={"partition_id": i}``) of that feature's category counts,
        aligned to a shared, globally sorted category index so partitions
        can be overlaid in Aim.

    ``partitioner.partition`` is invoked here for the tracking snapshot;
    the strategy will invoke it again inside ``train()``.  This assumes
    the partitioner is deterministic under a fixed seed (true for
    :class:`IIDPartitioner`).
    """
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
        unique_vals = sorted(metadata_df[col].astype(str).unique().tolist())
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
    threshold: float,
) -> Dict[str, float]:
    """Return pixel MSE + optional ROC-AUC / PR-AUC on the eval loader."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0
    y_true_chunks: list = []
    y_pred_chunks: list = []

    with torch.no_grad():
        for feature, label in loader:
            feature, label = features_to_device(feature, label, runtime_cfg)
            prediction = model(feature)

            total_loss += float(loss_fn(prediction, label).item())
            n_batches += 1

            y_true_chunks.append(
                (label.cpu().numpy() >= threshold).astype(np.uint8).ravel()
            )
            y_pred_chunks.append(prediction.cpu().numpy().ravel())

    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([])
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([])

    metrics: Dict[str, float] = {
        "pixel_mse": total_loss / n_batches if n_batches else float("nan"),
    }
    if y_true.size and 0 < y_true.mean() < 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


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

    def on_round_end(stats, global_state):
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
            metrics = _evaluate_global_model(eval_model, eval_loader, runtime_cfg, threshold)
            stats.global_metrics = metrics
            for k, v in metrics.items():
                run.track(
                    v,
                    name=f"Val {k}",
                    context={"subset": "val"},
                    step=stats.round_idx,
                )
            print(
                f"[val round {stats.round_idx}] "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
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

    torch.manual_seed(int(runtime_cfg.get("seed", 42)))

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
        eval_ds = DRCDataset(
            eval_df,
            feature_dir=data_cfg["feature_dir"],
            label_dir=data_cfg["label_dir"],
        )
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
    history = strategy.train(
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

    history_json_path = os.path.join(save_path, "history.json")
    history.save_json(history_json_path)
    run.log_info(f"History JSON saved to {history_json_path}")

    history_csv_path = os.path.join(save_path, "history.csv")
    history.to_dataframe().to_csv(history_csv_path, index=False)
    run.log_info(f"History CSV saved to {history_csv_path}")

    run["summary"] = {
        "strategy": strategy_cfg["type"],
        "partition_sizes": list(history.partition_sizes),
        "num_rounds": len(history.rounds),
        "final_mean_train_loss": (
            float(np.mean([c.train_loss for c in history.rounds[-1].client_stats]))
            if history.rounds and history.rounds[-1].client_stats
            else None
        ),
    }

    print(f"===> Done. Partition sizes: {history.partition_sizes}")


if __name__ == "__main__":
    train()
