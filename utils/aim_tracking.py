import pandas as pd
from aim import Distribution, Run
from federated.base import StateDict
from federated.statistics import RoundStats
from models import evaluate_global_model
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

def track_partition_distribution(
    run: Run,
    partitions,
    partitioner,
    metadata_df: pd.DataFrame,
) -> None:
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


def track_client_participation(
    run: Run,
    selection_counts: list,
    num_rounds: int,
) -> None:
    n_clients = len(selection_counts)
    if n_clients == 0:
        return

    run.track(
        Distribution.from_histogram(selection_counts, (0, n_clients)),
        name="client_rounds",
        step=0,
    )
    for client_id, count in enumerate(selection_counts):
        run.track(
            value=int(count),
            name="Client Rounds Participated",
            context={"subset": "train", "client_id": int(client_id)},
            step=0,
        )

    print(
        f"===> Participation over {num_rounds} rounds (per client): "
        f"min={min(selection_counts)} max={max(selection_counts)} "
        f"mean={sum(selection_counts) / n_clients:.1f}"
    )

def track_communication_cost(
    run: Run,
    download_mb: list,
    upload_mb: list,
) -> None:
    n_clients = len(download_mb)
    if n_clients == 0:
        return

    total_mb = [d + u for d, u in zip(download_mb, upload_mb)]
    series = {
        "download_cost": download_mb,
        "upload_cost": upload_mb,
        "total_comm_cost": total_mb,
    }
    for name, values in series.items():
        run.track(
            Distribution.from_histogram(values, (0, n_clients)),
            name=name,
            step=0,
        )

    scalars = {
        "Client Total Download (MB)": download_mb,
        "Client Total Upload (MB)": upload_mb,
        "Client Total Comm (MB)": total_mb,
    }
    for name, values in scalars.items():
        for client_id, value in enumerate(values):
            run.track(
                value=float(value),
                name=name,
                context={"subset": "train", "client_id": int(client_id)},
                step=0,
            )

    print(
        f"===> Comm cost: {sum(total_mb):.1f} MB total, "
        f"{sum(total_mb) / n_clients:.1f} MB mean per client "
        f"(min={min(total_mb):.1f} max={max(total_mb):.1f})"
    )


def build_round_callback(
    run: Run,
    eval_loader: Optional[DataLoader],
    eval_model: Optional[nn.Module],
    eval_freq: int,
    threshold: float,
    runtime_cfg: Dict[str, Any],
):

    def on_round_end(stats: RoundStats, global_state: StateDict):
        for c in stats.client_stats:
            client_ctx = {"subset": "train", "client_id": int(c.client_id)}
            run.track(
                value=c.train_loss,
                name="Client Train Loss",
                context=client_ctx,
                step=stats.round_idx,
            )
            run.track(
                value=c.local_update_time_s,
                name="Client Update Time (s)",
                context=client_ctx,
                step=stats.round_idx,
            )
            run.track(
                value=c.download_mb,
                name="Client Download (MB)",
                context=client_ctx,
                step=stats.round_idx,
            )
            run.track(
                value=c.upload_mb,
                name="Client Upload (MB)",
                context=client_ctx,
                step=stats.round_idx,
            )

        round_comm_mb = sum(
            c.download_mb + c.upload_mb for c in stats.client_stats
        )
        run.track(
            round_comm_mb,
            name="Round Comm Cost (MB)",
            context={"subset": "train"},
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
            val_loss, val_metrics = evaluate_global_model(eval_model, eval_loader, runtime_cfg, threshold)
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

