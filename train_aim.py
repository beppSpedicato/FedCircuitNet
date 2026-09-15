from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from utils.aim_tracking import build_round_callback, track_client_participation, track_communication_cost, track_partition_distribution
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from aim import Run
from torch.utils.data import DataLoader
from models import _build_model_fn
from partitioning import _build_partitioner
from utils.seed import set_random_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets import _build_dataset_fn
from utils.device import resolve_device
from federated.strategies import build_strategy



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
    partitions = partitioner.partition(metadata_df)
    track_partition_distribution(run, partitions, partitioner, metadata_df)

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

    on_round_end = build_round_callback(
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

    selection_counts = strategy.selection_counts
    download_mb = strategy.download_mb
    upload_mb = strategy.upload_mb
    track_client_participation(run, selection_counts, num_rounds)
    track_communication_cost(run, download_mb, upload_mb)

    run["summary"] = {
        "strategy": strategy_cfg["type"],
        "partition_sizes": list(partition_sizes),
        "round_clients": strategy.round_clients,
        "selection_counts": list(selection_counts),
        "download_mb": [round(v, 4) for v in download_mb],
        "upload_mb": [round(v, 4) for v in upload_mb],
        "total_comm_mb": round(sum(download_mb) + sum(upload_mb), 4),
        "final_mean_train_loss": (
            float(np.mean([c.train_loss for c in stats.client_stats]))
        ),
    }

    print(f"===> Done. Partition sizes: {partition_sizes}, final mean train loss: {run['summary']['final_mean_train_loss']:.4f}")

if __name__ == "__main__":
    train()
