"""Abstract federated learning strategy.

The FL strategies in scope (FedAvg, FedProx, SCAFFOLD, FedBN, ...) all share
the same training skeleton -- partition metadata, build per-client loaders,
instantiate clients, run a synchronous round loop, collect per-round
statistics -- and differ in only two well-defined places:

    1. How each client transforms the broadcast weights into a local update.
    2. How the server merges the collected updates.

This module encodes that split.  :class:`FederatedStrategy` owns the pipeline
in :meth:`train` and exposes two abstract hooks that concrete strategies
must implement:

    * :meth:`_build_client` -- returns a :class:`FederatedClient`
    * :meth:`_build_aggregator` -- returns an :class:`Aggregator`

Adding a new strategy is therefore: subclass, declare the strategy-specific
hyperparameters in ``__init__``, and implement the two hooks.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import Aggregator, FederatedClient, StateDict
from .server import FederatedServer
from .statistics import RoundStats, TrainingHistory

if TYPE_CHECKING:
    import pandas as pd

    from partitioning.base import DatasetPartitioner


DatasetFn = Callable[["pd.DataFrame"], Dataset]
ModelFn = Callable[[], nn.Module]
GlobalEvalFn = Callable[[nn.Module, DataLoader, str], Dict[str, float]]
RoundEndCallback = Callable[[RoundStats, StateDict], None]


class FederatedStrategy(ABC):
    """Template for a synchronous FL strategy.

    Args:
        batch_size: Minibatch size (``B``) used by all per-client loaders.
        device: Device on which local training and evaluation run.
        num_workers: ``DataLoader`` worker count for client loaders.
        shuffle: Whether client loaders shuffle each local epoch.
    """

    def __init__(
        self,
        batch_size: int,
        device: str = "cpu",
        num_workers: int = 0,
        shuffle: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.shuffle = shuffle

        self._server: Optional[FederatedServer] = None
        self._history: Optional[TrainingHistory] = None

    @abstractmethod
    def _build_client(
        self,
        client_id: int,
        data_loader: DataLoader,
        model_fn: ModelFn,
    ) -> FederatedClient:
        """Instantiate the per-client local trainer for this strategy."""
        ...

    @abstractmethod
    def _build_aggregator(self) -> Aggregator:
        """Instantiate the server-side merging strategy."""
        ...

    def train(
        self,
        partitioner: "DatasetPartitioner",
        metadata_df: "pd.DataFrame",
        model_fn: ModelFn,
        dataset_fn: DatasetFn,
        num_rounds: int,
        global_eval_loader: Optional[DataLoader] = None,
        global_eval_fn: Optional[GlobalEvalFn] = None,
        on_round_end: Optional[RoundEndCallback] = None,
        save_every: int = 0,
        save_dir: Optional[str] = None,
    ) -> TrainingHistory:
        """Run the full FL training pipeline and return a :class:`TrainingHistory`.

        Args:
            partitioner: Splits ``metadata_df`` into per-client subsets.
            metadata_df: DataFrame with at least a ``filename`` column.
            model_fn: Zero-arg factory building a fresh, initialised model.
            dataset_fn: Turns a partition DataFrame into a torch ``Dataset``.
            num_rounds: Number of communication rounds to run.
            global_eval_loader: Optional held-out loader for per-round
                global-model evaluation (used only when ``global_eval_fn``
                is also supplied).
            global_eval_fn: ``(model, loader, device) -> {metric: value}``.
            on_round_end: Optional ``(round_stats, global_state)`` callback
                fired after every round -- useful for live logging.
            save_every: If > 0, write a checkpoint every ``save_every``
                rounds to ``{save_dir}/model_round_{N:04d}.pth``.  Set to
                0 to disable periodic checkpointing.
            save_dir: Directory to write checkpoints into.  Required when
                ``save_every > 0``; created if missing.
        """
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
        if save_every < 0:
            raise ValueError(f"save_every must be >= 0, got {save_every}")
        if save_every > 0 and save_dir is None:
            raise ValueError("save_every > 0 requires save_dir to be set.")
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        partitions = partitioner.partition(metadata_df)
        loaders = [
            DataLoader(
                dataset_fn(part),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=False,
            )
            for part in partitions
        ]

        clients = [
            self._build_client(client_id=idx, data_loader=loader, model_fn=model_fn)
            for idx, loader in enumerate(loaders)
        ]

        self._server = FederatedServer(
            model_fn=model_fn,
            clients=clients,
            aggregator=self._build_aggregator(),
            device=self.device,
        )

        history = TrainingHistory(partition_sizes=[c.num_samples for c in clients])

        eval_model: Optional[nn.Module] = None
        if global_eval_fn is not None and global_eval_loader is not None:
            eval_model = model_fn().to(self.device)

        for _ in range(num_rounds):
            stats = self._server.run_round()

            if eval_model is not None and global_eval_fn is not None:
                eval_model.load_state_dict(self._server.global_state)
                stats.global_metrics = global_eval_fn(
                    eval_model, global_eval_loader, self.device
                )

            history.append_round(stats)

            if save_every > 0 and stats.round_idx % save_every == 0:
                self._save_checkpoint(stats.round_idx, save_dir)  # type: ignore[arg-type]

            if on_round_end is not None:
                on_round_end(stats, self._server.global_state)

        self._history = history
        return history

    def _save_checkpoint(self, round_idx: int, save_dir: str) -> str:
        """Persist the current global state to ``save_dir``.

        Returns the path written so callers can log or track it.
        """
        if self._server is None:
            raise RuntimeError("_save_checkpoint called before server was built.")
        path = os.path.join(save_dir, f"model_round_{round_idx:04d}.pth")
        torch.save(
            {"state_dict": self._server.global_state, "round": round_idx},
            path,
        )
        print(f"[checkpoint] round {round_idx} -> {path}")
        return path

    @property
    def global_state(self) -> StateDict:
        if self._server is None:
            raise RuntimeError("Call train() before accessing global_state.")
        return self._server.global_state

    @property
    def history(self) -> TrainingHistory:
        if self._history is None:
            raise RuntimeError("Call train() before accessing history.")
        return self._history
