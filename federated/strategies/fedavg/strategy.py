"""FedAvg strategy (McMahan et al. 2017).

The training pipeline itself lives in :class:`FederatedStrategy`; this
module only supplies the two strategy-specific hooks:

* :meth:`_build_client` -- constructs a :class:`FedAvgClient` that runs
  plain local SGD for ``E`` epochs at learning rate ``eta``.
* :meth:`_build_aggregator` -- returns a
  :class:`WeightedAverageAggregator`, i.e. the ``sum_k (n_k / n) w_k``
  merge from the paper.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from .aggregation import WeightedAverageAggregator
from ...base import Aggregator, FederatedClient
from .client import FedAvgClient
from ...strategy import FederatedStrategy, ModelFn


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class FedAvgStrategy(FederatedStrategy):
    """FederatedAveraging as a concrete :class:`FederatedStrategy`.

    Args:
        local_epochs: Local passes per round (``E``).
        learning_rate: Local SGD learning rate (``eta``).
        batch_size: Local minibatch size (``B``).
        loss_fn: Loss applied to ``(model(x), y)`` during local training.
        device: Device for local training / evaluation.
        num_workers: ``DataLoader`` worker count for client loaders.
        shuffle: Whether client loaders shuffle each local epoch.
        aggregator: Optional override for the merge -- kept optional so
            ablations (e.g. unweighted mean, FedBN-style BN skipping) can
            be dropped in without subclassing.
    """

    def __init__(
        self,
        local_epochs: int,
        learning_rate: float,
        batch_size: int,
        loss_fn: LossFn,
        device: str = "cpu",
        num_workers: int = 0,
        shuffle: bool = True,
        aggregator: Optional[Aggregator] = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self._aggregator_override = aggregator

    def _build_client(
        self,
        client_id: int,
        data_loader: DataLoader,
        model_fn: ModelFn,
    ) -> FederatedClient:
        return FedAvgClient(
            client_id=client_id,
            model_fn=model_fn,
            data_loader=data_loader,
            local_epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    def _build_aggregator(self) -> Aggregator:
        return self._aggregator_override or WeightedAverageAggregator()
