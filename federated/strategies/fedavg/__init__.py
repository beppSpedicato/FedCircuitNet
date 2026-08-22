"""FedAvg strategy package (McMahan et al. 2017).

Re-exports the concrete :class:`FedAvgStrategy` and its helpers, and
provides a config-driven factory :func:`build_fedavg` that is registered
with :mod:`federated.strategies` so a Hydra config can select it via
``strategy.type: fedavg``.
"""

from typing import Any, Dict

import torch.nn as nn

from .aggregation import WeightedAverageAggregator
from .client import FedAvgClient
from .strategy import FedAvgStrategy
from ....utils.device import resolve_device
from ....utils.losses import build_loss


def build_fedavg(
    strategy_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
) -> FedAvgStrategy:
    """Instantiate a :class:`FedAvgStrategy` from resolved config blocks.

    Args:
        strategy_cfg: Contents of the ``strategy`` config block minus its
            ``type`` field.  Expected keys: ``local_epochs``,
            ``learning_rate``, ``batch_size``, ``loss_type``.
        runtime_cfg: Contents of the ``runtime`` block (``cpu`` / ``gpu``
            / ``gpu_id`` as in ``code_examples/CircuitNet/drc_prediction/train.py``,
            plus , ``num_workers``, ``shuffle``).
    """
    loss_fn: nn.Module = build_loss(strategy_cfg)

    return FedAvgStrategy(
        local_epochs=int(strategy_cfg["local_epochs"]),
        learning_rate=float(strategy_cfg["learning_rate"]),
        batch_size=int(strategy_cfg["batch_size"]),
        loss_fn=loss_fn,
        device=resolve_device(runtime_cfg),
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        shuffle=bool(runtime_cfg.get("shuffle", True)),
    )


__all__ = [
    "FedAvgClient",
    "FedAvgStrategy",
    "WeightedAverageAggregator",
    "build_fedavg",
]
