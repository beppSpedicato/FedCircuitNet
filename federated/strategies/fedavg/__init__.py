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
from ...device import resolve_device


LOSS_REGISTRY: Dict[str, type] = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
}


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
            plus ``seed``, ``num_workers``, ``shuffle``).
    """
    loss_type = strategy_cfg.get("loss_type", "MSELoss")
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}; known: {list(LOSS_REGISTRY)}"
        )

    return FedAvgStrategy(
        local_epochs=int(strategy_cfg["local_epochs"]),
        learning_rate=float(strategy_cfg["learning_rate"]),
        batch_size=int(strategy_cfg["batch_size"]),
        loss_fn=LOSS_REGISTRY[loss_type](),
        device=resolve_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 42)),
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        shuffle=bool(runtime_cfg.get("shuffle", True)),
    )


__all__ = [
    "FedAvgClient",
    "FedAvgStrategy",
    "LOSS_REGISTRY",
    "WeightedAverageAggregator",
    "build_fedavg",
]
