"""Registry of concrete federated learning strategies.

Each subpackage ``federated.strategies.<id>`` contributes a factory
``build_<id>(strategy_cfg, runtime_cfg) -> FederatedStrategy`` which is
registered in :data:`STRATEGY_REGISTRY` here.  A new strategy is wired
in by:

    1. Creating ``federated/strategies/<id>/`` with strategy, client,
       aggregation and (optionally) server modules plus an ``__init__``
       that exposes a ``build_<id>`` factory.
    2. Adding ``"<id>": build_<id>`` to :data:`STRATEGY_REGISTRY` below.
    3. Setting ``strategy.type: <id>`` in the Hydra config.
"""

from typing import Any, Callable, Dict

from ..strategy import FederatedStrategy
from .fedavg import build_fedavg


StrategyFactory = Callable[[Dict[str, Any], Dict[str, Any]], FederatedStrategy]


STRATEGY_REGISTRY: Dict[str, StrategyFactory] = {
    "fedavg": build_fedavg,
}


def build_strategy(
    strategy_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
) -> FederatedStrategy:
    """Build a :class:`FederatedStrategy` from resolved config blocks.

    Args:
        strategy_cfg: The ``strategy`` block of the resolved Hydra config.
            Must contain a ``type`` field naming an entry in
            :data:`STRATEGY_REGISTRY`; remaining fields are forwarded to
            the factory.
        runtime_cfg: The ``runtime`` block (device, seed, num_workers).
    """
    stype = strategy_cfg.get("type")
    if stype is None:
        raise ValueError("strategy config must contain a 'type' field.")
    if stype not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy type={stype!r}; known: {list(STRATEGY_REGISTRY)}"
        )

    strategy_kwargs = {k: v for k, v in strategy_cfg.items() if k != "type"}
    return STRATEGY_REGISTRY[stype](strategy_kwargs, runtime_cfg)


__all__ = ["STRATEGY_REGISTRY", "StrategyFactory", "build_strategy"]
