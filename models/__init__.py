# Copyright 2022 CircuitNet. All rights reserved.

from typing import Any, Callable, Dict
from torch import nn
from .routenet import RouteNet

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

__all__ = ['RouteNet', '_build_model_fn']