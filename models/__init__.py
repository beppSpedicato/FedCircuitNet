# Copyright 2022 CircuitNet. All rights reserved.

from typing import Any, Callable, Dict
import torch.nn as nn
from .routenet import RouteNet
from .routenet_groupnorm import RouteNetGroupNorm
import torch
from torch.utils.data import DataLoader
from utils.device import features_to_device, resolve_device
from utils.losses import build_loss
from utils.metrics import build_metric

MODEL_REGISTRY: Dict[str, type] = {
    "RouteNet": RouteNet,
    "RouteNetGroupNorm": RouteNetGroupNorm,
}

def evaluate_global_model(
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

def load_model(
    model_cfg: Dict[str, Any],
    checkpoint: str,
    device: torch.device,
) -> torch.nn.Module:
    mtype = model_cfg["type"]
    if mtype not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.type={mtype!r}; known: {list(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[mtype]
    model = cls(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    model.to(device)
    return model

__all__ = ['RouteNet', 'RouteNetGroupNorm', '_build_model_fn', 'evaluate_global_model', 'load_model']