"""Device selection helpers, ported from the reference centralised trainer.

Mirrors ``code_examples/CircuitNet/drc_prediction/train.py`` so both
scripts read the same runtime keys:

* ``cpu`` (bool)               -- if truthy, use CPU and ignore ``gpu``.
* ``gpu`` / ``gpu_id`` (int)   -- CUDA device index when ``cpu`` is falsy.
                                  Defaults to 0.

:func:`resolve_device` also calls ``torch.cuda.set_device`` when a GPU
is selected, matching the reference behaviour.  :func:`features_to_device`
is a verbatim port of the reference helper of the same name.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch


def _gpu_id_from_cfg(runtime_cfg: Mapping[str, Any]) -> int:
    return int(runtime_cfg.get("gpu_id", runtime_cfg.get("gpu", 0)))


def resolve_device(runtime_cfg: Mapping[str, Any]) -> torch.device:
    """Return the :class:`torch.device` requested by *runtime_cfg*.

    When a GPU is selected this also calls ``torch.cuda.set_device`` so
    subsequent tensor allocations default to that device, exactly as the
    reference ``train.py`` does before ``model.to(device)``.
    """
    if runtime_cfg.get("cpu", False):
        return torch.device("cpu")
    gpu_id = _gpu_id_from_cfg(runtime_cfg)
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def features_to_device(
    feature: torch.Tensor,
    label: torch.Tensor,
    runtime_cfg: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move ``(feature, label)`` to the device implied by *runtime_cfg*."""
    if runtime_cfg.get("cpu", False):
        return feature, label
    gpu_id = _gpu_id_from_cfg(runtime_cfg)
    device = torch.device(f"cuda:{gpu_id}")
    return feature.to(device), label.to(device)


__all__ = ["features_to_device", "resolve_device"]
