"""Evaluation utilities shared with ``code_examples/CircuitNet/drc_prediction``.

Copied verbatim from that reference so a global model produced by the
federated pipeline (:mod:`train_aim`) can be evaluated with the exact
same metric implementations and ROC / PR-AUC computation used by the
centralised RouteNet baseline.

Public API:
    - :func:`build_metric`  -- name -> callable ``(target, pred) -> float``
        with the reference set ``nrms``, ``ssim``, ``psnr``, ``emd``.
    - :func:`build_loss`    -- name -> loss module (``MSELoss``,
        ``L1Loss``, ``BiasedMSELoss``).
    - :func:`multi_process_score` / :func:`roc_prc` -- multi-threshold
        confusion-matrix sweep that produces ROC-AUC and PR-AUC exactly
        the way the CircuitNet TCAD paper reports them.
    - :func:`build_roc_prc_metric` -- convenience wrapper that also
        parses an ann_file to locate the label directory.
    - :func:`set_random_seed` -- reference reproducibility helper.
"""

from .losses import BiasedMSELoss, L1Loss, MSELoss, build_loss
from .metrics import (
    build_metric,
    build_roc_prc_metric,
    multi_process_score,
    roc_prc,
)
from .seed import set_random_seed

__all__ = [
    "BiasedMSELoss",
    "L1Loss",
    "MSELoss",
    "build_loss",
    "build_metric",
    "build_roc_prc_metric",
    "multi_process_score",
    "roc_prc",
    "set_random_seed",
]
