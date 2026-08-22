# Copied verbatim from code_examples/CircuitNet/drc_prediction/utils/losses.py.
# Only change: `build_loss` looks the class up via ``globals()`` instead of
# a self-referential ``import utils.losses as losses``.

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY = ["L1Loss", "MSELoss", "BiasedMSELoss"]
__all__ = ["L1Loss", "MSELoss", "BiasedMSELoss", "build_loss"]


def build_loss(opt):
    if opt['loss_type'] not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_type={opt['loss_type']!r}; known: {list(LOSS_REGISTRY)}"
        )
    
    loss_type = opt.pop("loss_type")
    if loss_type == "BiasedMSELoss":
        return globals()[loss_type](
            s=float(opt.get("biased_loss_s", 1.0)),
            b=float(opt.get("biased_loss_b", 0.0)),
            loss_weight=float(opt.get("loss_weight", 100.0)),
            reduction=opt.get("reduction", "mean"),
            sample_wise=bool(opt.get("sample_wise", False)),
        )
    return globals()[loss_type]()


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction="mean", sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    elif reduction == "mean":
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss


def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", sample_wise=False, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper


@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction="mean", sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise,
        )


class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction="mean", sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise,
        )


class BiasedMSELoss(nn.Module):
    def __init__(self, s=1.0, b=0.0, loss_weight=100.0, reduction="mean", sample_wise=False):
        super().__init__()
        self.s = s
        self.b = b
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        bias_weight = 1.0 / (1.0 + torch.exp(self.s * (self.b - target)))
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight=bias_weight if weight is None else weight * bias_weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise,
        )
