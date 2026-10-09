"""Loss functions for imbalanced multiclass flow classification.

SMOTE is deliberately not used: synthetic rows cannot be placed as edges in a
real traffic graph. Imbalance is handled in the loss instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Lin et al. (ICCV 2017): FL = -alpha_y * (1 - p_y)^gamma * log p_y. Label -1 is ignored."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        keep = y >= 0
        logits, y = logits[keep], y[keep]
        if y.numel() == 0:
            return logits.sum() * 0.0
        logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        loss = -((1 - logp.exp()) ** self.gamma) * logp
        if self.alpha.numel():
            loss = loss * self.alpha[y]
        return loss.mean()


def class_weights(counts: torch.Tensor, kind: str) -> torch.Tensor | None:
    counts = counts.clamp(min=1)
    if kind == "inv_sqrt":
        w = counts.rsqrt()
    elif kind == "inv":
        w = 1.0 / counts
    elif kind == "none":
        return None
    else:
        raise ValueError(f"unknown weighting {kind!r}")
    return w / w.sum() * len(w)


def build_loss(tcfg: dict, counts: torch.Tensor) -> nn.Module:
    kind = tcfg["loss"]
    if kind == "focal":
        return FocalLoss(tcfg["focal_gamma"], class_weights(counts, tcfg["focal_alpha"]))
    if kind == "ce":
        return nn.CrossEntropyLoss(ignore_index=-1)
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights(counts, "inv"), ignore_index=-1)
    raise ValueError(f"unknown loss {kind!r}")
