# coding=utf-8
from __future__ import annotations

import torch
import torch.nn.functional as F


def entropy_from_prob(prob: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    prob = prob.clamp_min(eps)
    return -(prob * torch.log2(prob)).sum(dim=1)


def information_maximization_loss(logits: torch.Tensor, gent: bool = True) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    loss = entropy_from_prob(prob).mean()

    if gent:
        mean_prob = prob.mean(dim=0)
        mean_prob = mean_prob.clamp_min(1e-10)
        global_entropy = -(mean_prob * torch.log2(mean_prob)).sum()
        loss = loss - global_entropy

    return loss


__all__ = ["information_maximization_loss"]
