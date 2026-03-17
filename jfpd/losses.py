from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    return 1.0 - (x * y).sum(dim=-1)


def normalized_feature_divergence(ft: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    dist = cosine_distance(ft, zs)
    return dist / (1.0 + dist)


def entropy_from_prob(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def kl_div_prob(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_div_prob(p, m, eps) + 0.5 * kl_div_prob(q, m, eps)


def normalized_prediction_divergence(pt: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    dist = js_divergence(pt, ps)
    return dist / (1.0 + dist)


def jfpd_loss(
    ft: torch.Tensor,
    pt: torch.Tensor,
    zs: torch.Tensor,
    ps: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    d_feat = normalized_feature_divergence(ft, zs)
    d_pred = normalized_prediction_divergence(pt, ps)
    hs = entropy_from_prob(ps)
    ht = entropy_from_prob(pt)

    psi = 1.0 / (1.0 + hs + ht)
    phi = 1.0 / (1.0 + d_feat)
    loss = alpha * psi * d_feat + (1.0 - alpha) * phi * d_pred

    stats = {
        "d_feat": d_feat.mean().item(),
        "d_pred": d_pred.mean().item(),
        "psi": psi.mean().item(),
        "phi": phi.mean().item(),
    }
    return loss.mean(), stats
