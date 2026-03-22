from typing import Dict, Literal, Tuple

import torch
import torch.nn.functional as F


LossMode = Literal["jfpd", "fgpd", "pgfd"]


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
    mode: LossMode = "jfpd",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    d_feat = normalized_feature_divergence(ft, zs)
    d_pred = normalized_prediction_divergence(pt, ps)
    hs = entropy_from_prob(ps)
    ht = entropy_from_prob(pt)

    psi = 1.0 / (1.0 + hs + ht)
    phi = 1.0 / (1.0 + d_feat)

    feat_comp = psi * d_feat
    pred_comp = phi * d_pred

    if mode == "jfpd":
        loss = alpha * feat_comp + (1.0 - alpha) * pred_comp
    elif mode == "pgfd":
        loss = feat_comp
    elif mode == "fgpd":
        loss = pred_comp
    else:
        raise ValueError(f"Unsupported loss mode '{mode}'. Expected one of: jfpd, fgpd, pgfd.")

    stats = {
        "d_feat": d_feat.mean().item(),
        "d_pred": d_pred.mean().item(),
        "psi": psi.mean().item(),
        "phi": phi.mean().item(),
        "feat_comp": feat_comp.mean().item(),
        "pred_comp": pred_comp.mean().item(),
    }
    return loss.mean(), stats
