# coding=utf-8
from __future__ import annotations

from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.coeff * grad_output, None


def grad_reverse(x: torch.Tensor, coeff: float) -> torch.Tensor:
    if coeff <= 0.0:
        return x
    return _GradientReverse.apply(x, coeff)


class FeatureFusionLayer(nn.Module):
    """Batch-wise latent fusion used before transferability modeling."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
        if tokens.size(1) <= prefix_tokens:
            return tokens

        cls_tokens = tokens[:, :prefix_tokens, :]
        patch_tokens = tokens[:, prefix_tokens:, :]

        if patch_tokens.size(0) == 1:
            return tokens

        batch_context = patch_tokens.mean(dim=0, keepdim=True).expand_as(patch_tokens)
        gate = torch.sigmoid(self.gate(patch_tokens))
        fused_patches = patch_tokens + gate * self.proj(batch_context - patch_tokens)
        return torch.cat([cls_tokens, fused_patches], dim=1)


class PatchDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        inner = max(hidden_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(inner, 1),
        )

    def forward(self, patch_tokens: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        patch_tokens = grad_reverse(patch_tokens, grl_lambda)
        return self.net(patch_tokens)


class TransferabilityGraphBuilder(nn.Module):
    def forward(self, patch_domain_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Domain uncertainty (closer to 0.5) is treated as more transferable.
        patch_domain_prob = torch.sigmoid(patch_domain_logits).squeeze(-1)
        transferability = 1.0 - (2.0 * (patch_domain_prob - 0.5).abs())
        transferability = transferability.clamp_(0.0, 1.0)

        norm = transferability / (transferability.sum(dim=1, keepdim=True) + 1e-6)
        graph = torch.einsum("bi,bj->bij", norm, norm)
        eye = torch.eye(graph.size(-1), device=graph.device, dtype=graph.dtype).unsqueeze(0)
        graph = 0.5 * graph + 0.5 * eye
        graph = graph / (graph.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return graph, transferability


class TGGuidedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        graph: torch.Tensor,
        transferability: torch.Tensor,
        prefix_tokens: int,
    ) -> torch.Tensor:
        bsz, token_count, hidden_dim = x.shape
        qkv = self.qkv(x).reshape(bsz, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        mask = torch.ones((bsz, token_count, token_count), device=x.device, dtype=x.dtype)
        if prefix_tokens < token_count:
            patch_slice = slice(prefix_tokens, token_count)
            mask[:, patch_slice, patch_slice] = graph

            patch_weights = transferability / (transferability.sum(dim=1, keepdim=True) + 1e-6)
            for i in range(prefix_tokens):
                mask[:, i, patch_slice] = patch_weights
                mask[:, patch_slice, i] = patch_weights

        attn = attn * mask.unsqueeze(1)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, token_count, hidden_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TGGuidedBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1) -> None:
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = TGGuidedSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, attn_dropout=drop)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        graph: torch.Tensor,
        transferability: torch.Tensor,
        prefix_tokens: int,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), graph, transferability, prefix_tokens)
        x = x + self.mlp(self.norm2(x))
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        inner = max(hidden_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(inner, inner),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(inner, 1),
        )

    def forward(self, features: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        features = grad_reverse(features, grl_lambda)
        return self.net(features)


class FFTATModel(nn.Module):
    """
    FFTAT-style UDA model:
    - timm ViT encoder
    - feature fusion on patch tokens
    - patch discriminator -> transferability graph
    - transferability-guided attention block(s)
    - classifier head on class token
    - domain discriminator on global feature
    """

    def __init__(
        self,
        timm_model_name: str,
        num_classes: int,
        img_size: int,
        pretrained: bool = True,
        split_layer: int = 6,
        tg_layers: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            global_pool="",
        )

        if not hasattr(self.backbone, "patch_embed") or not hasattr(self.backbone, "blocks"):
            raise ValueError(f"Model '{timm_model_name}' is not a ViT-like encoder with patch tokens.")

        self.embed_dim = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features")))
        self.num_heads = int(getattr(self.backbone, "num_heads", 12))
        self.prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        self.num_blocks = len(self.backbone.blocks)
        if self.num_blocks <= 0:
            raise ValueError("Backbone has no transformer blocks.")

        self.split_layer = min(max(split_layer, 0), self.num_blocks - 1)

        self.feature_fusion = FeatureFusionLayer(self.embed_dim)
        self.patch_discriminator = PatchDiscriminator(self.embed_dim)
        self.graph_builder = TransferabilityGraphBuilder()
        self.tg_blocks = nn.ModuleList(
            [TGGuidedBlock(self.embed_dim, self.num_heads) for _ in range(max(tg_layers, 1))]
        )

        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(self.embed_dim)

    def _patch_embed_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)
        return x

    def _forward_single(self, images: torch.Tensor, grl_lambda: float) -> Dict[str, torch.Tensor]:
        tokens = self._patch_embed_tokens(images)
        patch_logits = None
        transferability = None

        for layer_idx, blk in enumerate(self.backbone.blocks):
            if layer_idx == self.split_layer:
                tokens = self.feature_fusion(tokens, self.prefix_tokens)
                patch_tokens = tokens[:, self.prefix_tokens:, :]
                patch_logits = self.patch_discriminator(patch_tokens, grl_lambda=grl_lambda)
                graph, transferability = self.graph_builder(patch_logits)
                for tg_block in self.tg_blocks:
                    tokens = tg_block(tokens, graph, transferability, self.prefix_tokens)

            tokens = blk(tokens)

        if patch_logits is None:
            patch_tokens = tokens[:, self.prefix_tokens:, :]
            patch_logits = self.patch_discriminator(patch_tokens, grl_lambda=grl_lambda)
            _, transferability = self.graph_builder(patch_logits)

        tokens = self.backbone.norm(tokens)
        cls_feature = tokens[:, 0]
        logits = self.classifier(cls_feature)

        return {
            "tokens": tokens,
            "cls_feature": cls_feature,
            "logits": logits,
            "patch_logits": patch_logits,
            "transferability": transferability,
        }

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        source = self._forward_single(x_s, grl_lambda=grl_lambda)
        target = self._forward_single(x_t, grl_lambda=grl_lambda)

        domain_in = torch.cat([source["cls_feature"], target["cls_feature"]], dim=0)
        domain_logits = self.domain_discriminator(domain_in, grl_lambda=grl_lambda)

        return {
            "logits_s": source["logits"],
            "logits_t": target["logits"],
            "feat_s": source["cls_feature"],
            "feat_t": target["cls_feature"],
            "patch_logits_s": source["patch_logits"],
            "patch_logits_t": target["patch_logits"],
            "transfer_s": source["transferability"],
            "transfer_t": target["transferability"],
            "domain_logits": domain_logits,
        }

    def infer(self, images: torch.Tensor) -> torch.Tensor:
        return self._forward_single(images, grl_lambda=0.0)["logits"]


__all__ = [
    "FFTATModel",
    "FeatureFusionLayer",
    "PatchDiscriminator",
    "TransferabilityGraphBuilder",
    "TGGuidedSelfAttention",
    "TGGuidedBlock",
    "DomainDiscriminator",
]
