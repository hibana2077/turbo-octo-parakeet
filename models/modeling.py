# coding=utf-8
from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.lossZoo as lossZoo


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackboneConfig:
    timm_name: str
    hidden_size: int
    num_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


CONFIGS = {
    "ViT-B_16": BackboneConfig("vit_base_patch16_224", 768, 12),
    "ViT-B_32": BackboneConfig("vit_base_patch32_224", 768, 12),
    "ViT-L_16": BackboneConfig("vit_large_patch16_224", 1024, 16),
    "ViT-L_32": BackboneConfig("vit_large_patch32_224", 1024, 16),
    "ViT-H_14": BackboneConfig("vit_huge_patch14_224", 1280, 16),
    "R50-ViT-B_16": BackboneConfig("vit_base_r50_s16_224", 768, 12),
}


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.fc1 = nn.Linear(hidden_size, self.patch_dim)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count = x.size(0), x.size(1)
        out = F.relu(self.fc1(x))
        out = out.view(-1, 3, self.patch_size, self.patch_size)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = torch.tanh(out)
        return out.view(batch_size, patch_count, 3, self.patch_size, self.patch_size)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        config: BackboneConfig,
        img_size=224,
        num_classes=21843,
        zero_head=False,
        vis=False,
        msa_layer=12,
        timm_model_name: str | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.vis = vis
        self.msa_layer = msa_layer

        model_name = timm_model_name or config.timm_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            global_pool="",
        )

        hidden_size = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", config.hidden_size)))
        num_heads = int(getattr(self.backbone, "num_heads", config.num_heads))
        if hidden_size % num_heads != 0:
            num_heads = config.num_heads

        self.backbone_cfg = BackboneConfig(
            timm_name=model_name,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.patch_size = self._resolve_patch_size()

        self.criterion = nn.MSELoss()
        self.decoder = Decoder(hidden_size=hidden_size, patch_size=self.patch_size)
        self.head = nn.Linear(hidden_size, num_classes)
        if self.zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        # Keep optimizer grouping compatibility with old FFTAT code.
        self.transformer = self.backbone

    def _resolve_patch_size(self) -> int:
        ps = getattr(self.backbone.patch_embed, "patch_size", 16)
        if isinstance(ps, tuple):
            if ps[0] != ps[1]:
                raise ValueError(f"Only square patch size is supported, got {ps}.")
            return int(ps[0])
        return int(ps)

    def _forward_tokens(self, x: torch.Tensor, ad_net, is_source: bool, cp_mask: torch.Tensor):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)

        loss_ad = x.new_zeros(())
        updated_cp_mask = cp_mask
        target_layer = min(max(self.msa_layer - 1, 0), len(self.backbone.blocks) - 1)

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i == target_layer and ad_net is not None:
                loss_ad = self._local_adversarial(x, ad_net, is_source=is_source)

        x = self.backbone.norm(x)
        return x, loss_ad, updated_cp_mask

    def _local_adversarial(self, tokens: torch.Tensor, ad_net, is_source: bool):
        batch_size, token_count, hidden_size = tokens.shape
        patch_tokens = tokens[:, self.prefix_tokens :, :]
        if patch_tokens.numel() == 0:
            return tokens.new_zeros(())

        heads = self.backbone_cfg.num_heads
        head_dim = hidden_size // heads
        patch_heads = patch_tokens.view(batch_size, -1, heads, head_dim).permute(0, 2, 1, 3).contiguous()

        _, loss_ad = lossZoo.adv_local(patch_heads, ad_net, is_source=is_source)
        return loss_ad

    def _build_cp_mask(self, score: torch.Tensor, prefix_tokens: int):
        score = score.float()
        score_max = torch.clamp(score.max(), min=1e-8)
        score = score / score_max

        inner = torch.matmul(score.unsqueeze(1), score.unsqueeze(0))
        inner = inner / torch.clamp(inner.max(), min=1e-8)

        total = score.numel() + prefix_tokens
        cp_mask = torch.ones(total, total, device=score.device, dtype=score.dtype)
        cp_mask[prefix_tokens:, prefix_tokens:] = inner

        if prefix_tokens > 0:
            cp_mask[:prefix_tokens, prefix_tokens:] = score.unsqueeze(0)
            cp_mask[prefix_tokens:, :prefix_tokens] = score.unsqueeze(1)
            cp_mask[:prefix_tokens, :prefix_tokens] = 1.0

        cp_mask = torch.where(cp_mask > 0.5, torch.sqrt(cp_mask), torch.square(cp_mask))
        return cp_mask.detach()

    def _target_patches(self, x: torch.Tensor) -> torch.Tensor:
        patch = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        return patch.permute(0, 2, 1).contiguous().view(x.size(0), -1, 3, self.patch_size, self.patch_size)

    def forward(self, x_s, cp_mask, optimal_flag, x_t=None, ad_net=None):
        del optimal_flag  # no longer used with timm backbone

        x_s_tokens, loss_ad_s, cp_mask = self._forward_tokens(x_s, ad_net=ad_net, is_source=True, cp_mask=cp_mask)
        logits_s = self.head(x_s_tokens[:, 0])

        if x_t is not None:
            x_t_tokens, loss_ad_t, _ = self._forward_tokens(x_t, ad_net=ad_net, is_source=False, cp_mask=cp_mask)
            logits_t = self.head(x_t_tokens[:, 0])

            # FFTAT reconstruction branch is kept as a no-op for API compatibility.
            loss_rec = x_s_tokens.new_zeros(())
            return logits_s, logits_t, (loss_ad_s + loss_ad_t) / 2.0, loss_rec, x_s_tokens, x_t_tokens, cp_mask

        return logits_s, None, None, cp_mask


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, "decay_mult": 2}]
