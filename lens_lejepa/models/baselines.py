"""Supervised reference models trained from scratch.

The physics-informed baselines in the paper (Lensiformer, LensPINN) and the
supervised ViT variants come from the ML4SCI DeepLense repository and are not
re-implemented here; see the README for links.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


def build_resnet18(num_outputs: int) -> nn.Module:
    """ResNet18 with a single-channel 3x3 stem and no max-pool (small images)."""
    model = resnet18(weights=None, num_classes=num_outputs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class ResNetRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_resnet18(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(-1)


class _ChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(4, channels // 16)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        return x + residual * self.channel_attention(residual)


class RCAN(nn.Module):
    """Compact RCAN-style 2x upsampler (8 channel-attention blocks)."""

    def __init__(self, channels: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[_ChannelAttentionBlock(channels) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        features = self.head(low_res)
        return torch.sigmoid(self.tail(self.body(features) + features))


class Bicubic(nn.Module):
    """Parameter-free bicubic upsampling baseline."""

    def __init__(self, output_size: int = 64) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        size = (self.output_size, self.output_size)
        return F.interpolate(low_res, size=size, mode="bicubic", align_corners=False, antialias=True)
