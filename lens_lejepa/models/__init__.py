"""Encoder, heads, adapters and baselines."""

from __future__ import annotations

from pathlib import Path

import torch

from .adapters import ADAPTERS, adapter_param_groups, canonical_adapter_name, inject_adapters
from .baselines import RCAN, Bicubic, ResNetRegressor, build_resnet18
from .heads import EncoderClassifier, EncoderRegressor, LeJEPAProjector, PatchSuperResolution
from .vit import VIT_CONFIGS, VisionTransformer, build_vit


def load_pretrained_encoder(name: str, image_size: int, patch_size: int, checkpoint: str | Path | None) -> VisionTransformer:
    """Build a ViT and, if a checkpoint is given, load its ``encoder`` weights."""
    encoder = build_vit(name, image_size, patch_size)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["encoder"] if "encoder" in state else state, strict=True)
    return encoder


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


__all__ = [
    "ADAPTERS",
    "RCAN",
    "VIT_CONFIGS",
    "Bicubic",
    "EncoderClassifier",
    "EncoderRegressor",
    "LeJEPAProjector",
    "PatchSuperResolution",
    "ResNetRegressor",
    "VisionTransformer",
    "adapter_param_groups",
    "build_resnet18",
    "build_vit",
    "canonical_adapter_name",
    "count_parameters",
    "inject_adapters",
    "load_pretrained_encoder",
]
