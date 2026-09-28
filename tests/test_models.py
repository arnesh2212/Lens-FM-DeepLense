import math

import pytest
import torch

from lens_lejepa.models import (
    ADAPTERS,
    EncoderClassifier,
    PatchSuperResolution,
    build_vit,
    count_parameters,
    inject_adapters,
)
from lens_lejepa.models.adapters import LoRALinear


def test_vit_small_shapes_and_size():
    vit = build_vit("vit_small", 160, 16)
    tokens = vit(torch.randn(2, 1, 160, 160))
    assert tokens.shape == (2, 100, 384)
    assert count_parameters(vit) == 21_431_424  # ViT-S/16 as reported in the paper
    assert not vit.pos_embed.requires_grad


def test_rslora_on_vit_small_matches_paper_count():
    vit = build_vit("vit_small", 160, 16)
    targets = inject_adapters(vit, "rslora", rank=32, alpha=64)
    model = EncoderClassifier(vit)
    assert len(targets) == 48  # qkv, proj, fc1, fc2 in each of 12 blocks
    assert count_parameters(model, trainable_only=True) == 2_360_451


@pytest.mark.parametrize("method", [m for m in ADAPTERS if m != "layerwise"] + ["layerwise"])
def test_adapters_start_as_the_frozen_model(method):
    torch.manual_seed(0)
    vit = build_vit("vit_tiny", 32, 16).eval()
    x = torch.randn(2, 1, 32, 32)
    before = vit(x)
    inject_adapters(vit, method, rank=4, alpha=8, rank_schedule=[2, 4] if method == "layerwise" else None)
    assert torch.allclose(vit(x), before, atol=1e-5)
    trainable = [name for name, p in vit.named_parameters() if p.requires_grad]
    assert trainable and all(any(k in name for k in ("lora_", "magnitude", "rank_logits")) for name in trainable)


def test_rslora_scaling():
    layer = torch.nn.Linear(8, 8)
    assert LoRALinear(layer, rank=16, alpha=64).scaling == pytest.approx(4.0)
    assert LoRALinear(layer, rank=16, alpha=64, rank_stabilised=True).scaling == pytest.approx(64 / math.sqrt(16))


def test_patch_super_resolution_output():
    vit = build_vit("vit_tiny", 32, 16)
    out = PatchSuperResolution(vit, 32, 16, output_size=64)(torch.randn(3, 1, 32, 32))
    assert out.shape == (3, 1, 64, 64)
    assert out.min() >= 0 and out.max() <= 1
