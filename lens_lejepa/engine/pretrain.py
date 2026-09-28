"""Self-supervised pretraining loop for Lens-LeJEPA and the LeJEPA control."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..config import PretrainConfig, config_to_dict
from ..data import UnlabelledLensDataset, make_loader
from ..models import LeJEPAProjector, build_vit, count_parameters
from ..ssl import LensLeJEPAObjective, pretraining_diagnostics
from ..utils import WarmupCosine, append_csv, check_finite, logger, make_run_dir, resolve_device, set_seed, write_json


def _param_groups(*modules: nn.Module) -> list[dict]:
    """Weight decay on matrices only; biases and norm parameters are excluded."""
    decay, no_decay = [], []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim == 1 or name.endswith("bias") else decay).append(param)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0, "no_decay": True}]


def pretrain(config: PretrainConfig, resume: str | Path | None = None) -> Path:
    """Train an encoder and return the run directory.

    Writes ``last_pretrain.pt`` (encoder, projector, optimiser state) and
    ``pretrain_metrics.csv`` after every epoch, so a run can be resumed.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = make_run_dir(config.output_dir, f"pretrain/{config.method}", config.run_name)
    write_json(run_dir / "config.json", config_to_dict(config))

    dataset = UnlabelledLensDataset(f"{config.data_root}/{config.dataset}", config.image_size, config.max_samples, config.seed)
    loader = make_loader(dataset, config.batch_size, shuffle=True, workers=config.workers)
    steps_per_epoch = len(loader) if config.max_steps_per_epoch is None else min(len(loader), config.max_steps_per_epoch)

    encoder = build_vit(config.encoder, config.image_size, config.patch_size).to(device)
    projector = LeJEPAProjector(encoder.embed_dim, config.projector_dim).to(device)
    objective = LensLeJEPAObjective(config.objective, config.patch_size).to(device)
    optimizer = torch.optim.AdamW(_param_groups(encoder, projector), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * steps_per_epoch
    lr_schedule = WarmupCosine(optimizer, "lr", config.start_lr, config.lr, config.final_lr, config.warmup_epochs * steps_per_epoch, total_steps)
    wd_schedule = WarmupCosine(optimizer, "weight_decay", config.weight_decay, config.weight_decay, config.final_weight_decay, 0, total_steps)
    logger.info(
        "Pretraining %s on %d images | encoder %.2fM params | %d steps/epoch | run dir %s",
        config.method, len(dataset), count_parameters(encoder) / 1e6, steps_per_epoch, run_dir,
    )

    start_epoch, collapse_epochs = 1, 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        encoder.load_state_dict(state["encoder"])
        projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        objective.sigreg.step.fill_(state["sigreg_step"])
        start_epoch = state["epoch"] + 1
        collapse_epochs = state.get("collapse_epochs", 0)
        lr_schedule.step_count = wd_schedule.step_count = state["step"]
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda")
    for epoch in range(start_epoch, config.epochs + 1):
        encoder.train()
        projector.train()
        sums: dict[str, float] = {}
        grad_norms: list[float] = []
        progress = tqdm(loader, total=steps_per_epoch, desc=f"pretrain {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if step > steps_per_epoch:
                break
            images = batch["image"].to(device, non_blocking=True)
            lr_schedule.step()
            wd_schedule.step()
            with autocast:
                loss, parts = objective(encoder, projector, images)
            check_finite("pretraining loss", loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), float("inf"))))
            optimizer.step()
            for key, value in {"loss": loss, **{k: v for k, v in parts.items() if k != "canonical_tokens"}}.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Diagnostics on the last batch of the epoch: detect representation collapse.
        diagnostics = pretraining_diagnostics(parts["canonical_tokens"])
        metrics = {"epoch": epoch, **{k: v / steps_per_epoch for k, v in sums.items()}, "grad_norm": float(np.mean(grad_norms)), **diagnostics}
        collapsed = diagnostics["feature_std"] < config.collapse_std_threshold or diagnostics["effective_rank_fraction"] < config.collapse_rank_threshold
        collapse_epochs = collapse_epochs + 1 if collapsed else 0
        append_csv(run_dir / "pretrain_metrics.csv", metrics)
        logger.info(
            "epoch %d | loss %.4f | d4 %.2e | ring %.2e | std %.3f | eff-rank %.3f",
            epoch, metrics["loss"], metrics["d4"], metrics["ring"], diagnostics["feature_std"], diagnostics["effective_rank_fraction"],
        )
        if collapse_epochs >= config.collapse_patience:
            raise RuntimeError(f"Representation collapsed for {collapse_epochs} epochs: {diagnostics}. Run rejected.")
        if not math.isfinite(metrics["loss"]):
            raise FloatingPointError("Non-finite epoch loss.")
        # Only healthy checkpoints are written, so a collapsed run can never be used downstream.
        torch.save(
            {
                "method": config.method,
                "epoch": epoch,
                "step": lr_schedule.step_count,
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sigreg_step": int(objective.sigreg.step),
                "collapse_epochs": collapse_epochs,
                "config": config_to_dict(config),
                "diagnostics": diagnostics,
            },
            run_dir / "last_pretrain.pt",
        )
    logger.info("Pretraining finished: %s", run_dir / "last_pretrain.pt")
    return run_dir
