"""Dark matter substructure classification: axion vs. CDM vs. no substructure."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset

from ..data import CLASSES, LensClassificationDataset, few_shot_split, stratified_split
from ..metrics import classification_metrics
from ..models import EncoderClassifier, build_resnet18
from .base import Task


class ClassificationTask(Task):
    """Three-class substructure classification (axion / CDM / no substructure)."""

    name = "classification"
    monitor = "accuracy"
    higher_is_better = True

    def build_datasets(self):
        c = self.config
        dataset = LensClassificationDataset(f"{c.data_root}/{c.train_dataset}", c.image_size, cache=c.cache)
        if c.shots is None:
            train_idx, val_idx = stratified_split(dataset.labels, c.train_fraction, c.seed)
        else:
            train_idx, val_idx = few_shot_split(dataset.labels, c.shots, c.val_fraction, c.seed)
        return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

    def build_test_dataset(self, root: str):
        return LensClassificationDataset(root, self.config.image_size, cache=self.config.cache)

    def build_head(self, encoder: nn.Module) -> nn.Module:
        return EncoderClassifier(encoder, num_classes=len(CLASSES))

    def build_baseline(self, name: str) -> nn.Module:
        if name == "resnet18":
            return build_resnet18(len(CLASSES))
        return super().build_baseline(name)

    def loss(self, output, batch):
        return F.cross_entropy(output.float(), batch["label"], label_smoothing=self.config.label_smoothing)

    def collect(self, output, batch):
        return {
            "probabilities": torch.softmax(output.float(), dim=1).cpu().numpy(),
            "labels": batch["label"].cpu().numpy(),
        }

    def metrics(self, outputs: dict[str, np.ndarray]) -> dict:
        return classification_metrics(outputs["labels"], outputs["probabilities"], CLASSES)
