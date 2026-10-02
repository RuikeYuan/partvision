"""Central configuration. Paths can be moved with the PARTVISION_HOME env var."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("PARTVISION_HOME", Path(__file__).resolve().parent.parent))


def data_dir() -> Path:
    return home() / "data"


def models_dir() -> Path:
    return home() / "models"


def db_path() -> Path:
    return data_dir() / "partvision.db"


def upload_dir() -> Path:
    return data_dir() / "uploads"


def basic_auth() -> tuple[str, str] | None:
    """(user, password) when PARTVISION_PASSWORD is set; None disables auth (local dev)."""
    password = os.environ.get("PARTVISION_PASSWORD")
    if not password:
        return None
    return os.environ.get("PARTVISION_USER", "admin"), password


@dataclass
class TrainConfig:
    arch: str = "resnet50"          # see model.SUPPORTED (ResNet, ConvNeXt, EfficientNetV2, DINOv2)
    pretrained: bool = True         # ImageNet weights (needs internet the first time)
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 12                # total epochs, including head-only warm-up
    head_epochs: int = 2            # stage 1: train only the new classifier head
    lr: float = 1e-3                # head learning rate
    backbone_lr_mult: float = 0.1   # stage 2: backbone gets lr * mult (discriminative LR)
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 4               # early stopping on validation top-1
    val_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionPolicy:
    """Turns calibrated probabilities into a workflow decision.

    auto_accept: confident AND the embedding neighbours agree -> no human needed
    confirm:     show top-3, employee clicks one
    manual:      model is unsure, employee types/selects the label
    unknown:     image is far from everything seen in training -> likely a new part type
    """

    auto_accept: float = 0.90
    confirm: float = 0.40
    knn_weight: float = 0.3         # weight of kNN votes in the fused ranking
