"""Datasets, augmentation and sampling."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

from .labels import build_flip_map

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Item = tuple[Path, str]


# --------------------------------------------------------------------------- manifests
def read_manifest(path: str | Path) -> list[Item]:
    """CSV with columns ``path,label``. Relative paths resolve against the CSV's folder."""
    path = Path(path)
    items: list[Item] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row["path"])
            if not p.is_absolute():
                p = path.parent / p
            items.append((p, row["label"].strip()))
    return items


def write_manifest(items: list[Item], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for p, label in items:
            w.writerow([str(p), label])


def stratified_split(items: list[Item], val_fraction: float, seed: int) -> tuple[list[Item], list[Item]]:
    """Per-class split so rare classes still appear in validation (at least 1 if possible)."""
    by_class: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_class[it[1]].append(it)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        group = by_class[label][:]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        if len(group) >= 2:
            n_val = max(1, n_val)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


# --------------------------------------------------------------------------- transforms
def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        # NOTE: deliberately no RandomHorizontalFlip here -- flipping is handled
        # in PartsDataset because it must also change the label for sided parts.
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class PartsDataset(Dataset):
    """Image dataset with handedness-aware horizontal flipping.

    If ``flip_p > 0``: with that probability the image is mirrored and the label is
    mapped through the flip map (headlight_left -> headlight_right, wheel_rim ->
    wheel_rim). For sided classes whose mirror is unknown the flip is skipped.
    Besides avoiding label noise, this doubles the effective data of rare sided parts:
    every left headlight photo is also a valid right headlight photo.
    """

    def __init__(self, items: list[Item], classes: list[str], transform, flip_p: float = 0.0):
        self.items = items
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.flip_p = flip_p
        self.flip_map = build_flip_map(classes)
        missing = {lbl for _, lbl in items if lbl not in self.class_to_idx}
        if missing:
            raise ValueError(f"labels not in class list: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.items)

    @property
    def targets(self) -> list[int]:
        return [self.class_to_idx[lbl] for _, lbl in self.items]

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = load_rgb(path)
        y = self.class_to_idx[label]
        if self.flip_p > 0 and random.random() < self.flip_p and self.flip_map[y] != -1:
            img = TF.hflip(img)
            y = self.flip_map[y]
        return self.transform(img), y


def make_balanced_sampler(targets: list[int], num_classes: int) -> WeightedRandomSampler:
    """Sample each class equally often per epoch (inverse-frequency weights).

    Part inventories are long-tailed: thousands of mirrors, a handful of rare
    sensors. Without this the model learns to favour frequent classes.
    """
    counts = Counter(targets)
    class_w = torch.tensor([1.0 / counts.get(c, 1) for c in range(num_classes)], dtype=torch.double)
    weights = class_w[torch.tensor(targets)]
    return WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)
