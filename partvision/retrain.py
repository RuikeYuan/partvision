"""Retrain from human feedback (the "flywheel").

    python -m partvision.retrain --base data/train.csv --holdout data/holdout.csv

1. Take the original training manifest.
2. Add every image an employee labelled (latest label wins per image).
3. Drop feedback images that are byte-identical to holdout images (leakage:
   otherwise the model is evaluated on photos it trained on and looks better
   than it is).
4. Classes with too few examples stay in the kNN reference index instead of
   becoming classifier classes.
5. Train a new version; it is promoted only if it passes the gate on the same
   holdout set as the current model.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from . import config
from .data import read_manifest, write_manifest
from .db import Store
from .registry import ModelRegistry
from .train import train_and_register


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_retrain_set(base, holdout, feedback, min_per_class: int = 8, log=print):
    holdout_hashes = {file_sha256(p) for p, _ in holdout}
    clean, leaked = [], 0
    for path, label in feedback:
        if not Path(path).exists():
            continue
        if file_sha256(path) in holdout_hashes:
            leaked += 1
            continue
        clean.append((Path(path), label))
    combined = list(base) + clean
    counts = Counter(lbl for _, lbl in combined)
    too_small = {c for c, n in counts.items() if n < min_per_class}
    combined = [it for it in combined if it[1] not in too_small]
    log(f"base={len(base)} feedback={len(feedback)} used={len(clean)} "
        f"dropped_holdout_leak={leaked} classes_too_small={sorted(too_small)}")
    return combined, clean


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--min-per-class", type=int, default=8)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--from-scratch", action="store_true", help="do not warm-start from the current model")
    a = ap.parse_args(argv)

    base, holdout = read_manifest(a.base), read_manifest(a.holdout)
    feedback = Store(config.db_path()).labelled_images()
    items, used = build_retrain_set(base, holdout, feedback, a.min_per_class)
    if not used:
        print("no usable feedback yet; nothing to retrain")
        return
    out = config.data_dir() / "manifests"
    reg = ModelRegistry(config.models_dir())
    write_manifest(items, out / f"retrain_after_{reg.current_version() or 'none'}.csv")

    cfg = config.TrainConfig()
    if a.arch: cfg.arch = a.arch
    if a.no_pretrained: cfg.pretrained = False
    if a.image_size: cfg.image_size = a.image_size
    if a.epochs: cfg.epochs = a.epochs
    if a.lr: cfg.lr = a.lr
    init_from = None if a.from_scratch else reg.current_dir()
    if init_from is not None:
        cfg.arch = reg.meta(init_from.name)["arch"]
    res = train_and_register(items, holdout, cfg, reg, promote="gate", init_from=init_from,
                             notes=f"retrain with {len(used)} feedback images")
    if res["promoted"]:
        print("promoted. Call POST /api/admin/reload to serve it without a restart.")


if __name__ == "__main__":
    main()
