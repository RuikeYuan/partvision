"""Training pipeline.

    python -m partvision.train --manifest data/train.csv --holdout data/holdout.csv

Steps
  1. Stratified train/val split (val is for early stopping + calibration).
  2. Two-stage fine-tuning (when using pretrained weights):
       stage 1: freeze backbone, train only the new head (head starts random; large
                gradients through a random head would wreck pretrained features)
       stage 2: unfreeze, backbone LR = lr * 0.1 (discriminative learning rates)
     Cosine LR schedule, label smoothing, class-balanced sampling, early stopping.
  3. Temperature scaling on val -> calibrated probabilities.
  4. Build the kNN embedding index from the training images.
  5. Derive the "unknown part" similarity threshold from val data.
  6. Evaluate on the fixed holdout set, save as a new version, promote if it
     passes the gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .calibration import expected_calibration_error, fit_temperature
from .data import (PartsDataset, build_transforms, make_balanced_sampler, read_manifest,
                   stratified_split)
from .index import EmbeddingIndex
from .metrics import (confusion_matrix, per_class_accuracy, risk_coverage, top_confusions,
                      topk_accuracy)
from .model import PartNet
from .registry import ModelRegistry, passes_gate


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def items_hash(items) -> str:
    h = hashlib.sha1()
    for p, label in sorted((Path(p).name, lbl) for p, lbl in items):
        h.update(f"{p}\t{label}\n".encode())
    return h.hexdigest()[:12]


@torch.no_grad()
def collect(model: PartNet, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model once; return logits, labels and embeddings for every image."""
    model.eval()
    all_logits, all_labels, all_emb = [], [], []
    for x, y in loader:
        emb = model.embed(x.to(device))
        all_emb.append(emb.cpu())
        all_logits.append(model.head(emb).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_emb)


def _run_epoch(model, loader, criterion, optimizer, scheduler, device, freeze_backbone_bn: bool) -> float:
    model.train()
    if freeze_backbone_bn:
        # frozen backbone: keep BatchNorm running stats from pretraining
        model.backbone.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def train_model(train_items, val_items, classes, cfg: config.TrainConfig, device, log=print, init_from: Path | None = None):
    set_seed(cfg.seed)
    train_ds = PartsDataset(train_items, classes, build_transforms(cfg.image_size, True), flip_p=0.5)
    val_ds = PartsDataset(val_items, classes, build_transforms(cfg.image_size, False))
    sampler = make_balanced_sampler(train_ds.targets, len(classes))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, drop_last=len(train_ds) > cfg.batch_size)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers)

    model = PartNet(cfg.arch, len(classes), pretrained=cfg.pretrained and init_from is None,
                    image_size=cfg.image_size).to(device)
    if init_from is not None:
        # warm start from the production model: always reuse the backbone; reuse the
        # head only if the class list is unchanged (new classes need a new head)
        prev_meta = json.loads((init_from / "meta.json").read_text())
        state = torch.load(init_from / "model.pt", map_location=device, weights_only=True)
        if prev_meta["arch"] != cfg.arch:
            raise ValueError(f"warm start needs same arch ({prev_meta['arch']} != {cfg.arch})")
        same_head = prev_meta["classes"] == classes
        keep = {k: v for k, v in state.items() if same_head or k.startswith("backbone.")}
        model.load_state_dict(keep, strict=same_head)
        log(f"warm start from {init_from.name} (head reused: {same_head})")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # stage plan: (name, epochs, backbone trainable, backbone lr multiplier)
    warm = cfg.pretrained or init_from is not None
    if warm and cfg.head_epochs > 0:
        stages = [("head", cfg.head_epochs, False, 0.0),
                  ("full", max(cfg.epochs - cfg.head_epochs, 1), True, cfg.backbone_lr_mult)]
    else:
        stages = [("full", cfg.epochs, True, 1.0)]   # from scratch: one LR for all

    best = {"top1": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    history, epoch, bad = [], 0, 0
    for name, n_epochs, trainable, mult in stages:
        model.set_backbone_trainable(trainable)
        groups = model.param_groups(cfg.lr, mult) if trainable else [{"params": model.head.parameters(), "lr": cfg.lr}]
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs * len(train_dl), 1))
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            train_loss = _run_epoch(model, train_dl, criterion, opt, sched, device, freeze_backbone_bn=not trainable)
            logits, labels, _ = collect(model, val_dl, device)
            val_loss = float(F.cross_entropy(logits, labels))
            acc = topk_accuracy(logits, labels)
            history.append({"epoch": epoch, "stage": name, "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4), **{k: round(v, 4) for k, v in acc.items()}})
            log(f"epoch {epoch:2d} [{name}] train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
                f"val_top1={acc['top1']:.3f} val_top3={acc['top3']:.3f} ({time.time() - t0:.0f}s)")
            if (acc["top1"], -val_loss) > (best["top1"], -best["loss"]):
                best = {"top1": acc["top1"], "loss": val_loss, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
                bad = 0
            elif name == "full":
                bad += 1
                if bad >= cfg.patience:
                    log(f"early stopping (no improvement for {cfg.patience} epochs)")
                    break
    model.load_state_dict(best["state"])
    model.set_backbone_trainable(True)
    model.eval()
    log(f"best epoch: {best['epoch']} (val top1={best['top1']:.3f})")
    return model, val_dl, history


OOD_K = 5            # k-th neighbour used as the "is this a known part?" score
OOD_RECALL = 0.95    # threshold keeps 95% of known parts above it


def build_index(model, items, classes, cfg, device) -> EmbeddingIndex:
    ds = PartsDataset(items, classes, build_transforms(cfg.image_size, False))
    _, labels, emb = collect(model, DataLoader(ds, batch_size=cfg.batch_size * 2), device)
    return EmbeddingIndex.build_centered(emb.numpy(), [classes[i] for i in labels.tolist()])


def evaluate(model, items, classes, cfg, temperature: float, device) -> dict:
    ds = PartsDataset(items, classes, build_transforms(cfg.image_size, False))
    logits, labels, _ = collect(model, DataLoader(ds, batch_size=cfg.batch_size * 2), device)
    probs = F.softmax(logits / temperature, dim=1)
    preds = probs.argmax(1)
    cm = confusion_matrix(preds, labels, len(classes))
    return {
        "n": len(items),
        **topk_accuracy(logits, labels),
        "ece": expected_calibration_error(probs, labels),
        "ece_uncalibrated": expected_calibration_error(F.softmax(logits, 1), labels),
        "per_class": per_class_accuracy(cm, classes),
        "top_confusions": top_confusions(cm, classes),
        "risk_coverage": risk_coverage(probs, labels),
    }


def train_and_register(train_items, holdout_items, cfg: config.TrainConfig, registry: ModelRegistry,
                       promote: str = "gate", notes: str = "", log=print, init_from: Path | None = None) -> dict:
    device = pick_device()
    classes = sorted({lbl for _, lbl in train_items})
    log(f"device={device} classes={len(classes)} images={len(train_items)} arch={cfg.arch} pretrained={cfg.pretrained}")
    tr, va = stratified_split(train_items, cfg.val_fraction, cfg.seed)

    model, val_dl, history = train_model(tr, va, classes, cfg, device, log, init_from=init_from)

    # calibration on validation logits
    val_logits, val_labels, val_emb = collect(model, val_dl, device)
    temperature = fit_temperature(val_logits, val_labels)
    log(f"temperature T={temperature:.3f}")

    # embedding index from the training split; open-set threshold from val
    index = build_index(model, tr, classes, cfg, device)
    val_ood = index.kth_similarity(val_emb.numpy(), OOD_K)
    unknown_similarity = float(np.quantile(val_ood, 1 - OOD_RECALL))
    log(f"unknown-part threshold (k={OOD_K} neighbour similarity)={unknown_similarity:.3f}")

    hold = [it for it in holdout_items if it[1] in set(classes)]
    if len(hold) < len(holdout_items):
        log(f"warning: {len(holdout_items) - len(hold)} holdout images have labels unknown to this model; skipped")
    holdout = evaluate(model, hold, classes, cfg, temperature, device)
    log(f"holdout top1={holdout['top1']:.3f} top3={holdout['top3']:.3f} "
        f"ECE={holdout['ece']:.3f} (uncalibrated {holdout['ece_uncalibrated']:.3f})")

    vdir = registry.new_version_dir()
    torch.save(model.state_dict(), vdir / "model.pt")
    index.save(vdir / "index.npz")
    meta = {
        "version": vdir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": classes,
        "arch": cfg.arch,
        "image_size": cfg.image_size,
        "temperature": temperature,
        "unknown_similarity": unknown_similarity,
        "ood_k": OOD_K,
        "train_config": cfg.to_dict(),
        "n_train": len(tr), "n_val": len(va),
        "train_hash": items_hash(train_items),
        "holdout_hash": items_hash(hold),
        "history": history,
        "holdout": holdout,
        "notes": notes,
        "warm_start_from": init_from.name if init_from else None,
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2))

    current_v = registry.current_version()
    current_meta = registry.meta(current_v) if current_v else None
    ok, reasons = passes_gate(meta, current_meta)
    promoted = promote == "force" or (promote == "gate" and ok)
    if promoted:
        registry.promote(vdir.name)
    log(f"saved {vdir.name}; gate={'PASS' if ok else 'FAIL'} {reasons or ''}; promoted={promoted}")
    return {"version": vdir.name, "promoted": promoted, "gate_ok": ok, "gate_reasons": reasons, "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a part recognition model and register it.")
    ap.add_argument("--manifest", required=True, help="CSV path,label for training (+val split)")
    ap.add_argument("--holdout", required=True, help="fixed CSV used to compare model versions")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (no ImageNet weights)")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone-lr-mult", type=float, default=None, help="stage-2 backbone LR = lr * this")
    ap.add_argument("--head-epochs", type=int, default=None, help="stage-1 head-only epochs")
    ap.add_argument("--promote", choices=["gate", "force", "never"], default="gate")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)

    cfg = config.TrainConfig()
    for k in ("arch", "image_size", "epochs", "batch_size", "lr", "backbone_lr_mult", "head_epochs"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.no_pretrained:
        cfg.pretrained = False
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    train_and_register(read_manifest(a.manifest), read_manifest(a.holdout), cfg,
                       ModelRegistry(config.models_dir()), promote=a.promote, notes=a.notes)


if __name__ == "__main__":
    main()
