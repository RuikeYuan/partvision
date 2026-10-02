"""Evaluation metrics beyond plain accuracy."""

from __future__ import annotations

import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 3)) -> dict[str, float]:
    out = {}
    maxk = min(max(ks), logits.shape[1])
    top = logits.topk(maxk, dim=1).indices
    for k in ks:
        k_eff = min(k, maxk)
        out[f"top{k}"] = float(top[:, :k_eff].eq(labels[:, None]).any(dim=1).float().mean())
    return out


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, n: int) -> torch.Tensor:
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: torch.Tensor, classes: list[str]) -> dict[str, float | None]:
    support = cm.sum(dim=1)
    return {
        c: (float(cm[i, i] / support[i]) if support[i] > 0 else None)
        for i, c in enumerate(classes)
    }


def top_confusions(cm: torch.Tensor, classes: list[str], k: int = 5) -> list[dict]:
    """Most frequent (true -> predicted) mistakes; tells you where to collect data."""
    off = cm.clone()
    off.fill_diagonal_(0)
    pairs = []
    for idx in off.flatten().argsort(descending=True)[:k].tolist():
        t, p = divmod(idx, off.shape[1])
        if off[t, p] > 0:
            pairs.append({"true": classes[t], "predicted": classes[p], "count": int(off[t, p])})
    return pairs


def risk_coverage(probs: torch.Tensor, labels: torch.Tensor, thresholds=(0.0, 0.4, 0.6, 0.8, 0.9, 0.95)) -> list[dict]:
    """Selective prediction: if we only auto-accept above threshold t,
    what share of parts is automated (coverage) and how accurate are those?"""
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({
            "threshold": t,
            "coverage": float(m.float().mean()),
            "accuracy": float(correct[m].float().mean()) if m.any() else None,
        })
    return rows
