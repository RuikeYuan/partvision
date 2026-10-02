"""Inference: classifier + kNN + decision policy."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DecisionPolicy
from .data import build_transforms
from .index import EmbeddingIndex, vote_from_hits
from .model import load_partnet


class Predictor:
    """Loads one registered model version and answers ``predict(image)``.

    Thread-safe for concurrent requests: torch inference is re-entrant; the only
    mutable state is the index, guarded by a lock when new references are added.
    """

    def __init__(self, version_dir: str | Path, policy: DecisionPolicy | None = None, device: str = "cpu"):
        self.dir = Path(version_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.version = self.meta["version"]
        self.classes: list[str] = self.meta["classes"]
        self.temperature: float = self.meta["temperature"]
        self.unknown_similarity: float = self.meta["unknown_similarity"]
        self.ood_k: int = self.meta.get("ood_k", 5)
        self.policy = policy or DecisionPolicy()
        self.device = torch.device(device)
        state = torch.load(self.dir / "model.pt", map_location=self.device, weights_only=True)
        self.model = load_partnet(state, self.meta["arch"], len(self.classes),
                                  image_size=self.meta["image_size"]).to(self.device)
        self.transform = build_transforms(self.meta["image_size"], train=False)
        self.index = EmbeddingIndex.load(self.dir / "index.npz")
        self._extra_path = self.dir / "index_extra.npz"
        self.extra = (EmbeddingIndex.load(self._extra_path) if self._extra_path.exists()
                      else EmbeddingIndex(dim=self.index.dim, center=self.index.center))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ embedding
    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        return self.model.embed(x).cpu().numpy()

    @property
    def known_classes(self) -> list[str]:
        """Classifier classes + classes only known through reference photos."""
        return sorted(set(self.classes) | set(self.extra.labels))

    def add_references(self, images: list[Image.Image], label: str) -> int:
        """Make a new (or rare) part type recognisable without retraining."""
        emb = self.embed(images)
        with self._lock:
            self.extra.add(emb, label)
            self.extra.save(self._extra_path)
        return len(images)

    # ------------------------------------------------------------------ predict
    @torch.inference_mode()
    def predict(self, image: Image.Image, k: int = 3) -> dict:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        emb_t = self.model.embed(x)
        logits = self.model.head(emb_t)[0].cpu()
        emb = emb_t[0].cpu().numpy()

        probs = F.softmax(logits / self.temperature, dim=0)
        p_clf = {c: float(p) for c, p in zip(self.classes, probs.tolist())}

        # neighbours from the training index + any added reference photos
        with self._lock:
            hits = self.index.search(emb, 10) + self.extra.search(emb, 10)
        hits = sorted(hits, key=lambda h: -h[2])[:10]
        p_knn = vote_from_hits(hits)
        max_sim = hits[0][2] if hits else 0.0
        # open-set score: k-th neighbour in the training index (reference photos of
        # new classes are handled by new_class_hit below)
        ood_score = float(self.index.kth_similarity(emb, self.ood_k)[0])

        # fused ranking: classifier knows its classes well; kNN also covers new ones
        w = self.policy.knn_weight
        labels = set(p_clf) | set(p_knn)
        fused = {c: (1 - w) * p_clf.get(c, 0.0) + w * p_knn.get(c, 0.0) for c in labels}
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        candidates = [
            {"label": c, "score": round(s, 4), "classifier_prob": round(p_clf.get(c, 0.0), 4),
             "knn_vote": round(p_knn.get(c, 0.0), 4)}
            for c, s in ranked[:k]
        ]

        # uncertainty measures (used for decision + active learning queue)
        sorted_p = sorted(p_clf.values(), reverse=True)
        top1_p = sorted_p[0]
        margin = top1_p - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
        entropy = -sum(p * math.log(p) for p in sorted_p if p > 0) / math.log(max(len(sorted_p), 2))
        clf_top = max(p_clf, key=p_clf.get)
        knn_top = next(iter(p_knn), None)
        new_class_hit = knn_top is not None and knn_top not in p_clf and p_knn[knn_top] >= 0.6

        if new_class_hit:
            decision, reason = "confirm", f"matches reference photos of '{knn_top}' (not in classifier yet)"
        elif ood_score < self.unknown_similarity:
            decision, reason = "unknown", f"far from all known photos (score {ood_score:.2f} < {self.unknown_similarity:.2f}); possibly a new part type"
        elif top1_p >= self.policy.auto_accept and clf_top == knn_top:
            decision, reason = "auto_accept", f"calibrated confidence {top1_p:.0%} and neighbours agree"
        elif top1_p >= self.policy.confirm:
            decision, reason = "confirm", f"confidence {top1_p:.0%}; employee picks from top-{k}"
        else:
            decision, reason = "manual", f"low confidence {top1_p:.0%}"

        return {
            "model_version": self.version,
            "candidates": candidates,
            "decision": decision,
            "reason": reason,
            "classifier_top1": clf_top,
            "confidence": round(top1_p, 4),
            "margin": round(margin, 4),
            "entropy": round(entropy, 4),
            "max_similarity": round(max_sim, 4),
            "ood_score": round(ood_score, 4),
            "neighbours": [{"label": lbl, "similarity": round(s, 4)} for _, lbl, s in hits[:5]],
        }
