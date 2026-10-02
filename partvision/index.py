"""Embedding index for nearest-neighbour search over part photos.

Why a kNN index next to the classifier?
  * New part types: add a few reference photos and they become recognisable
    immediately -- no retraining (the classifier head has a fixed set of classes).
  * Explainability: "looks like these 5 stored photos" is something the warehouse
    employee can check at a glance.
  * Open-set detection: if a photo is far from *everything* we have seen, it is
    probably a part type we do not know yet.

Embeddings are L2-normalised so a dot product is cosine similarity. At inventory
scale (100k-1M photos) brute force numpy is still a few ms; beyond that swap in
FAISS or pgvector without changing the interface.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def vote_from_hits(hits: list[tuple[int, str, float]], tau: float = 0.05) -> dict[str, float]:
    """Turn (idx, label, similarity) hits into a class distribution.

    Weights are softmax(sim / tau): with a small tau the closest neighbours dominate,
    which works better than a flat majority vote when classes are visually similar.
    """
    if not hits:
        return {}
    sims = np.array([s for _, _, s in hits])
    w = np.exp((sims - sims.max()) / tau)
    scores: dict[str, float] = defaultdict(float)
    for (_, label, _), wi in zip(hits, w):
        scores[label] += float(wi)
    total = sum(scores.values())
    return {k: v / total for k, v in sorted(scores.items(), key=lambda kv: -kv[1])}


class EmbeddingIndex:
    """Cosine kNN over (optionally centred) L2-normalised embeddings.

    Centring: ResNet pooled features come after a ReLU, so every dimension is >= 0
    and all cosine similarities bunch up near 1 (0.93-0.99 in our tests), which
    makes thresholds useless. Subtracting the mean training embedding before
    normalising spreads them out.
    """

    def __init__(self, embeddings: np.ndarray | None = None, labels: list[str] | None = None,
                 dim: int | None = None, center: np.ndarray | None = None):
        if embeddings is None:
            if dim is None:
                raise ValueError("need embeddings or dim")
            embeddings = np.zeros((0, dim), dtype=np.float32)
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.emb = self._prep(embeddings) if len(embeddings) else np.asarray(embeddings, dtype=np.float32)
        self.labels: list[str] = list(labels or [])
        if len(self.labels) != len(self.emb):
            raise ValueError("embeddings and labels length mismatch")

    @classmethod
    def build_centered(cls, embeddings: np.ndarray, labels: list[str]) -> "EmbeddingIndex":
        center = l2_normalize(embeddings).mean(axis=0)
        return cls(embeddings, labels, center=center)

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = l2_normalize(np.atleast_2d(x))
        return l2_normalize(x - self.center) if self.center is not None else x

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.labels))

    def add(self, embeddings: np.ndarray, label: str) -> None:
        e = self._prep(embeddings)
        self.emb = np.concatenate([self.emb, e], axis=0)
        self.labels.extend([label] * len(e))

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, str, float]]:
        if len(self) == 0:
            return []
        sims = self.emb @ self._prep(query)[0]
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), self.labels[i], float(sims[i])) for i in idx]

    def vote(self, query: np.ndarray, k: int = 10, tau: float = 0.05) -> dict[str, float]:
        """Similarity-weighted class distribution among the k nearest neighbours."""
        return vote_from_hits(self.search(query, k), tau)

    def kth_similarity(self, queries: np.ndarray, k: int = 5) -> np.ndarray:
        """Similarity to the k-th nearest neighbour, per query row.

        Out-of-distribution score from "Out-of-Distribution Detection with Deep
        Nearest Neighbors" (Sun et al., 2022): a known part has several close
        neighbours; an unknown one at best one accidental match. k-th is more
        robust than the single nearest neighbour.
        """
        sims = self._prep(queries) @ self.emb.T
        k = min(k, sims.shape[1])
        return -np.partition(-sims, k - 1, axis=1)[:, k - 1]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        extra = {"center": self.center} if self.center is not None else {}
        np.savez_compressed(path, emb=self.emb, labels=np.array(self.labels, dtype=object), **extra)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        data = np.load(path, allow_pickle=True)
        idx = cls(dim=data["emb"].shape[1], center=data["center"] if "center" in data else None)
        idx.emb = data["emb"].astype(np.float32)
        idx.labels = [str(x) for x in data["labels"]]
        return idx
