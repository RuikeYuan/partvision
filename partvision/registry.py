"""Versioned model registry with a promotion gate.

models/
  v001/ model.pt  meta.json  index.npz  [index_extra.npz]
  v002/ ...
  current.json   -> {"version": "v002"}

A retrained model is only promoted if it is not worse than the current one on
the SAME fixed holdout set. Without this, a batch of bad feedback labels could
silently degrade production. Rolling back = pointing current.json at an older
version.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_VERSION_RE = re.compile(r"^v(\d{3,})$")


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def versions(self) -> list[str]:
        vs = [p.name for p in self.root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
        return sorted(vs, key=lambda v: int(v[1:]))

    def new_version_dir(self) -> Path:
        vs = self.versions()
        n = int(vs[-1][1:]) + 1 if vs else 1
        d = self.root / f"v{n:03d}"
        d.mkdir()
        return d

    def current_version(self) -> str | None:
        f = self.root / "current.json"
        if not f.exists():
            return None
        return json.loads(f.read_text())["version"]

    def current_dir(self) -> Path | None:
        v = self.current_version()
        return self.root / v if v else None

    def promote(self, version: str) -> None:
        if not (self.root / version / "model.pt").exists():
            raise FileNotFoundError(f"{version} has no model.pt")
        tmp = self.root / "current.json.tmp"
        tmp.write_text(json.dumps({"version": version, "promoted_at": datetime.now(timezone.utc).isoformat()}))
        tmp.replace(self.root / "current.json")      # atomic swap

    def meta(self, version: str) -> dict:
        return json.loads((self.root / version / "meta.json").read_text())


def passes_gate(candidate: dict, current: dict | None, max_top1_drop: float = 0.01, max_ece: float = 0.10) -> tuple[bool, list[str]]:
    """Compare holdout metrics. Returns (ok, reasons)."""
    reasons = []
    if candidate["holdout"]["ece"] > max_ece:
        reasons.append(f"ECE {candidate['holdout']['ece']:.3f} > {max_ece}")
    if current is not None:
        if candidate.get("holdout_hash") != current.get("holdout_hash"):
            reasons.append("holdout set differs from current model's; metrics not comparable")
        drop = current["holdout"]["top1"] - candidate["holdout"]["top1"]
        if drop > max_top1_drop:
            reasons.append(f"top-1 dropped by {drop:.3f} (> {max_top1_drop})")
    return (not reasons), reasons
