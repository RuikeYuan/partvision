"""Simulate warehouse usage against the API: new photos come in, the model
predicts, an "employee" confirms or corrects the label.

Shows (1) how accurate each decision bucket is in practice and (2) produces
labelled feedback for ``python -m partvision.retrain``.

    python scripts/simulate_production.py --per-class 40
    python scripts/simulate_production.py --url http://localhost:8000   # against a running server
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic_data import classes, render  # noqa: E402

from partvision import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--url", default=None, help="running server; default: in-process app")
    a = ap.parse_args()

    if a.url:
        import httpx
        client = httpx.Client(base_url=a.url, timeout=60)
    else:
        from fastapi.testclient import TestClient

        from partvision.api import create_app
        client = TestClient(create_app())

    rng = random.Random(a.seed)
    stream_dir = config.data_dir() / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(lambda: [0, 0, 0])      # decision -> [n, top1 correct, in top-3]
    jobs = [lbl for lbl in classes() for _ in range(a.per_class)]
    rng.shuffle(jobs)
    for i, true_label in enumerate(jobs):
        img = render(true_label, rng)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        r = client.post("/api/predict", files={"file": (f"{i}.jpg", buf.getvalue(), "image/jpeg")})
        r.raise_for_status()
        res = r.json()
        labels = [c["label"] for c in res["candidates"]]
        b = buckets[res["decision"]]
        b[0] += 1
        b[1] += labels[0] == true_label
        b[2] += true_label in labels
        # the employee always ends up with the right label (confirm / correct)
        client.post("/api/feedback", json={"prediction_id": res["prediction_id"], "label": true_label}).raise_for_status()

    total = sum(b[0] for b in buckets.values())
    print(f"\n{total} photos processed\n")
    print(f"{'decision':<12}{'share':>8}{'top-1 acc':>11}{'top-3 acc':>11}")
    for d in ("auto_accept", "confirm", "manual", "unknown"):
        if d in buckets:
            n, c1, c3 = buckets[d]
            print(f"{d:<12}{n / total:>8.0%}{c1 / n:>11.1%}{c3 / n:>11.1%}")
    print("\nfeedback stored; run `python -m partvision.retrain ...` to train on it")


if __name__ == "__main__":
    main()
