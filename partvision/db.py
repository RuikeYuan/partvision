"""SQLite storage for predictions and human feedback.

Every prediction is logged with the model version and its uncertainty. When the
employee confirms or corrects the label, a feedback row is written. That gives us:
  * a labelled dataset that grows with daily use (input for retraining),
  * online accuracy per model version (did v002 actually help in production?),
  * an active-learning queue: unlabelled predictions sorted by uncertainty.

For production with several users this would move to PostgreSQL; the schema is
the same.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    image_path    TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    model_version TEXT NOT NULL,
    top1          TEXT NOT NULL,
    confidence    REAL NOT NULL,
    margin        REAL NOT NULL,
    entropy       REAL NOT NULL,
    decision      TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pred_sha ON predictions(sha256);
CREATE INDEX IF NOT EXISTS ix_pred_version ON predictions(model_version);

CREATE TABLE IF NOT EXISTS feedback (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(id),
    label         TEXT NOT NULL,
    was_correct   INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        # one short-lived connection per operation: safe across FastAPI worker threads
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def log_prediction(self, image_path: str, sha256: str, result: dict) -> str:
        pid = uuid.uuid4().hex
        with self.conn() as c:
            c.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (pid, _now(), image_path, sha256, result["model_version"], result["candidates"][0]["label"],
                 result["confidence"], result["margin"], result["entropy"], result["decision"], json.dumps(result)),
            )
        return pid

    def get_prediction(self, pid: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None

    def add_feedback(self, pid: str, label: str) -> dict:
        pred = self.get_prediction(pid)
        if pred is None:
            raise KeyError(pid)
        correct = int(pred["top1"] == label)
        with self.conn() as c:
            # upsert: the employee may correct their own correction
            c.execute(
                "INSERT INTO feedback VALUES (?,?,?,?) ON CONFLICT(prediction_id) DO UPDATE SET "
                "label=excluded.label, was_correct=excluded.was_correct, created_at=excluded.created_at",
                (pid, label, correct, _now()),
            )
        return {"prediction_id": pid, "label": label, "model_was_correct": bool(correct)}

    def review_queue(self, limit: int = 20) -> list[dict]:
        """Active learning: unlabelled, non-auto-accepted predictions, most uncertain
        first (smallest margin between top-1 and top-2). Labelling these teaches the
        model the most per minute of human time."""
        with self.conn() as c:
            rows = c.execute(
                """SELECT p.id, p.created_at, p.image_path, p.top1, p.confidence, p.margin, p.decision, p.model_version
                   FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id
                   WHERE f.prediction_id IS NULL AND p.decision != 'auto_accept'
                   ORDER BY p.margin ASC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def labelled_images(self) -> list[tuple[str, str]]:
        """(image_path, human label) pairs, deduplicated by image hash (latest wins)."""
        with self.conn() as c:
            rows = c.execute(
                """SELECT image_path, label FROM (
                       SELECT p.image_path, f.label,
                              ROW_NUMBER() OVER (PARTITION BY p.sha256
                                                 ORDER BY f.created_at DESC, p.created_at DESC) AS rn
                       FROM feedback f JOIN predictions p ON p.id = f.prediction_id)
                   WHERE rn = 1"""
            ).fetchall()
        return [(r["image_path"], r["label"]) for r in rows]

    def stats(self) -> dict:
        with self.conn() as c:
            per_version = c.execute(
                """SELECT p.model_version AS version,
                          COUNT(*) AS predictions,
                          SUM(f.prediction_id IS NOT NULL) AS with_feedback,
                          AVG(f.was_correct) AS online_accuracy
                   FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id
                   GROUP BY p.model_version ORDER BY p.model_version"""
            ).fetchall()
            decisions = c.execute("SELECT decision, COUNT(*) AS n FROM predictions GROUP BY decision").fetchall()
            auto = c.execute(
                """SELECT COUNT(*) AS n, AVG(f.was_correct) AS acc FROM predictions p
                   JOIN feedback f ON f.prediction_id = p.id WHERE p.decision = 'auto_accept'"""
            ).fetchone()
        return {
            "per_version": [dict(r) for r in per_version],
            "decisions": {r["decision"]: r["n"] for r in decisions},
            "auto_accept_audited": {"n": auto["n"], "accuracy": auto["acc"]},
        }
