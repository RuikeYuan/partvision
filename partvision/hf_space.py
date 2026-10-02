"""Entry point for the Hugging Face Space (free tier, ephemeral disk).

    python -m partvision.hf_space

On start it
  1. downloads the model registry (``current.json`` + version dirs) from the HF
     model repo in HF_MODEL_REPO,
  2. if HF_DATA_REPO is set, restores the last snapshot of feedback data
     (SQLite DB, uploaded photos, added reference photos) from that dataset repo
     and keeps pushing new snapshots every SYNC_EVERY_MIN minutes,
  3. serves the API on port 7860.

HF_TOKEN (a Space secret) is picked up by huggingface_hub automatically. Data
written after the last snapshot is lost when the Space restarts.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from huggingface_hub import CommitScheduler, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

from . import config

log = logging.getLogger("partvision.hf_space")
DB_SNAPSHOT = "partvision.db.snapshot"


def download_models(repo_id: str) -> None:
    snapshot_download(repo_id, repo_type="model", local_dir=config.models_dir())


def restore_data(repo_id: str) -> None:
    """Copy the last pushed snapshot into data/ and models/. A new, empty repo is fine."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            snap = Path(snapshot_download(repo_id, repo_type="dataset", local_dir=tmp))
        except RepositoryNotFoundError:
            return                                  # first start: the scheduler creates the repo
        if (snap / "data" / DB_SNAPSHOT).exists():
            config.data_dir().mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snap / "data" / DB_SNAPSHOT, config.db_path())
        if (snap / "data" / "uploads").exists():
            shutil.copytree(snap / "data" / "uploads", config.upload_dir(), dirs_exist_ok=True)
        for extra in (snap / "models").glob("*/index_extra.npz"):
            dest = config.models_dir() / extra.parent.name
            if dest.exists():                       # skip versions no longer in the model repo
                shutil.copyfile(extra, dest / extra.name)


class DataScheduler(CommitScheduler):
    """CommitScheduler that first takes a consistent copy of the live SQLite DB.

    Uploading the DB file directly could catch it mid-write; the backup API
    gives a transactionally consistent snapshot.
    """

    def push_to_hub(self):
        db = config.db_path()
        if db.exists():
            with self.lock:
                src, dst = sqlite3.connect(db), sqlite3.connect(config.data_dir() / DB_SNAPSHOT)
                try:
                    src.backup(dst)
                finally:
                    src.close()
                    dst.close()
        return super().push_to_hub()


def start_sync(repo_id: str, every_min: float) -> list[CommitScheduler]:
    return [
        DataScheduler(repo_id=repo_id, repo_type="dataset", folder_path=config.data_dir(),
                      path_in_repo="data", every=every_min, private=True,
                      allow_patterns=[DB_SNAPSHOT, "uploads/**"]),
        CommitScheduler(repo_id=repo_id, repo_type="dataset", folder_path=config.models_dir(),
                        path_in_repo="models", every=every_min, private=True,
                        allow_patterns=["*/index_extra.npz"]),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    model_repo = os.environ.get("HF_MODEL_REPO")
    if not model_repo:
        raise SystemExit("set HF_MODEL_REPO (Space variable) to the model repo, e.g. user/partvision-models")
    download_models(model_repo)

    data_repo = os.environ.get("HF_DATA_REPO")
    if data_repo:
        restore_data(data_repo)
        start_sync(data_repo, float(os.environ.get("SYNC_EVERY_MIN", "5")))
        log.info("syncing feedback data to %s", data_repo)
    else:
        log.warning("HF_DATA_REPO not set: feedback data is lost when the Space restarts")

    import uvicorn
    from .api import app             # imported after downloads: building the app reads the registry
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
