"""Assemble (and optionally push) the Hugging Face Docker Space.

    python scripts/build_hf_space.py                                   # build only
    python scripts/build_hf_space.py --push user/partvision \\
        --model-repo user/partvision-models --data-repo user/partvision-data

The Space holds code only. Model weights live in the HF model repo and are
downloaded when the Space starts (see partvision/hf_space.py). --push replaces
the Space contents and sets HF_MODEL_REPO / HF_DATA_REPO as Space variables.
Secrets (HF_TOKEN, PARTVISION_PASSWORD) are set by hand in the Space settings.
Needs a token with write access (`hf auth login` or the HF_TOKEN env var).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCKERFILE = """\
FROM python:3.12-slim

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . /app
RUN mkdir -p /app/data /app/models && chown -R user /app

ENV PARTVISION_HOME=/app
USER user
EXPOSE 7860
CMD ["python", "-m", "partvision.hf_space"]
"""

README = """\
---
title: PartVision
emoji: 🔧
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Car part recognition with calibrated confidence, kNN open-set search and a human feedback loop.
Deployed from GitHub Actions; model weights are pulled from the HF model repo at startup.
"""


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(ROOT / "partvision", out / "partvision",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (ROOT / "data" / "samples").exists():
        shutil.copytree(ROOT / "data" / "samples", out / "data" / "samples")
    # runtime deps only: everything above the "# dev" marker, plus the Hub client
    reqs = (ROOT / "requirements.txt").read_text().split("# dev")[0].strip()
    (out / "requirements.txt").write_text(reqs + "\nhuggingface_hub>=0.25\n")
    (out / "Dockerfile").write_text(DOCKERFILE)
    (out / "README.md").write_text(README, encoding="utf-8")
    print(f"built {out}")


def push(out: Path, space: str, model_repo: str, data_repo: str | None) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(space, "HF_MODEL_REPO", model_repo)
    if data_repo:
        api.add_space_variable(space, "HF_DATA_REPO", data_repo)
    # delete_patterns="*" drops remote files that are no longer in `out`
    api.upload_folder(repo_id=space, repo_type="space", folder_path=out,
                      delete_patterns="*", commit_message="Deploy from build_hf_space.py")
    print(f"pushed to https://huggingface.co/spaces/{space}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "hf_space")
    ap.add_argument("--push", metavar="SPACE", help="Space repo id, e.g. user/partvision")
    ap.add_argument("--model-repo", help="HF model repo with current.json + version dirs (required with --push)")
    ap.add_argument("--data-repo", help="private HF dataset repo for feedback-data snapshots (optional)")
    args = ap.parse_args()
    if args.push and not args.model_repo:
        ap.error("--push needs --model-repo")

    build(args.out)
    if args.push:
        push(args.out, args.push, args.model_repo, args.data_repo)


if __name__ == "__main__":
    main()
