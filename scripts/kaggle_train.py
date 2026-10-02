"""Train candidate backbones on Kaggle's free GPU and pull the results back.

    python scripts/kaggle_train.py            # upload, push, wait, download
    python scripts/kaggle_train.py --fetch    # only download output of the last run
    python scripts/kaggle_train.py --runs "dinov2_vits14:224:--backbone-lr-mult 0.01 --head-epochs 5"

Needs the kaggle CLI (pip install kaggle) and an API token from
kaggle.com/settings -> API (kaggle.json in ~/.kaggle/, or KAGGLE_API_TOKEN).
The Kaggle account must be phone-verified to get GPU + internet in kernels.

New versions are copied into models/ but NOT promoted: compare summary.json, then
promote the winner with ModelRegistry.promote() and POST /api/admin/reload.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_SLUG = "partvision-bundle"
KERNEL_SLUG = "partvision-arch-compare"
KERNEL_SRC = ROOT / "scripts" / "kaggle" / "train_kernel.py"


def kaggle(*args: str, check: bool = True) -> str:
    exe = shutil.which("kaggle") or "kaggle"
    p = subprocess.run([exe, *args], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    name = api.get_config_value("username") or getattr(api, "config_values", {}).get("username")
    if not name:
        raise SystemExit("could not read Kaggle username from credentials")
    return name


def build_bundle(dst: Path) -> None:
    """Code + data + current model, so the kernel continues the same version numbering
    and evaluates on the same holdout set."""
    models = ROOT / "models"
    current = json.loads((models / "current.json").read_text())["version"]
    with tarfile.open(dst, "w:gz") as t:
        t.add(ROOT / "partvision", "partvision", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        for f in ("train.csv", "holdout.csv"):
            t.add(ROOT / "data" / f, f"data/{f}")
        t.add(ROOT / "data" / "images", "data/images")
        t.add(models / "current.json", "models/current.json")
        t.add(models / current, f"models/{current}")
        # empty placeholders for the other local versions, so Kaggle numbers new
        # versions after ours and the fetched folders never collide
        for v in models.iterdir():
            if v.is_dir() and re.fullmatch(r"v\d{3,}", v.name) and v.name != current:
                ti = tarfile.TarInfo(f"models/{v.name}")
                ti.type, ti.mode = tarfile.DIRTYPE, 0o755
                t.addfile(ti)
    print(f"bundle {dst.stat().st_size / 1e6:.1f} MB (baseline {current})")


def upload_dataset(user: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        build_bundle(d / "bundle.tar.gz")
        (d / "dataset-metadata.json").write_text(json.dumps({
            "title": "partvision bundle", "id": f"{user}/{DATASET_SLUG}",
            "licenses": [{"name": "other"}]}))
        exists = "ready" in kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).lower()
        if exists:
            print(kaggle("datasets", "version", "-p", str(d), "-m", time.strftime("%Y-%m-%d %H:%M")))
        else:
            print(kaggle("datasets", "create", "-p", str(d)))
    for _ in range(60):  # kernel push fails if the dataset is still processing
        if "ready" in kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).lower():
            return
        time.sleep(10)
    raise SystemExit("dataset not ready after 10 minutes")


def parse_runs(specs: list[str]) -> list[tuple[str, int, list[str]]]:
    """'arch:size[:extra train.py args]' -> (arch, size, [args])"""
    runs = []
    for spec in specs:
        arch, size, *rest = spec.split(":", 2)
        runs.append((arch, int(size), shlex.split(rest[0]) if rest else []))
    return runs


def push_kernel(user: str, runs: list[tuple[str, int, list[str]]] | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = KERNEL_SRC.read_text(encoding="utf-8")
        if runs:
            body = "".join(f"    {r!r},\n" for r in runs)
            src, n = re.subn(r"^RUNS = \[.*?^\]", lambda _: f"RUNS = [\n{body}]", src, flags=re.S | re.M)
            if n != 1:
                raise SystemExit("could not find the RUNS list in the kernel script")
        (d / KERNEL_SRC.name).write_text(src, encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": KERNEL_SRC.name,
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)))


def wait(user: str, timeout_min: int = 120) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        status = kaggle("kernels", "status", f"{user}/{KERNEL_SLUG}", check=False)
        print(f"[{(time.time() - t0) / 60:4.1f} min] {status}", flush=True)
        low = status.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            return low
        time.sleep(60)
    raise SystemExit("timed out waiting for the kernel")


def fetch(user: str) -> None:
    out = ROOT / "data" / "kaggle_runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    print(kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(out)))
    models = ROOT / "models"
    renamed: dict[str, str] = {}      # name on Kaggle -> name in local models/
    for v in sorted((out / "models").glob("v*"), key=lambda d: int(d.name[1:])) if (out / "models").exists() else []:
        dest = v.name
        if (models / dest).exists():  # Kaggle numbered it without knowing our local versions
            nums = [int(d.name[1:]) for d in models.iterdir() if re.fullmatch(r"v\d{3,}", d.name)]
            dest = f"v{max(nums) + 1:03d}"
        shutil.copytree(v, models / dest)
        renamed[v.name] = dest
        if dest != v.name:
            meta_path = models / dest / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["version"] = dest
            meta_path.write_text(json.dumps(meta, indent=2))
        print(f"added models/{dest}" + (f" (was {v.name} on Kaggle)" if dest != v.name else "") + " (not promoted)")
    summary = out / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        b = s["baseline"]
        print(f"\n{'version':8} {'arch':20} {'top1':>6} {'top3':>6} {'ece':>6}")
        print(f"{b['version']:8} {'(current)':20} {b['top1']:6.3f} {b['top3']:6.3f} {b['ece']:6.3f}")
        for c in s["candidates"]:
            print(f"{renamed.get(c['version'], c['version']):8} {c['arch']:20} {c['top1']:6.3f} {c['top3']:6.3f} {c['ece']:6.3f}")
    print(f"\nfull output: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="only download the last run's output")
    ap.add_argument("--runs", nargs="+", metavar="ARCH:SIZE[:ARGS]",
                    help='override the kernel runs, e.g. "dinov2_vits14:224:--backbone-lr-mult 0.01"')
    a = ap.parse_args()
    user = username()
    if not a.fetch:
        upload_dataset(user)
        push_kernel(user, parse_runs(a.runs) if a.runs else None)
        status = wait(user)
        if "complete" not in status:
            print("kernel did not complete; downloading logs anyway", file=sys.stderr)
    fetch(user)


if __name__ == "__main__":
    main()
