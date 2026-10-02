"""Kaggle kernel: train several backbones on the same data and compare them on the holdout set.

Pushed by scripts/kaggle_train.py. Input: the partvision-bundle dataset (code + data +
current models). Output in /kaggle/working:
    models/vNNN/...   one registered (not promoted) version per arch
    summary.json      holdout metrics side by side
    logs/<arch>.log
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

RUNS = [  # (arch, image_size, extra train.py args); kaggle_train.py --runs rewrites this list
    ("resnet50", 224, []),
    ("convnext_tiny", 224, []),
    ("efficientnet_v2_s", 224, []),
    ("dinov2_vits14", 224, []),
]

INPUT = Path("/kaggle/input")
WORK = Path("/tmp/pv")
OUT = Path("/kaggle/working")


def find_bundle() -> None:
    """The dataset holds bundle.tar.gz; Kaggle may or may not have extracted it."""
    for p in INPUT.rglob("*"):
        if p.name == "bundle.tar.gz":
            WORK.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p) as t:
                t.extractall(WORK)
            return
        if p.is_dir() and (p / "partvision" / "__init__.py").exists():
            shutil.copytree(p, WORK)
            return
    raise SystemExit(f"bundle not found under {INPUT}: {[str(x) for x in INPUT.rglob('*')][:20]}")


def check_cuda() -> None:
    """Some Kaggle GPUs (P100) are not supported by recent torch builds: fall back to CPU."""
    import torch
    if not torch.cuda.is_available():
        print("no CUDA, training on CPU")
        return
    try:
        (torch.ones(8, device="cuda") * 2).sum().item()
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:  # noqa: BLE001
        print(f"CUDA unusable ({e}); training on CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> None:
    find_bundle()
    check_cuda()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0"], check=False)
    env = {**os.environ, "PARTVISION_HOME": str(WORK)}
    before = set(os.listdir(WORK / "models"))
    (OUT / "logs").mkdir(exist_ok=True)

    runs = []
    for arch, size, extra in RUNS:
        t0 = time.time()
        cmd = [sys.executable, "-m", "partvision.train", "--manifest", "data/train.csv",
               "--holdout", "data/holdout.csv", "--arch", arch, "--image-size", str(size),
               "--promote", "never", "--notes", f"kaggle compare {arch}@{size} {' '.join(extra)}".strip(), *extra]
        print("\n$", " ".join(cmd), flush=True)
        with open(OUT / "logs" / f"{arch}.log", "w") as log:
            p = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = p.wait()
        runs.append({"arch": arch, "image_size": size, "extra": extra, "returncode": rc, "seconds": round(time.time() - t0)})

    new_versions = sorted(set(os.listdir(WORK / "models")) - before - {"current.json"}, key=lambda v: int(v[1:]))
    summary = []
    for v in new_versions:
        meta = json.loads((WORK / "models" / v / "meta.json").read_text())
        h = meta["holdout"]
        summary.append({"version": v, "arch": meta["arch"], "image_size": meta["image_size"],
                        "top1": h["top1"], "top3": h["top3"], "ece": h["ece"],
                        "top_confusions": h["top_confusions"][:3]})
        shutil.copytree(WORK / "models" / v, OUT / "models" / v)
    current = json.loads((WORK / "models" / "current.json").read_text())["version"]
    ref = json.loads((WORK / "models" / current / "meta.json").read_text())["holdout"]
    result = {"baseline": {"version": current, "top1": ref["top1"], "top3": ref["top3"], "ece": ref["ece"]},
              "runs": runs, "candidates": summary}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
