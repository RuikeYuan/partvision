"""Generate a synthetic car-part photo dataset for demos and tests.

Real photos from the parts inventory would replace this. The synthetic set is
built so the interesting problems are present:
  * left/right parts are exact mirror images (handedness matters),
  * varied background, lighting, rotation, scale, blur and noise,
  * one extra part type (door_handle) that is NOT in training, to demonstrate
    open-set detection and adding a class via reference photos.

    python scripts/make_synthetic_data.py --out data --per-class 140
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

S = 192  # canvas size


def _paint(rng):
    return rng.choice([(180, 30, 30), (30, 60, 140), (220, 220, 220), (30, 30, 30), (90, 90, 95), (20, 90, 60)])


def _jit(c, rng, d=18):
    return tuple(max(0, min(255, v + rng.randint(-d, d))) for v in c)


def headlight(d: ImageDraw.ImageDraw, rng):
    # swept housing: tall on the left (inner) end, tapering to the right (outer) end
    d.polygon([(28, 62), (150, 72), (168, 100), (160, 118), (34, 132)], fill=_jit((45, 45, 50), rng), outline=(20, 20, 20))
    d.ellipse((42, 76, 92, 126), fill=_jit((205, 215, 225), rng), outline=(120, 120, 130), width=3)   # main lens
    d.ellipse((100, 86, 128, 114), fill=_jit((190, 200, 210), rng))                                # small lens
    d.line([(40, 128), (150, 112)], fill=_jit((240, 240, 255), rng), width=5)                      # DRL strip


def taillight(d, rng):
    d.polygon([(24, 70), (166, 58), (166, 96), (40, 128)], fill=_jit((170, 20, 25), rng), outline=(60, 10, 10))
    d.rectangle((126, 66, 160, 92), fill=_jit((235, 235, 235), rng))       # reverse light on outer end
    d.polygon([(40, 100), (80, 92), (80, 112), (48, 120)], fill=_jit((230, 140, 20), rng))  # indicator


def side_mirror(d, rng):
    d.rounded_rectangle((30, 50, 150, 130), radius=34, fill=_jit(_paint(rng), rng), outline=(20, 20, 20), width=2)
    d.rounded_rectangle((42, 60, 138, 120), radius=26, fill=_jit((170, 185, 195), rng))  # glass
    d.polygon([(138, 110), (172, 120), (172, 150), (130, 140)], fill=(25, 25, 25))         # mount arm on one side


def wheel_rim(d, rng):
    cx = cy = S // 2
    r = rng.randint(62, 74)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_jit((200, 200, 205), rng), outline=(120, 120, 125), width=4)
    n = rng.choice([5, 6, 7, 10])
    off = rng.random() * math.tau
    for i in range(n):
        a = off + i * math.tau / n
        d.line([(cx, cy), (cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a))], fill=_jit((150, 150, 155), rng), width=9)
    d.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=_jit((90, 90, 95), rng))


def brake_disc(d, rng):
    cx = cy = S // 2
    r = rng.randint(64, 74)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_jit((110, 100, 95), rng), outline=(70, 60, 55), width=3)
    d.ellipse((cx - 34, cy - 34, cx + 34, cy + 34), fill=_jit((80, 80, 85), rng))
    for i in range(5):
        a = i * math.tau / 5
        x, y = cx + 20 * math.cos(a), cy + 20 * math.sin(a)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(30, 30, 30))
    for i in range(18):
        a = i * math.tau / 18
        x, y = cx + (r - 14) * math.cos(a), cy + (r - 14) * math.sin(a)
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(40, 35, 35))


def alternator(d, rng):
    cx = cy = S // 2
    d.ellipse((cx - 62, cy - 58, cx + 62, cy + 66), fill=_jit((175, 175, 170), rng), outline=(100, 100, 100), width=3)
    for i in range(16):
        a = i * math.tau / 16
        d.line([(cx + 36 * math.cos(a), cy + 4 + 36 * math.sin(a)), (cx + 56 * math.cos(a), cy + 4 + 56 * math.sin(a))], fill=(60, 60, 60), width=5)
    d.ellipse((cx - 22, cy - 18, cx + 22, cy + 26), fill=_jit((50, 50, 55), rng))
    for k in range(3):
        d.ellipse((cx - 18 + k * 5, cy - 14 + k * 5, cx + 18 - k * 5, cy + 22 - k * 5), outline=(120, 120, 120), width=1)
    d.rectangle((cx - 12, cy - 82, cx + 12, cy - 56), fill=_jit((160, 160, 155), rng))   # mounting ear
    d.ellipse((cx - 6, cy - 76, cx + 6, cy - 64), fill=(40, 40, 40))


def door_handle(d, rng):
    d.rounded_rectangle((30, 80, 162, 112), radius=16, fill=_jit((210, 210, 215), rng), outline=(120, 120, 125), width=3)
    d.ellipse((40, 88, 56, 104), fill=(40, 40, 40))      # key cylinder on one end
    d.rectangle((60, 92, 150, 100), fill=_jit((150, 150, 155), rng))


BASE = {"headlight": headlight, "taillight": taillight, "side_mirror": side_mirror}
SYMMETRIC = {"wheel_rim": wheel_rim, "brake_disc": brake_disc, "alternator": alternator}
OPEN_SET = {"door_handle": door_handle}


def classes() -> list[str]:
    sided = [f"{b}_{s}" for b in BASE for s in ("left", "right")]
    return sorted(sided + list(SYMMETRIC))


def _background(rng) -> Image.Image:
    base = (rng.randint(150, 235), rng.randint(145, 230), rng.randint(135, 220))
    bg = Image.new("RGB", (S, S), base)
    d = ImageDraw.Draw(bg)
    for _ in range(rng.randint(3, 10)):       # clutter: shelf edges, cables, shadows
        c = _jit(base, rng, 60)
        d.line([(rng.randint(0, S), rng.randint(0, S)), (rng.randint(0, S), rng.randint(0, S))], fill=c, width=rng.randint(1, 6))
    return bg


def render(label: str, rng: random.Random) -> Image.Image:
    if label.endswith("_left") or label.endswith("_right"):
        draw_fn = BASE[label.rsplit("_", 1)[0]]
    else:
        draw_fn = {**SYMMETRIC, **OPEN_SET}[label]
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer), rng)
    if label.endswith("_right"):
        layer = ImageOps.mirror(layer)          # right part = mirror image of left part
    scale = rng.uniform(0.5, 1.1)
    layer = layer.resize((int(S * scale), int(S * scale)), Image.BICUBIC).rotate(rng.uniform(-35, 35), Image.BICUBIC, expand=False)
    img = _background(rng)
    ox = rng.randint(-24, 24) + (S - layer.width) // 2
    oy = rng.randint(-24, 24) + (S - layer.height) // 2
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow.putalpha(layer.getchannel("A").point(lambda a: int(a * 0.35)))
    img.paste(shadow, (ox + 5, oy + 6), shadow)
    img.paste(layer, (ox, oy), layer)
    # occlusion: hands, other parts, packaging
    d = ImageDraw.Draw(img)
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        x, y = rng.randint(0, S - 30), rng.randint(0, S - 30)
        w, h = rng.randint(25, 80), rng.randint(25, 80)
        d.rectangle((x, y, x + w, y + h), fill=_jit((rng.randint(60, 220),) * 3, rng, 40))
    # lighting + camera effects
    img = Image.eval(img, lambda v, g=rng.uniform(0.5, 1.3), b=rng.randint(-30, 30): max(0, min(255, int(v * g + b))))
    if rng.random() < 0.15:
        img = ImageOps.grayscale(img).convert("RGB")   # removes colour cues (red taillight)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.8)))
    noise = Image.effect_noise((S, S), rng.uniform(8, 30)).convert("RGB")
    return Image.blend(img, noise, rng.uniform(0.05, 0.22))


def generate(out: Path, per_class: int, holdout_per_class: int, samples_per_class: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    splits = {"train": per_class, "holdout": holdout_per_class, "samples": samples_per_class}
    # long-tailed like a real inventory: lots of wheels and mirrors, few alternators
    train_share = {"wheel_rim": 1.0, "side_mirror_left": 0.85, "side_mirror_right": 0.85,
                   "headlight_left": 0.65, "headlight_right": 0.65, "taillight_left": 0.45,
                   "taillight_right": 0.45, "alternator": 0.3, "brake_disc": 0.22}
    for split, n_split in splits.items():
        rows = []
        for label in classes():
            n = max(8, int(n_split * train_share[label])) if split == "train" else n_split
            d = out / "images" / split / label if split != "samples" else out / "samples"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                name = f"{label}_{i:04d}.jpg" if split != "samples" else f"{label}_{i}.jpg"
                path = d / name
                render(label, rng).save(path, "JPEG", quality=rng.randint(70, 95))
                rows.append((path.relative_to(out).as_posix(), label))
        if split != "samples":
            with (out / f"{split}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "label"])
                w.writerows(rows)
    # open-set demo: an unseen part type
    nd = out / "new_part_demo" / "door_handle"
    nd.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        render("door_handle", rng).save(nd / f"door_handle_{i}.jpg", "JPEG", quality=90)
    print(f"wrote {len(classes())} classes x {per_class} train / {holdout_per_class} holdout to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-class", type=int, default=140)
    ap.add_argument("--holdout-per-class", type=int, default=40)
    ap.add_argument("--samples-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(Path(a.out), a.per_class, a.holdout_per_class, a.samples_per_class, a.seed)
