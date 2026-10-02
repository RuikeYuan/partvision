import random

import numpy as np
import pytest
import torch
from PIL import Image

from partvision.calibration import expected_calibration_error, fit_temperature
from partvision.data import PartsDataset, stratified_split
from partvision.labels import build_flip_map, is_valid_label, mirror_label
from partvision.registry import passes_gate


def test_mirror_label():
    assert mirror_label("headlight_left") == "headlight_right"
    assert mirror_label("headlight_right") == "headlight_left"
    assert mirror_label("wheel_rim") == "wheel_rim"


def test_flip_map_blocks_sided_class_without_mirror():
    classes = ["door_left", "headlight_left", "headlight_right", "wheel_rim"]
    assert build_flip_map(classes) == [-1, 2, 1, 3]


def test_label_validation():
    assert is_valid_label("side_mirror_left")
    assert not is_valid_label("Side Mirror")
    assert not is_valid_label("../etc")


def _asym_image(path):
    img = Image.new("RGB", (8, 8), "black")
    img.putpixel((0, 0), (255, 0, 0))          # marker in the top-left corner
    img.save(path)


def test_flip_swaps_label_and_mirrors_pixels(tmp_path):
    p = tmp_path / "a.png"
    _asym_image(p)
    classes = ["headlight_left", "headlight_right"]
    to_t = lambda im: torch.from_numpy(np.asarray(im).copy())   # (H, W, 3)
    ds = PartsDataset([(p, "headlight_left")], classes, to_t, flip_p=1.0)
    x, y = ds[0]
    assert classes[y] == "headlight_right"
    assert x[0, 7, 0] == 255 and x[0, 0, 0] == 0   # marker moved to top-right


def test_symmetric_class_keeps_label_when_flipped(tmp_path):
    p = tmp_path / "a.png"
    _asym_image(p)
    ds = PartsDataset([(p, "wheel_rim")], ["wheel_rim"], lambda im: im, flip_p=1.0)
    assert ds[0][1] == 0


def test_stratified_split_keeps_rare_class_in_val():
    items = [(f"{i}.jpg", "common") for i in range(50)] + [("r1.jpg", "rare"), ("r2.jpg", "rare")]
    tr, va = stratified_split(items, 0.15, seed=0)
    assert any(lbl == "rare" for _, lbl in va) and any(lbl == "rare" for _, lbl in tr)
    assert len(tr) + len(va) == len(items)


def test_temperature_scaling_fixes_overconfidence():
    torch.manual_seed(0)
    n, c = 2000, 5
    labels = torch.randint(0, c, (n,))
    logits = torch.randn(n, c)
    # make 70% of predictions correct, then blow logits up -> overconfident
    correct = torch.rand(n) < 0.7
    logits[correct, labels[correct]] += 2.0
    logits *= 4
    before = expected_calibration_error(torch.softmax(logits, 1), labels)
    t = fit_temperature(logits, labels)
    after = expected_calibration_error(torch.softmax(logits / t, 1), labels)
    assert t > 1.5
    assert after < before / 2


def test_temperature_accepts_inference_mode_tensors():
    # regression: logits collected under torch.inference_mode() broke LBFGS
    with torch.inference_mode():
        logits, labels = torch.randn(40, 3) * 5, torch.randint(0, 3, (40,))
    assert fit_temperature(logits, labels) > 0


def test_gate():
    cur = {"holdout": {"top1": 0.90, "ece": 0.03}, "holdout_hash": "h"}
    ok, _ = passes_gate({"holdout": {"top1": 0.905, "ece": 0.04}, "holdout_hash": "h"}, cur)
    assert ok
    ok, reasons = passes_gate({"holdout": {"top1": 0.85, "ece": 0.04}, "holdout_hash": "h"}, cur)
    assert not ok and "dropped" in reasons[0]
    ok, reasons = passes_gate({"holdout": {"top1": 0.95, "ece": 0.04}, "holdout_hash": "other"}, cur)
    assert not ok
