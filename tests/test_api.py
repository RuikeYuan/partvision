import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from partvision.index import EmbeddingIndex
from partvision.model import PartNet
from partvision.registry import ModelRegistry

CLASSES = ["headlight_left", "headlight_right", "wheel_rim"]


def _png(color, size=64):
    import io
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PARTVISION_HOME", str(tmp_path))
    torch.manual_seed(0)
    reg = ModelRegistry(tmp_path / "models")
    vdir = reg.new_version_dir()
    model = PartNet("resnet18", len(CLASSES)).eval()
    torch.save(model.state_dict(), vdir / "model.pt")
    rng = np.random.default_rng(0)
    EmbeddingIndex(np.abs(rng.normal(size=(30, model.embed_dim))), [CLASSES[i % 3] for i in range(30)]).save(vdir / "index.npz")
    meta = {"version": vdir.name, "classes": CLASSES, "arch": "resnet18", "image_size": 64,
            "temperature": 1.5, "unknown_similarity": 0.0,
            "holdout": {"n": 3, "top1": 0.5, "top3": 1.0, "ece": 0.05, "ece_uncalibrated": 0.1,
                        "top_confusions": [], "risk_coverage": []}}
    (vdir / "meta.json").write_text(json.dumps(meta))
    reg.promote(vdir.name)
    from partvision.api import create_app
    return TestClient(create_app())


def test_predict_feedback_and_queue(client):
    r = client.post("/api/predict", files={"file": ("a.png", _png((200, 30, 30)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["candidates"]) == 3
    assert body["decision"] in {"auto_accept", "confirm", "manual", "unknown"}
    assert abs(sum(c["classifier_prob"] for c in body["candidates"]) - 1.0) < 1e-3

    if body["decision"] != "auto_accept":
        queue = client.get("/api/review-queue").json()
        assert queue[0]["id"] == body["prediction_id"]

    fb = client.post("/api/feedback", json={"prediction_id": body["prediction_id"], "label": "wheel_rim"})
    assert fb.status_code == 200
    assert client.get("/api/review-queue").json() == []
    stats = client.get("/api/stats").json()
    assert stats["per_version"][0]["with_feedback"] == 1


def test_rejects_non_image_and_bad_label(client):
    assert client.post("/api/predict", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 415
    pid = client.post("/api/predict", files={"file": ("a.png", _png((0, 0, 0)), "image/png")}).json()["prediction_id"]
    assert client.post("/api/feedback", json={"prediction_id": pid, "label": "Bad Label!"}).status_code == 422
    assert client.post("/api/feedback", json={"prediction_id": "0" * 32, "label": "wheel_rim"}).status_code == 404


def test_add_reference_class(client):
    files = [("files", (f"{i}.png", _png((10, 200, 10)), "image/png")) for i in range(3)]
    r = client.post("/api/classes/door_handle/references", files=files)
    assert r.status_code == 200
    assert "door_handle" in r.json()["known_classes"]
    # the exact same image should now hit the new class through kNN
    body = client.post("/api/predict", files={"file": ("q.png", _png((10, 200, 10)), "image/png")}).json()
    assert body["neighbours"][0]["label"] == "door_handle"
    assert body["neighbours"][0]["similarity"] > 0.99


def test_basic_auth_when_password_set(client, monkeypatch):
    monkeypatch.setenv("PARTVISION_PASSWORD", "s3cret")
    from partvision.api import create_app
    c = TestClient(create_app())
    assert c.get("/api/health").status_code == 200          # probes stay open
    assert c.get("/api/stats").status_code == 401
    assert c.get("/api/stats", auth=("admin", "wrong")).status_code == 401
    assert c.get("/api/stats", auth=("admin", "s3cret")).status_code == 200
