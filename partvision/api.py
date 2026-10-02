"""HTTP API (FastAPI).

    uvicorn partvision.api:app --reload

Endpoints are plain ``def`` (not async): FastAPI runs them in a thread pool, so
CPU-bound inference does not block the event loop.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config
from .db import Store
from .labels import is_valid_label
from .registry import ModelRegistry
from .service import Predictor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


class FeedbackIn(BaseModel):
    prediction_id: str = Field(min_length=8, max_length=64)
    label: str = Field(min_length=1, max_length=64)


def _read_image(raw: bytes) -> Image.Image:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()                         # detects truncated / non-image files
        img = Image.open(io.BytesIO(raw))    # verify() invalidates the object; reopen
        img.draft("RGB", (1024, 1024))       # fast JPEG downscale for huge phone photos
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, f"not a readable image: {e}") from None


def _credentials_ok(header: str, user: str, password: str) -> bool:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        got_user, _, got_password = base64.b64decode(encoded).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    # compare both fields, constant time, so timing does not reveal which one was wrong
    return secrets.compare_digest(got_user.encode(), user.encode()) & secrets.compare_digest(
        got_password.encode(), password.encode())


def create_app() -> FastAPI:
    app = FastAPI(title="PartVision", version="0.1.0",
                  description="Car part recognition with human-in-the-loop feedback")
    store = Store(config.db_path())
    registry = ModelRegistry(config.models_dir())
    config.upload_dir().mkdir(parents=True, exist_ok=True)
    state: dict = {"predictor": None, "others": {}}   # others: cached non-live versions for /api/compare
    lock = threading.Lock()

    auth = config.basic_auth()
    if auth is not None:
        @app.middleware("http")
        async def require_basic_auth(request: Request, call_next):
            # health stays open for container / load-balancer probes
            if request.url.path != "/api/health" and not _credentials_ok(
                    request.headers.get("authorization", ""), *auth):
                return Response("authentication required", status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="PartVision"'})
            return await call_next(request)

    def load_current() -> Predictor | None:
        d = registry.current_dir()
        return Predictor(d) if d else None

    def predictor() -> Predictor:
        if state["predictor"] is None:
            with lock:
                if state["predictor"] is None:
                    state["predictor"] = load_current()
        if state["predictor"] is None:
            raise HTTPException(503, "no model has been trained/promoted yet")
        return state["predictor"]

    app.mount("/uploads", StaticFiles(directory=config.upload_dir()), name="uploads")
    samples = config.data_dir() / "samples"
    if samples.exists():
        app.mount("/samples", StaticFiles(directory=samples), name="samples")

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": state["predictor"] is not None}

    @app.get("/api/model")
    def model_info():
        p = predictor()
        h = p.meta["holdout"]
        return {
            "version": p.version,
            "arch": p.meta["arch"],
            "classes": p.classes,
            "known_classes": p.known_classes,
            "temperature": p.temperature,
            "unknown_similarity": p.unknown_similarity,
            "holdout": {k: h[k] for k in ("n", "top1", "top3", "ece", "ece_uncalibrated", "top_confusions", "risk_coverage")},
            "versions": registry.versions(),
            "policy": vars(p.policy),
        }

    @app.get("/api/models")
    def list_models():
        """Every registered version with its holdout metrics, for the Compare tab."""
        cur = registry.current_version()
        rows = []
        for v in registry.versions():
            try:
                m = registry.meta(v)
            except (FileNotFoundError, ValueError):
                continue
            h = m.get("holdout", {})
            tc = m.get("train_config", {})
            weights = registry.root / v / "model.pt"
            rows.append({
                "version": v, "current": v == cur, "arch": m.get("arch"), "image_size": m.get("image_size"),
                "pretrained": tc.get("pretrained"), "n_train": m.get("n_train"), "notes": m.get("notes", ""),
                "created_at": m.get("created_at"), "size_mb": round(weights.stat().st_size / 1e6, 1) if weights.exists() else None,
                "top1": h.get("top1"), "top3": h.get("top3"), "ece": h.get("ece"),
                "top_confusions": h.get("top_confusions", [])[:3],
            })
        return rows

    def predictor_for(version: str) -> Predictor:
        """Cached Predictor for any registered version (the live one is reused as-is)."""
        live = state["predictor"]
        if live is not None and live.version == version:
            return live
        with lock:
            cached = state["others"].get(version)
            if cached is None:
                if version not in registry.versions():
                    raise HTTPException(404, f"unknown model version {version}")
                cached = state["others"][version] = Predictor(registry.root / version)
            return cached

    @app.post("/api/compare")
    def compare(file: UploadFile = File(...), versions: str = Form("")):
        """Run one photo through several model versions. Nothing is logged: this is for evaluation."""
        img = _read_image(file.file.read(MAX_UPLOAD_BYTES + 1))
        wanted = [v for v in versions.split(",") if v] or registry.versions()
        out = []
        for v in wanted:
            p = predictor_for(v)
            t0 = time.perf_counter()
            r = p.predict(img)
            ms = (time.perf_counter() - t0) * 1000
            out.append({"version": v, "arch": p.meta["arch"], "latency_ms": round(ms, 1),
                        "decision": r["decision"], "confidence": r["confidence"], "ood_score": r["ood_score"],
                        "candidates": r["candidates"]})
        return out

    @app.get("/api/samples")
    def sample_list():
        if not samples.exists():
            return []
        return sorted(f"/samples/{f.name}" for f in samples.iterdir() if f.suffix.lower() in {".jpg", ".png"})

    @app.post("/api/predict")
    def predict(file: UploadFile = File(...)):
        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        img = _read_image(raw)
        sha = hashlib.sha256(raw).hexdigest()
        path = config.upload_dir() / f"{sha[:24]}.jpg"
        if not path.exists():                       # content-addressed: duplicates stored once
            img.save(path, "JPEG", quality=92)
        result = predictor().predict(img)
        pid = store.log_prediction(str(path), sha, result)
        return {"prediction_id": pid, "image_url": f"/uploads/{path.name}", **result}

    @app.post("/api/feedback")
    def feedback(body: FeedbackIn):
        if not is_valid_label(body.label):
            raise HTTPException(422, "label must be snake_case, e.g. headlight_left")
        try:
            return store.add_feedback(body.prediction_id, body.label)
        except KeyError:
            raise HTTPException(404, "unknown prediction_id") from None

    @app.get("/api/review-queue")
    def review_queue(limit: int = 20):
        rows = store.review_queue(min(max(limit, 1), 200))
        for r in rows:
            r["image_url"] = f"/uploads/{Path(r.pop('image_path')).name}"
        return rows

    @app.post("/api/classes/{label}/references")
    def add_references(label: str, files: list[UploadFile] = File(...)):
        if not is_valid_label(label):
            raise HTTPException(422, "label must be snake_case, e.g. door_handle_left")
        imgs = [_read_image(f.file.read(MAX_UPLOAD_BYTES + 1)) for f in files]
        n = predictor().add_references(imgs, label)
        return {"label": label, "added": n, "known_classes": predictor().known_classes}

    @app.get("/api/stats")
    def stats():
        return store.stats()

    @app.post("/api/admin/reload")
    def reload():
        """Pick up a newly promoted model version without restarting the server."""
        new = load_current()
        with lock:
            state["predictor"] = new
        return {"version": new.version if new else None}

    return app


app = create_app()
