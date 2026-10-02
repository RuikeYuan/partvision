# PartVision

Recognise car parts from a photo and route each one to the right workflow:
**auto-accept**, **confirm from top-3**, **manual**, or **unknown part type**.
Every human confirmation flows back as training data, and a retrained model is
only promoted if it beats the current one on a fixed holdout set.

Built as a technical demo for a parts dismantler workflow: an employee photographs
a part at intake, the system suggests the label, and the employee confirms with one tap.

```
photo ──► ResNet backbone ──► embedding (512/2048-d) ──┬──► classifier head ──► temperature-scaled probs ──┐
                                                       └──► cosine kNN index ──► neighbour votes ───────────┤
                                                                     │                                      ▼
                                                                     └──► k-th NN score (open-set) ──► decision policy
                                                                                                            │
         SQLite: predictions + feedback ◄──── employee confirms / corrects ◄── top-3 UI ◄───────────────────┘
                     │
                     └──► retrain (warm start) ──► eval gate on fixed holdout ──► promote ──► hot reload
```

## Results (synthetic demo data, CPU, ResNet-18 from scratch)

9 classes, 867 long-tailed training images (160 wheel rims, 35 brake discs),
occlusion, blur, grayscale, ±35° rotation. 360-image fixed holdout.

| | v001 (baseline) | v002 (+360 feedback images, warm start) |
|---|---|---|
| Holdout top-1 | 93.6% | 94.2% |
| Holdout top-3 | 97.8% | 98.6% |
| ECE before → after temperature scaling | 0.095 → 0.035 | 0.082 → 0.029 |
| Auto-accept at p ≥ 0.9: coverage / accuracy | 85% / 98.0% | 88% / 98.4% |

Simulated production run (360 new photos through the API, v001):

| decision | share | top-1 accuracy | correct label in top-3 |
|---|---|---|---|
| auto_accept | 87% | 98.4% | 99.7% |
| confirm | 5% | 68.4% | **100%** |
| unknown | 8% | 71.4% | 92.9% |

In other words, 87% of parts need no human input, and when the model is unsure the
correct answer is almost always one tap away.

## Design decisions

**Handedness-aware augmentation** (`data.py`, `labels.py`). Left and right headlights
are mirror images. A standard `RandomHorizontalFlip` would turn a left headlight
into a right one while keeping the label "left", which teaches the model to ignore
exactly the feature that matters. Instead, the flip also swaps the label
(`headlight_left ↔ headlight_right`); symmetric parts keep theirs. As a bonus,
every left photo is also a valid right photo, which doubles data for sided parts.

**Class-balanced sampling.** Inventories are long-tailed. A `WeightedRandomSampler`
with inverse-frequency weights makes every class equally likely per batch.

**Two-stage fine-tuning** (pretrained or warm start). Stage 1 trains only the new
head with a frozen backbone (BatchNorm in eval mode); a random head would otherwise
send large gradients into good pretrained features. Stage 2 unfreezes everything
with backbone LR = 0.1 × head LR. AdamW, cosine schedule, label smoothing, early
stopping on validation top-1.

**Calibration** (`calibration.py`). The decision thresholds only make sense if
"90%" means right 90% of the time. A single temperature T is fit on validation
logits with LBFGS (Guo et al., 2017). Here T = 0.53: the model was *under*-confident,
a known side effect of label smoothing. Temperature scaling never changes the argmax,
so accuracy is unaffected.

**Risk–coverage instead of one accuracy number.** The business question is
"what share can we automate, and how accurate is that share?" The model page shows
the curve, so the auto-accept threshold becomes a business choice.

**kNN embedding index** (`index.py`). It serves as a second opinion: auto-accept
requires the classifier and the nearest neighbours to agree. It also explains the
prediction ("looks like these stored photos"), and it lets you register a new part
type from a few reference photos without retraining. Embeddings are mean-centred
before L2 normalisation, because post-ReLU features are all non-negative and raw
cosine similarities bunch up at 0.93–0.99.

**Open-set detection.** The score is the similarity to the 5th nearest training
neighbour (Sun et al., 2022, "OOD Detection with Deep Nearest Neighbors"). The
threshold is set on validation data so that 95% of known parts pass.

**Active learning.** `/api/review-queue` lists unlabelled predictions sorted by
smallest top-1/top-2 margin, so human time goes to the most informative photos.

**Model registry + gate** (`registry.py`). Each training run is a versioned folder
(weights, index, metadata, metrics, data hashes). A new version is promoted only if
(a) the holdout set is identical (checked by hash), (b) top-1 did not drop by more
than 1 point, and (c) ECE ≤ 0.10. Rollback means pointing `current.json` at an older
version. Feedback images that are byte-identical to holdout images are dropped
before retraining, to prevent leakage.

**Serving.** FastAPI with sync endpoints, so CPU-bound inference runs in the thread
pool rather than blocking the event loop. Uploads are validated (`PIL.verify`,
10 MB cap) and stored content-addressed (SHA-256). SQLite runs in WAL mode with one
connection per request; the same schema works on PostgreSQL. `/api/admin/reload`
hot-swaps the model after a promotion.

## Known limitations (and what I'd do with real data)

- **Synthetic data, no pretrained weights.** The sandbox this was built in could not
  download ImageNet weights, so the demo trains ResNet-18 from scratch. With real
  photos, use `--arch resnet50` with pretrained weights (the default config).
- **Open-set detection and new classes are weak with this backbone.** Only 5/10
  unseen door handles were flagged as unknown. Registering door handles from 5
  reference photos did *not* make them recognisable: they still sat closer to wheel
  rims (similarity 0.87–0.96) than to each other (0.18–0.83). The cause is that a
  classifier trained on 9 classes learns features that separate *those 9 classes
  only* ("feature collapse"). The fix is to build the kNN index on a general-purpose
  embedding (ImageNet-pretrained, or better self-supervised DINOv2/CLIP), kept
  separate from the classifier. The human confirm step is the safety net meanwhile.
- **The unknown threshold is noisy.** It was estimated on about 130 validation images
  (10% false "unknown" on holdout vs 5% target). More validation data would stabilise it.
- **Part numbers.** Visual classification gives the part *type*. The exact OEM number
  is better read with OCR from the label or stamping, combined with the donor
  vehicle's data (e.g. RDW lookup by licence plate).

## Run it

```bash
pip install -r requirements.txt          # CPU torch is fine

# 1. data (or point a manifest CSV "path,label" at real photos)
python scripts/make_synthetic_data.py --out data --per-class 160

# 2. train + register (pretrained ResNet-50 on a real machine with internet)
python -m partvision.train --manifest data/train.csv --holdout data/holdout.csv
#   offline / quick demo:
python -m partvision.train --manifest data/train.csv --holdout data/holdout.csv \
    --arch resnet18 --no-pretrained --image-size 112 --epochs 16 --lr 2e-3

# 3. serve (UI at http://localhost:8000, API docs at /docs)
uvicorn partvision.api:app --reload

# 4. simulate a day of warehouse use, then retrain on the feedback
python scripts/simulate_production.py --per-class 40
python -m partvision.retrain --base data/train.csv --holdout data/holdout.csv \
    --no-pretrained --image-size 112 --epochs 8 --lr 1e-3
curl -X POST localhost:8000/api/admin/reload

pytest -q                                # 14 tests
```

The repo ships with model `v002` already trained, so step 3 works right away.

## Deploy (single server, Docker + Caddy HTTPS)

Needs one Linux server (2 vCPU / 4 GB RAM is enough for CPU inference) with Docker,
ports 80/443 open, and a domain whose A record points at the server.

```bash
# on your machine: copy code + model versions (data/ only if you want existing feedback)
rsync -av --exclude data --exclude build --exclude .pytest_cache ./ user@server:/opt/partvision/

# on the server
cd /opt/partvision
cp .env.example .env        # set DOMAIN and PARTVISION_PASSWORD (openssl rand -base64 24)
docker compose up -d --build
docker compose logs -f app  # wait for "Application startup complete"
```

The app then runs at `https://$DOMAIN` behind HTTP Basic auth. Compose refuses to start
without `PARTVISION_PASSWORD`. `models/` and `data/` are bind-mounted, so a promoted model
or collected feedback survives `docker compose up -d --build`. Back up both directories.
Run a single app instance: the SQLite DB and the reference-photo index are per process.

## Deploy for free (GitHub Actions + Hugging Face Space)

GitHub holds the code, an HF model repo holds the weights, and the Space serves the app.
Every push to `main` runs the tests and, if they pass, redeploys the Space
([.github/workflows/deploy.yml](.github/workflows/deploy.yml)).

```bash
# once, on your machine: put the live model version into an HF model repo
hf auth login
hf repo create <user>/partvision-models --repo-type model --private
hf upload <user>/partvision-models models/v004 v004
hf upload <user>/partvision-models models/current.json current.json
```

GitHub repo settings, under Secrets and variables, then Actions:

| kind | name | value |
|---|---|---|
| secret | `HF_TOKEN` | HF token with write access |
| variable | `HF_SPACE` | `<user>/partvision` |
| variable | `HF_MODEL_REPO` | `<user>/partvision-models` |
| variable | `HF_DATA_REPO` | `<user>/partvision-data` (optional, see below) |

After the first deploy, open the Space settings, then Variables and secrets, and add the
secrets `HF_TOKEN` (needed for the private model repo and for data sync) and
`PARTVISION_PASSWORD`.

The Space disk is wiped on every restart. With `HF_DATA_REPO` set,
[partvision/hf_space.py](partvision/hf_space.py) restores the last snapshot of the
SQLite DB, uploaded photos and added reference photos on start, and pushes a new snapshot
to that private dataset repo every 5 minutes (`SYNC_EVERY_MIN`). Anything written after
the last snapshot is lost on restart. A free Space also sleeps after about 48 hours
without traffic and takes about a minute to wake up.

## API

| method | path | purpose |
|---|---|---|
| POST | `/api/predict` | multipart image → top-3, decision, uncertainty, neighbours |
| POST | `/api/feedback` | `{prediction_id, label}`: confirm or correct |
| GET | `/api/review-queue` | active-learning queue |
| POST | `/api/classes/{label}/references` | add reference photos for a new part type |
| GET | `/api/model` | version, holdout metrics, risk–coverage, confusions |
| GET | `/api/stats` | online accuracy per model version from feedback |
| POST | `/api/admin/reload` | serve the newly promoted version |

## Layout

```
partvision/
  labels.py        left/right taxonomy, flip map
  data.py          manifests, stratified split, transforms, flip-aware dataset, balanced sampler
  model.py         ResNet backbone + separate head
  train.py         two-stage training, calibration, index, OOD threshold, registration
  calibration.py   temperature scaling, ECE
  metrics.py       top-k, confusions, risk–coverage
  index.py         centred cosine kNN, k-th NN OOD score
  service.py       Predictor: fusion + decision policy
  registry.py      versions, promotion gate
  db.py            SQLite predictions/feedback, review queue, online stats
  api.py           FastAPI app + static UI
  retrain.py       feedback → leakage check → warm-start retrain → gate
scripts/           synthetic data generator, production simulator
tests/             unit + API tests
```
