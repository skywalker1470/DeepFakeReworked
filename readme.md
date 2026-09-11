
# DeepFake Detector

**⚠️READ BEFORE TESTING: this model is trained on Celeb-DF v2, whose fakes are generated with a modified autoencoder-based face-swap pipeline (an enhanced FakeApp/DFaker-style method — refined resolution, color matching, and temporal smoothing, NOT a GAN or diffusion model). Deepfake detectors are known to generalize poorly across generation methods, so a video made with a different technique (a GAN-based face-swap app, a diffusion model, etc.) may be misclassified as real. For a meaningful test of this demo, use a video generated with a similar autoencoder-based face-swap method, or one of the sample videos in [`videos/`](videos/). This is a well-documented limitation of the field, not specific to this implementation — see the [Celeb-DF paper](https://arxiv.org/abs/1909.12962) for details.**

Upload a video, get a real/fake verdict with confidence scores and annotated playback — a face-level deepfake detector trained on **Celeb-DF v2**, fine-tuning **EfficientNet-B0** on **YOLOv8n**-cropped face frames, deployed live on AWS.

**🔴 Live demo: [http://51.21.152.188:8000](http://51.21.152.188:8000)** — upload your own video and see it detect real vs. fake in real time.

![Sample detection output](videos/output_003_000.gif)

## Results

Video-level evaluation on the held-out Celeb-DF v2 test split:

| Metric | Value |
|---|---|
| Accuracy (tuned threshold) | **94.8%** |
| ROC-AUC | **0.986** |
| Accuracy (default 0.5 threshold) | 92.6% |

## What this demonstrates

- **Applied deep learning**: transfer learning (EfficientNet-B0 on ImageNet weights), a two-stage face-detection + classification pipeline (YOLOv8n-face → EfficientNet-B0), careful attention to normalization and data leakage (video-level train/val/test splits, not frame-level).
- **Full ML lifecycle**: dataset prep → training → evaluation/threshold tuning → productionized inference, not just a notebook.
- **Real deployment**: containerized with Docker, running live on AWS EC2, with actual constraints handled (a 1GB RAM instance needed swap space and careful memory budgeting to run two models reliably) rather than assumed away.
- **REST API design**: a separate FastAPI service (see `app/`) with `/health`, `/predict`, `/metrics` endpoints, tested (`tests/test_api.py`), and CI-deployable via GitHub Actions.
- **Awareness of real limitations**: this model generalizes well within its training distribution (Celeb-DF v2's autoencoder-based face-swap method) but, like all deepfake detectors, degrades on out-of-distribution generation methods — a known, documented constraint discussed above rather than glossed over. Extending training to multiple generation methods (e.g. FaceForensics++, DFDC) is a natural next step for improving cross-method generalization.

## How it works

1. **`data_create.py`**: downloads the Celeb-DF v2 dataset from Kaggle Hub (`reubensuju/celeb-df-v2`).
2. **`extract_frames.py`**: for every video in `Celeb-real`, `YouTube-real`, and `Celeb-synthesis`, uses **YOLOv8n-face** (`ultralytics`, weights from [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face)) to detect and crop 2 face frames (first frame + middle frame) and saves them as JPEGs under a `frames/` directory.
3. **`prepare_celebdf_list.py`**: builds train/val/test split text files (`data_list/celebdf_{train,val,test}.txt`), each line being `<frame_path> <label>` (0 = real, 1 = fake). Splitting is done **by video** (not by frame) to avoid leakage, and the official Celeb-DF test list is respected. Dataset size is capped via `MAX_VIDEOS_PER_CLASS`.
4. **`train_celebdf_xception.py`**: fine-tunes `torchvision.models.efficientnet_b0` (ImageNet weights) with a replaced binary classification head. Inputs are resized to 224×224 and normalized with ImageNet mean/std. Freezes the first half of the feature layers, uses weighted `BCEWithLogitsLoss`, AMP mixed precision, and saves the best checkpoint to `output/best_model.pth` based on validation accuracy.
5. **`testing.py`**: evaluates the trained model on the held-out test split, aggregating frame-level probabilities to **video-level** predictions (mean pooling), then reports accuracy, ROC-AUC, a confusion matrix at threshold 0.5, and a scan for the best accuracy threshold.
6. **`app.py`**: the Flask app behind the live demo above. Uploads a video, runs YOLOv8n-face and the trained EfficientNet-B0 on every 5th frame, overlays a REAL/FAKE label and confidence per sampled frame, re-encodes the output to browser-compatible H.264 via `ffmpeg`, and renders a verdict plus fake/real frame percentages.
7. **`face_detector.py`**: shared `YOLOFaceDetector` wrapper used by both `extract_frames.py` and `app.py` — runs YOLOv8n-face, picks the largest detected face per frame, crops with margin, and resizes.
8. **`app/`**: a separate FastAPI REST service wrapping the same model + detector, demonstrating REST API design independent of the demo UI (see below).

## Running it yourself

### Requirements

- Python 3.10–3.12 (PyTorch has no stable wheels for 3.13/3.14 yet)
- **`ffmpeg`** available on your `PATH` (required by `app.py` to re-encode annotated output for browser playback)
- An NVIDIA GPU with CUDA is strongly recommended for training/extraction, though everything falls back to CPU
- A [Kaggle account](https://www.kaggle.com/) configured for `kagglehub` if you want to download the dataset yourself (`data_create.py`)
- YOLOv8n-face weights (`yolov8n-face-lindevs.pt`) from [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face), placed at the project root

### Setup

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# Install PyTorch (CUDA 12.1 build) first
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Building the model yourself

Run these in order from the project root:

```bash
python data_create.py              # download Celeb-DF v2 via kagglehub
python extract_frames.py           # YOLOv8-face crop extraction -> frames/
python prepare_celebdf_list.py     # build train/val/test split lists -> data_list/
python train_celebdf_xception.py   # fine-tune EfficientNet-B0 -> output/best_model.pth
python testing.py                  # evaluate on the held-out test set
```

Useful flags (all scripts use `argparse` with sensible defaults):

- `extract_frames.py --dataset_root ./celebdf --frames_root ./frames`
- `prepare_celebdf_list.py --dataset_root ./celebdf --frames_root ./frames --output_dir ./data_list --val_split 0.1`
- `train_celebdf_xception.py --train_list ./data_list/celebdf_train.txt --val_list ./data_list/celebdf_val.txt --epochs 20 --batch_size 16 --lr 1e-4 --output_dir ./output`

A pretrained checkpoint is already included at [output/best_model.pth](output/best_model.pth), so the training steps above are optional if you just want to run the app.

### Running the web app locally

```bash
python app.py
```

Then open **http://127.0.0.1:5000**, upload a video (`.mp4`, `.avi`, `.mov`, `.mkv`), and click **Run Analysis**. The app will:

- Sample every 5th frame (`FRAME_SKIP = 5`)
- Run YOLOv8n-face detection + EfficientNet-B0 classification (`THRESHOLD = 0.74`, tuned via `testing.py`)
- Draw a REAL/FAKE label + confidence on each processed frame
- Re-encode the output with `ffmpeg` for browser playback
- Show the verdict, fake/real frame percentages, and the annotated video inline

Uploaded videos are saved to `uploads/`, processed/annotated output to `processed/` (both created automatically). Requests are rate-limited (5/minute per IP, see `ratelimit.py`).

### Running the REST API (FastAPI)

A separate, cloud-deployable REST API lives under `app/`, wrapping the same model and detector — this is the "productionized service" half of the project, independent of the demo UI above.

```bash
uvicorn app.main:app --reload
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Accepts a video file, returns prediction (real/fake) with confidence |
| `GET` | `/metrics` | Returns request count and average latency |

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@sample_video.mp4"
```

#### Docker

```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

Or with Compose:

```bash
docker compose up --build
```

#### Tests

```bash
pytest tests/
```

#### Cloud deployment (AWS EC2)

The live demo above runs on an AWS EC2 `t3.micro` instance (free tier). Its 1GB RAM is tight for a PyTorch workload with two models loaded (EfficientNet-B0 + YOLOv8n-face measure ~700MB RSS on their own), so the instance has a 2GB swapfile added as a safety margin against OOM under load, and only one service (Flask or FastAPI) runs at a time to fit the memory budget. See [idea.txt](idea.txt) for the full rationale.

`deploy.sh` automates build → push → deploy → smoke-test for the FastAPI service against the EC2 instance running Docker. It expects `REGISTRY`, `IMAGE_NAME`, `VM_HOST`, and `VM_KEY` environment variables — see the script header for details.

```bash
./deploy.sh
```

A GitHub Actions workflow (`.github/workflows/deploy.yml`) can run this same flow automatically on push to `main`, given the right repo secrets.

## Project structure

```
app.py                        Flask web app for interactive inference (the live demo)
ratelimit.py                   Shared per-IP rate limiter used by both app.py and the FastAPI service
templates/index.html           Web UI (upload form + results)
Dockerfile.flask               Container image for the Flask demo
app_flask_requirements.txt     Lean inference-only dependencies for the Flask demo image
app/                           FastAPI REST service (cloud-deployable)
  main.py                        Endpoints: /health, /predict, /metrics
  model.py                        Model loading + inference
  preprocess.py                   Frame extraction + face crop generator
  ratelimit.py                    FastAPI adapter over the shared rate limiter
  requirements.txt                Lean inference-only dependencies for the API image
face_detector.py               Shared YOLOv8n-face detector wrapper
data_create.py                 Downloads Celeb-DF v2 via kagglehub
extract_frames.py              YOLOv8-face crop extraction from raw videos
prepare_celebdf_list.py        Builds train/val/test split lists
train_celebdf_xception.py      Trains the EfficientNet-B0 classifier
testing.py                     Video-level evaluation (accuracy, AUC, confusion matrix)
output/best_model.pth          Pretrained model checkpoint
videos/                        Sample test videos
tests/test_api.py              FastAPI endpoint tests
Dockerfile                     Container image for the FastAPI service
docker-compose.yml             Local Docker Compose setup
deploy.sh                      Build/push/deploy/smoke-test automation for AWS EC2
requirements.txt               Python dependencies (full dev/training set)
```

> Note: some filenames/strings (`train_celebdf_xception.py`) reference the project's original Xception backbone. The model was later switched to **EfficientNet-B0**, and the filename just wasn't updated.
