
# DeepFake Detector

A face-level deepfake detection pipeline trained on **Celeb-DF v2**. It fine-tunes an **EfficientNet-B0** binary classifier on YOLOv8-cropped face frames, and ships both a Flask demo app and a FastAPI REST service (Dockerized, deployable to AWS) for running inference on uploaded videos.

> Note: some filenames/strings (`train_celebdf_xception.py`, the page title "XceptionNet") reference the project's original Xception backbone. The model was later switched to **EfficientNet-B0**, and the naming just wasn't updated.

## Demo

![Sample detection output](videos/output_003_000.gif)

## Results

Video-level evaluation on the held-out Celeb-DF v2 test split (`testing.py`):

| Metric | Value |
|---|---|
| Accuracy (threshold 0.5) | 92.6% |
| ROC-AUC | 0.986 |
| Accuracy (tuned threshold 0.74) | 94.8% |

Confusion matrix at the tuned threshold (rows = actual, columns = predicted; `[real, fake]`):

```
[[171   7]
 [  7  85]]
```

These numbers reflect the current pipeline: YOLOv8n face detection + ImageNet-normalized inputs. An earlier version without input normalization performed substantially worse, since normalization is required for the ImageNet-pretrained EfficientNet-B0 backbone to transfer properly.

## How it works

1. **`data_create.py`**: downloads the Celeb-DF v2 dataset from Kaggle Hub (`reubensuju/celeb-df-v2`).
2. **`extract_frames.py`**: for every video in `Celeb-real`, `YouTube-real`, and `Celeb-synthesis`, uses **YOLOv8n-face** (`ultralytics`, weights from [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face)) to detect and crop 2 face frames (first frame + middle frame) and saves them as JPEGs under a `frames/` directory.
3. **`prepare_celebdf_list.py`**: builds train/val/test split text files (`data_list/celebdf_{train,val,test}.txt`), each line being `<frame_path> <label>` (0 = real, 1 = fake). Splitting is done **by video** (not by frame) to avoid leakage, and the official Celeb-DF test list is respected. Dataset size is capped via `MAX_VIDEOS_PER_CLASS`.
4. **`train_celebdf_xception.py`**: fine-tunes `torchvision.models.efficientnet_b0` (ImageNet weights) with a replaced binary classification head. Inputs are resized to 224×224 and normalized with ImageNet mean/std. Freezes the first half of the feature layers, uses weighted `BCEWithLogitsLoss`, AMP mixed precision, and saves the best checkpoint to `output/best_model.pth` based on validation accuracy.
5. **`testing.py`**: evaluates the trained model on the held-out test split, aggregating frame-level probabilities to **video-level** predictions (mean pooling), then reports accuracy, ROC-AUC, a confusion matrix at threshold 0.5, and a scan for the best accuracy threshold.
6. **`app.py`**: a Flask app for interactive inference. Uploads a video, runs YOLOv8n-face and the trained EfficientNet-B0 on every 5th frame (`FRAME_SKIP`), overlays a REAL/FAKE label and confidence per sampled frame, re-encodes the output to browser-compatible H.264 via `ffmpeg`, and renders a verdict plus fake/real frame percentages in `templates/index.html`.
7. **`face_detector.py`**: shared `YOLOFaceDetector` wrapper used by both `extract_frames.py` and `app.py` — runs YOLOv8n-face, picks the largest detected face per frame, crops with margin, and resizes.
8. **`app/`**: a FastAPI REST service wrapping the same model + detector for programmatic/cloud use (see below).

## Requirements

- Python 3.10–3.12 (PyTorch has no stable wheels for 3.13/3.14 yet)
- **`ffmpeg`** available on your `PATH` (required by `app.py` to re-encode annotated output for browser playback)
- An NVIDIA GPU with CUDA is strongly recommended for training/extraction, though everything falls back to CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)
- A [Kaggle account](https://www.kaggle.com/) configured for `kagglehub` if you want to download the dataset yourself (`data_create.py`)
- YOLOv8n-face weights (`yolov8n-face-lindevs.pt`) from [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face), placed at the project root

## Setup

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

## Building the model yourself

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

## Running the web app (Flask demo)

```bash
python app.py
```

Then open **http://127.0.0.1:5000**, upload a video (`.mp4`, `.avi`, `.mov`, `.mkv`), and click **Run Analysis**. The app will:

- Sample every 5th frame (`FRAME_SKIP = 5`)
- Run YOLOv8n-face detection + EfficientNet-B0 classification (`THRESHOLD = 0.74` sigmoid probability for "FAKE", tuned via `testing.py`)
- Draw a REAL/FAKE label + confidence on each processed frame
- Re-encode the output with `ffmpeg` for browser playback
- Show the verdict, fake/real frame percentages, and the annotated video inline

Uploaded videos are saved to `uploads/`, processed/annotated output to `processed/` (both created automatically).

## Running the REST API (FastAPI)

A cloud-deployable REST API lives under `app/`, wrapping the same model and detector.

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

### Docker

```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

Or with Compose:

```bash
docker compose up --build
```

### Tests

```bash
pytest tests/
```

### AWS deployment

`deploy.sh` automates build → push → deploy → smoke-test against an EC2 instance running Docker. It expects `REGISTRY`, `IMAGE_NAME`, `EC2_HOST`, and `EC2_KEY` environment variables — see the script header for details.

```bash
./deploy.sh
```

## Project structure

```
app.py                        Flask web app for interactive inference
app/                           FastAPI REST service (cloud-deployable)
  main.py                        Endpoints: /health, /predict, /metrics
  model.py                        Model loading + inference
  preprocess.py                   Frame extraction + face crop generator
  requirements.txt                Lean inference-only dependencies
face_detector.py               Shared YOLOv8n-face detector wrapper
data_create.py                 Downloads Celeb-DF v2 via kagglehub
extract_frames.py              YOLOv8-face crop extraction from raw videos
prepare_celebdf_list.py        Builds train/val/test split lists
train_celebdf_xception.py      Trains the EfficientNet-B0 classifier
testing.py                     Video-level evaluation (accuracy, AUC, confusion matrix)
templates/index.html           Web UI (upload form + results)
output/best_model.pth          Pretrained model checkpoint
videos/                        Sample test videos
tests/test_api.py              FastAPI endpoint tests
Dockerfile                     Container image for the FastAPI service
docker-compose.yml             Local Docker Compose setup
deploy.sh                      Build/push/deploy/smoke-test automation for AWS EC2
requirements.txt               Python dependencies (full dev/training set)
```
