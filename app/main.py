import time
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.model import load_model, predict_face, THRESHOLD, DEVICE
from app.preprocess import iter_face_crops
from face_detector import YOLOFaceDetector

app = FastAPI(title="Deepfake Detector API")

model = load_model(device=DEVICE)
detector = YOLOFaceDetector(image_size=224, margin=20, conf_threshold=0.5, device=DEVICE)

metrics = {
    "request_count": 0,
    "total_latency_ms": 0.0,
}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/metrics")
def get_metrics():
    count = metrics["request_count"]
    avg_latency = metrics["total_latency_ms"] / count if count else 0.0
    return {
        "request_count": count,
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    try:
        probs = [
            predict_face(model, face, device=DEVICE)
            for face in iter_face_crops(video_path, detector)
        ]
    finally:
        os.remove(video_path)

    if not probs:
        raise HTTPException(status_code=422, detail="No face detected in video")

    avg_prob = sum(probs) / len(probs)
    prediction = "fake" if avg_prob > THRESHOLD else "real"

    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics["request_count"] += 1
    metrics["total_latency_ms"] += elapsed_ms

    return {
        "prediction": prediction,
        "confidence": round(avg_prob, 4),
        "frames_analyzed": len(probs),
        "inference_time_ms": round(elapsed_ms, 2),
    }
