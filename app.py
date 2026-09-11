import os
import uuid
import cv2
import torch
import torch.nn as nn
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, abort
from torchvision import transforms, models
from face_detector import YOLOFaceDetector
from ratelimit import is_rate_limited
import subprocess
# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "output/best_model.pth"   # <-- your trained model
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
FRAME_SKIP = 5
THRESHOLD = 0.74
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# --------------------------------------------------
# LOAD EFFICIENTNET-B0 (MATCHES TRAINING)
# --------------------------------------------------
model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded on:", DEVICE)

# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------
detector = YOLOFaceDetector(
    image_size=224,
    margin=20,
    conf_threshold=0.5,
    device=DEVICE
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------
def process_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temporary raw output (OpenCV)
    temp_output = output_path.replace(".mp4", "_temp.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    fake_count = 0
    real_count = 0
    processed_frames = 0
    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP != 0:
                writer.write(frame)
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face = detector.detect(rgb)

            if face is not None:
                face = transform(face).unsqueeze(0).to(DEVICE)

                output = model(face)
                prob = torch.sigmoid(output).item()

                label = "FAKE" if prob > THRESHOLD else "REAL"

                if label == "FAKE":
                    fake_count += 1
                    color = (0, 0, 255)
                else:
                    real_count += 1
                    color = (0, 255, 0)

                processed_frames += 1

                cv2.putText(frame,
                            f"{label} {prob:.2f}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2)

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()

    # 🔥 RE-ENCODE TO H.264 (Browser Compatible)
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", temp_output,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(temp_output)

    if processed_frames == 0:
        return None

    fake_pct = round((fake_count / processed_frames) * 100, 2)
    real_pct = round((real_count / processed_frames) * 100, 2)

    verdict = "FAKE" if fake_pct > real_pct else "REAL"

    return {
        "verdict": verdict,
        "fake_pct": fake_pct,
        "real_pct": real_pct,
        "total_frames": processed_frames
    }
# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        if is_rate_limited(client_ip.split(",")[0].strip()):
            return render_template("index.html", error="Rate limit exceeded — please wait a minute and try again"), 429

        if "video" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["video"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        safe_name = secure_filename(file.filename)
        ext = os.path.splitext(safe_name)[1] or ".mp4"
        unique_id = uuid.uuid4().hex
        upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}{ext}")
        file.save(upload_path)

        output_name = f"processed_{unique_id}.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        result = process_video(upload_path, output_path)

        if result is None:
            return render_template("index.html",
                                   error="No face detected in video")

        return render_template("index.html",
                               result=result,
                               filename=output_name)

    return render_template("index.html")

@app.route("/processed/<filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)