import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_PATH = "yolov8n-face-lindevs.pt"


class YOLOFaceDetector:
    def __init__(self, weights_path=WEIGHTS_PATH, image_size=224, margin=20,
                 conf_threshold=0.5, device=None):
        self.model = YOLO(weights_path)
        self.image_size = image_size
        self.margin = margin
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame_rgb):
        """Detect the largest face in an RGB uint8 numpy frame.

        Returns an RGB uint8 numpy crop resized to (image_size, image_size),
        or None if no face is found.
        """
        results = self.model.predict(
            frame_rgb,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best = xyxy[np.argmax(areas)]

        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = best
        x1 = max(0, int(x1 - self.margin))
        y1 = max(0, int(y1 - self.margin))
        x2 = min(w, int(x2 + self.margin))
        y2 = min(h, int(y2 + self.margin))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.image_size, self.image_size))
        return crop
