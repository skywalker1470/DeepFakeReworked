import cv2

from face_detector import YOLOFaceDetector

FRAME_SKIP = 5


def iter_face_crops(video_path, detector: YOLOFaceDetector, frame_skip=FRAME_SKIP):
    """Yield RGB uint8 face crops sampled every `frame_skip` frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detector.detect(rgb)
            if face is not None:
                yield face

        frame_idx += 1

    cap.release()
