import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# models/face_landmarker.task, two folders above this file
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

# Landmark numbers around each eye (6 points each), used later for EAR.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


class FaceMesh:
    """Finds the face in a camera frame and returns its landmark points."""

    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,  # only the driver
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.monotonic()
        self._last_ts = -1

    def process(self, frame):
        """Takes a camera frame (BGR). Returns an array of (x, y) pixel
        points for the face, or None if no face is found."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))

        # Video mode needs a timestamp that always increases
        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self.landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])

    def close(self):
        self.landmarker.close()


if __name__ == "__main__":
    # Test: show the webcam with the landmarks drawn on it.
    from backend.core.camera import Camera, load_config

    cfg = load_config()["camera"]
    cam = Camera(cfg["index"], cfg["width"], cfg["height"])
    face_mesh = FaceMesh()

    print("Press 'q' in the video window to quit.")
    while True:
        frame = cam.read()
        if frame is None:
            break

        points = face_mesh.process(frame)
        if points is None:
            cv2.putText(frame, "No face found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for x, y in points:                      # all face points: green
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            for i in RIGHT_EYE + LEFT_EYE:           # eye points: red, bigger
                x, y = points[i]
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        cv2.imshow("Face mesh test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cam.release()
    cv2.destroyAllWindows()