from pathlib import Path

import cv2
import yaml


class Camera:
    """Opens the webcam and gives us one frame at a time."""

    def __init__(self, index=0, width=640, height=480):
        # CAP_DSHOW makes the webcam open faster and more reliably on Windows
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_open(self):
        return self.cap.isOpened()

    def read(self):
        """Returns one frame (an image), or None if reading failed."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


def load_config():
    # config.yaml is two folders above this file (backend/core -> project root)
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # This part only runs when you start this file directly: it's a test.
    cfg = load_config()["camera"]
    cam = Camera(cfg["index"], cfg["width"], cfg["height"])

    if not cam.is_open():
        print("ERROR: could not open the camera.")
        raise SystemExit(1)

    print("Camera opened. Press 'q' in the video window to quit.")
    while True:
        frame = cam.read()
        if frame is None:
            print("ERROR: could not read a frame.")
            break
        cv2.imshow("Camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()