import time
from collections import deque

import numpy as np

from backend.core.face_mesh import LEFT_EYE, RIGHT_EYE


def eye_aspect_ratio(points, indices):
    """EAR for one eye. 'points' is the array of all face points,
    'indices' are the 6 landmark numbers of that eye, in this order:
    [outer corner, top1, top2, inner corner, bottom2, bottom1]."""
    p = points[indices]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear(points):
    """Average EAR of both eyes."""
    return (eye_aspect_ratio(points, RIGHT_EYE)
            + eye_aspect_ratio(points, LEFT_EYE)) / 2.0


class EarTracker:
    """Turns a stream of EAR values into: eye closed?, how long closed,
    blink count, and 'drowsy' (eyes closed too long)."""

    def __init__(self, threshold=0.21, drowsy_seconds=1.5, window_seconds=60):
        self.threshold = threshold
        self.drowsy_seconds = drowsy_seconds
        self.window_seconds = window_seconds
        self.closed_since = None
        self.blink_count = 0
        self.blink_times = deque()

    def update(self, ear, now=None):
        now = time.monotonic() if now is None else now
        closed = ear < self.threshold

        if closed:
            if self.closed_since is None:
                self.closed_since = now
            duration = now - self.closed_since
        else:
            duration = 0.0
            if self.closed_since is not None:
                # Eye just reopened: a short closure counts as a blink
                if now - self.closed_since < self.drowsy_seconds:
                    self.blink_count += 1
                    self.blink_times.append(now)
                self.closed_since = None

        # Forget blinks older than the window
        while self.blink_times and now - self.blink_times[0] > self.window_seconds:
            self.blink_times.popleft()

        return {
            "ear": float(ear),
            "eye_closed": closed,
            "closed_duration": duration,
            "drowsy": duration >= self.drowsy_seconds,
            "blink_count": self.blink_count,
            "blinks_last_minute": len(self.blink_times),
        }


if __name__ == "__main__":
    # Test: show live EAR, blink count and drowsiness on the webcam video.
    import cv2

    from backend.core.camera import Camera, load_config
    from backend.core.face_mesh import FaceMesh

    cfg = load_config()
    cam_cfg, ear_cfg = cfg["camera"], cfg["ear"]
    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    face_mesh = FaceMesh()
    tracker = EarTracker(ear_cfg["threshold"], ear_cfg["drowsy_seconds"])

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
            state = tracker.update(average_ear(points))
            color = (0, 0, 255) if state["eye_closed"] else (0, 255, 0)
            cv2.putText(frame, f"EAR: {state['ear']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Blinks: {state['blink_count']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Closed for: {state['closed_duration']:.1f}s",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if state["drowsy"]:
                cv2.putText(frame, "DROWSY!", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("EAR test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cam.release()
    cv2.destroyAllWindows()