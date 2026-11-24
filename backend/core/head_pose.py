import cv2
import numpy as np

# Landmark numbers: nose tip, chin, right-eye outer corner, left-eye outer corner,
# right mouth corner, left mouth corner.
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# A rough 3D model of a face (in millimetres) for those same 6 points.
# Axes: x right, y DOWN, z away from the camera. The nose tip is the origin.
MODEL_POINTS = np.array([
    [0, 0, 0],          # nose tip
    [0, 330, 65],       # chin
    [-225, -170, 135],  # right eye outer corner (left side of the image)
    [225, -170, 135],   # left eye outer corner (right side of the image)
    [-150, 150, 125],   # right mouth corner
    [150, 150, 125],    # left mouth corner
], dtype=np.float64)


class HeadPoseEstimator:
    """Estimates where the head is pointing: pitch (up/down), yaw (left/right)."""

    def __init__(self, yaw_threshold=25.0, pitch_down_threshold=20.0):
        self.yaw_threshold = yaw_threshold
        self.pitch_down_threshold = pitch_down_threshold
        self._offset_pitch = 0.0   # set by calibrate(): "this is looking forward"
        self._offset_yaw = 0.0
        self._raw = None

    def estimate(self, points, frame_w, frame_h):
        """Returns (pitch, yaw, roll) in degrees, or None if it fails.
        Positive pitch = looking down. Yaw is positive/negative for turning
        to either side. (0, 0) = looking straight at the camera."""
        image_points = points[LANDMARK_IDS].astype(np.float64)
        focal = float(frame_w)
        camera_matrix = np.array([[focal, 0, frame_w / 2],
                                  [0, focal, frame_h / 2],
                                  [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix,
                                   np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = cv2.RQDecomp3x3(rotation)[0]
        self._raw = (pitch, yaw)
        return pitch - self._offset_pitch, yaw - self._offset_yaw, roll

    def calibrate(self):
        """Call while the driver looks straight ahead: makes this pose = 0, 0."""
        if self._raw is not None:
            self._offset_pitch, self._offset_yaw = self._raw

    def classify(self, pitch, yaw):
        """Returns 'Side', 'Down' or 'Forward'."""
        if abs(yaw) > self.yaw_threshold:
            return "Side"
        if pitch > self.pitch_down_threshold:
            return "Down"
        return "Forward"


if __name__ == "__main__":
    # Test: show pitch, yaw and the head-pose label on the webcam video.
    from backend.core.camera import Camera, load_config
    from backend.core.face_mesh import FaceMesh

    cfg = load_config()
    cam_cfg, hp_cfg = cfg["camera"], cfg["head_pose"]
    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    face_mesh = FaceMesh()
    estimator = HeadPoseEstimator(hp_cfg["yaw_threshold"],
                                  hp_cfg["pitch_down_threshold"])

    print("Look straight ahead and press 'c' to calibrate. Press 'q' to quit.")
    while True:
        frame = cam.read()
        if frame is None:
            break
        h, w = frame.shape[:2]

        points = face_mesh.process(frame)
        if points is None:
            cv2.putText(frame, "No face found (Away)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            result = estimator.estimate(points, w, h)
            if result is not None:
                pitch, yaw, roll = result
                label = estimator.classify(pitch, yaw)
                color = (0, 255, 0) if label == "Forward" else (0, 0, 255)
                cv2.putText(frame, f"Pitch: {pitch:.0f}  Yaw: {yaw:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Head: {label}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Head pose test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            estimator.calibrate()
            print("Calibrated.")

    face_mesh.close()
    cam.release()
    cv2.destroyAllWindows()