import csv
import os
import time
from datetime import datetime


class BehaviorClassifier:
    """Combines EAR tracking, head pose classification, and YOLO object detections 
    over time into Normal, Drowsy, or Distracted states with precise reasons.
    """

    def __init__(
        self,
        ear_threshold=0.21,
        drowsy_seconds=1.5,
        away_seconds=2.0,
        object_use_seconds=1.5,
        log_csv="data/events.csv",
    ):
        self.ear_threshold = ear_threshold
        self.drowsy_seconds = drowsy_seconds
        self.away_seconds = away_seconds
        self.object_use_seconds = object_use_seconds
        self.log_csv = log_csv

        # Timers / tracking state variables
        self._looking_away_since = None
        self._object_seen_since = None

        self.current_state = "Normal"
        self.current_reason = "Attentive"
        self._init_csv()

    def _init_csv(self):
        folder = os.path.dirname(self.log_csv)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "state", "reason", "ear", "head_pose", "objects"])

    def _log_event(self, state, reason, ear, head_pose, objects):
        with open(self.log_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                state,
                reason,
                f"{ear:.3f}" if ear is not None else "",
                head_pose or "Missing",
                ";".join(objects) if objects else "",
            ])

    def update(self, ear_state, head_label, detections):
        """
        Takes output from EarTracker, HeadPoseEstimator, and ObjectDetector.
        Evaluates safety rules and returns (state, reason).
        Priority: Drowsy > Distracted (Objects / Head / Missing) > Normal.
        """
        now = time.monotonic()
        obj_labels = [d["label"] for d in detections]

        # 1. Track head turned away, looking down, or face completely missing
        if head_label in ("Side", "Down", None):
            if self._looking_away_since is None:
                self._looking_away_since = now
            away_dur = now - self._looking_away_since
        else:
            self._looking_away_since = None
            away_dur = 0.0

        # 2. Track distraction objects (phone, cup, bottle, food items)
        if obj_labels:
            if self._object_seen_since is None:
                self._object_seen_since = now
            obj_dur = now - self._object_seen_since
        else:
            self._object_seen_since = None
            obj_dur = 0.0

        # 3. Evaluate state hierarchy
        new_state = "Normal"
        reason = "Attentive"

        # Drowsiness takes highest safety priority (driven by EarTracker)
        if ear_state["drowsy"]:
            new_state = "Drowsy"
            reason = f"Eyes closed ({ear_state['closed_duration']:.1f}s)"
        elif obj_dur >= self.object_use_seconds:
            new_state = "Distracted"
            unique_objs = ", ".join(sorted(set(obj_labels)))
            reason = f"Using/Holding {unique_objs} ({obj_dur:.1f}s)"
        elif away_dur >= self.away_seconds:
            new_state = "Distracted"
            pose_desc = "Face missing" if head_label is None else f"Head {head_label}"
            reason = f"{pose_desc} ({away_dur:.1f}s)"

        # Log state changes
        if new_state != self.current_state or reason != self.current_reason:
            # Only log state transitions or significant duration updates to keep CSV clean
            if new_state != self.current_state:
                self._log_event(
                    new_state,
                    reason,
                    ear_state["ear"],
                    head_label,
                    obj_labels,
                )
            self.current_state = new_state
            self.current_reason = reason

        return self.current_state, self.current_reason


if __name__ == "__main__":
    import cv2
    from backend.core.camera import Camera, load_config
    from backend.core.ear_calculator import EarTracker, average_ear
    from backend.core.face_mesh import FaceMesh
    from backend.core.head_pose import HeadPoseEstimator
    from backend.core.object_detector import ObjectDetector

    cfg = load_config()
    cam_cfg = cfg["camera"]
    hp_cfg = cfg["head_pose"]
    ear_cfg = cfg["ear"]
    obj_cfg = cfg["object_detector"]
    beh_cfg = cfg["behavior_classifier"]

    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    face_mesh = FaceMesh()
    pose_est = HeadPoseEstimator(hp_cfg["yaw_threshold"], hp_cfg["pitch_down_threshold"])
    ear_tracker = EarTracker(ear_cfg["threshold"], ear_cfg["drowsy_seconds"])
    obj_det = ObjectDetector(obj_cfg["model"], obj_cfg["confidence"], obj_cfg["target_classes"])
    classifier = BehaviorClassifier(
        ear_threshold=beh_cfg["ear_threshold"],
        drowsy_seconds=beh_cfg["drowsy_seconds"],
        away_seconds=beh_cfg["away_seconds"],
        object_use_seconds=beh_cfg["object_use_seconds"],
        log_csv=beh_cfg["log_csv"],
    )

    print("Look straight ahead and press 'c' to calibrate head pose. Press 'q' to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            break

        h, w = frame.shape[:2]
        points = face_mesh.process(frame)

        ear_state = {"ear": 0.0, "eye_closed": False, "closed_duration": 0.0, "drowsy": False, "blink_count": 0}
        head_label = None

        if points is not None:
            ear_val = average_ear(points)
            ear_state = ear_tracker.update(ear_val)

            pose_res = pose_est.estimate(points, w, h)
            if pose_res is not None:
                pitch, yaw, _ = pose_res
                head_label = pose_est.classify(pitch, yaw)

        # Detect distraction items (phones, cups, food)
        detections = obj_det.detect(frame)

        # Classify overall driver behavior
        state, reason = classifier.update(ear_state, head_label, detections)

        # Draw object boxes
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                frame,
                f"{det['label']} ({det['conf']:.2f})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        # Color-coded UI banner
        colors = {"Normal": (0, 255, 0), "Distracted": (0, 165, 255), "Drowsy": (0, 0, 255)}
        banner_color = colors.get(state, (255, 255, 255))

        cv2.putText(frame, f"State: {state}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, banner_color, 2)
        cv2.putText(frame, f"Reason: {reason}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
        cv2.putText(
            frame,
            f"EAR: {ear_state['ear']:.2f} | Blinks: {ear_state['blink_count']} | Head: {head_label or 'Missing'}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Driver Behavior Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            pose_est.calibrate()
            print("Head pose calibrated.")

    face_mesh.close()
    cam.release()
    cv2.destroyAllWindows()