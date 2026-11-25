import cv2
from backend.core.alert_manager import AlertManager
from backend.core.behavior_classifier import BehaviorClassifier
from backend.core.camera import Camera, load_config
from backend.core.ear_calculator import EarTracker, average_ear
from backend.core.face_mesh import FaceMesh
from backend.core.head_pose import HeadPoseEstimator
from backend.core.object_detector import ObjectDetector
from backend.storage.csv_logger import CsvLogger


def main():
    cfg = load_config()
    cam_cfg = cfg["camera"]
    hp_cfg = cfg["head_pose"]
    ear_cfg = cfg["ear"]
    obj_cfg = cfg["object_detector"]
    beh_cfg = cfg["behavior_classifier"]
    alert_cfg = cfg.get("alert_manager", {"sound_enabled": True, "cooldown_seconds": 3.0})

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
    alert_mgr = AlertManager(
        sound_enabled=alert_cfg["sound_enabled"],
        cooldown_seconds=alert_cfg["cooldown_seconds"],
    )

    # One new session file (S01, S02, ...) per run
    log_cfg = cfg.get("logging", {"sessions_dir": "data/sessions", "interval_seconds": 1.0})
    logger = CsvLogger(log_cfg["sessions_dir"], log_cfg["interval_seconds"])
    print(f"Logging session {logger.session_id} to {logger.path}")

    print("Driver Monitoring System Running...")
    print("1. Look straight ahead and press 'c' to calibrate head pose.")
    print("2. Press 'q' to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            break

        h, w = frame.shape[:2]
        points = face_mesh.process(frame)

        ear_state = {
            "ear": 0.0,
            "eye_closed": False,
            "closed_duration": 0.0,
            "drowsy": False,
            "blink_count": 0,
        }
        head_label = None

        if points is not None:
            ear_val = average_ear(points)
            ear_state = ear_tracker.update(ear_val)

            pose_res = pose_est.estimate(points, w, h)
            if pose_res is not None:
                pitch, yaw, _ = pose_res
                head_label = pose_est.classify(pitch, yaw)

        detections = obj_det.detect(frame)
        state, reason = classifier.update(ear_state, head_label, detections)

        # Trigger audio & visual alerts based on behavior state
        alert_mgr.trigger(state)

        # Save the result to this session's CSV file
        logger.log(
            behavior=classifier.behavior,
            ear=ear_state["ear"] if points is not None else None,
            head_pose=head_label or "Away",
            confidence=classifier.confidence,
            alert=(state != "Normal"),
        )

        # Draw detected objects
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

        # HUD Banner overlay
        colors = {"Normal": (0, 255, 0), "Distracted": (0, 165, 255), "Drowsy": (0, 0, 255)}
        banner_color = colors.get(state, (255, 255, 255))

        cv2.putText(frame, f"State: {classifier.behavior}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, banner_color, 2)
        cv2.putText(frame, f"Reason: {reason}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
        cv2.putText(
            frame,
            f"EAR: {ear_state['ear']:.2f} | Head: {head_label or 'Missing'}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Driver Monitoring System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            pose_est.calibrate()
            print("Head pose calibrated.")

    logger.close()
    face_mesh.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()