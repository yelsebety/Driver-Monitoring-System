import cv2
from ultralytics import YOLO


class ObjectDetector:
    """Detects distraction objects (phone, cup, food) using YOLOv8."""

    def __init__(self, model_path="yolov8n.pt", confidence=0.45, target_classes=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_classes = set(
            target_classes
            if target_classes is not None
            else ["cell phone", "cup", "bottle"]
        )
        # Map COCO class IDs -> readable names
        self.names = self.model.names

    def detect(self, frame):
        """Returns a list of dicts: [{'label': str, 'conf': float, 'box': (x1, y1, x2, y2)}, ...]"""
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.names[cls_id]
            if label not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "label": label,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

        return detections


if __name__ == "__main__":
    from backend.core.camera import Camera, load_config

    cfg = load_config()
    cam_cfg, obj_cfg = cfg["camera"], cfg["object_detector"]

    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    detector = ObjectDetector(
        model_path=obj_cfg["model"],
        confidence=obj_cfg["confidence"],
        target_classes=obj_cfg["target_classes"],
    )

    print("Hold a phone, cup, or bottle in front of the camera. Press 'q' to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            text = f"{det['label']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        status = f"Objects: {len(detections)}"
        color = (0, 0, 255) if detections else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Object detector test", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()