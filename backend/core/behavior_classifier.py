import time
from backend.utils.hold_timer import HoldTimer

# Object names (as reported by YOLO) grouped into the behaviors from the project PDF
PHONE_LABELS = {"cell phone", "phone", "mobile phone"}
DRINK_LABELS = {"cup", "bottle", "wine glass"}
FOOD_LABELS = {"sandwich", "banana", "apple", "orange", "broccoli", "carrot",
               "hot dog", "pizza", "donut", "cake", }


class BehaviorClassifier:
    """Combines EAR tracking, head pose and YOLO detections over time.

    update() returns (state, reason) where state is Normal / Drowsy / Distracted
    (used for alerts and colours). After every update() two more values are
    available for logging:
        self.behavior   one of: Normal, Drowsy, Distracted, Using Phone, Eating, Drinking
        self.confidence a number from 0 to 1 (see _strength / _object_behavior)
    """

    def __init__(
        self,
        grace_seconds=0.5,
        ear_threshold=0.21,
        drowsy_seconds=1.5,
        away_seconds=2.0,
        object_use_seconds=1.5,
        log_csv=None,   # no longer used: logging is done by backend/storage/csv_logger.py
    ):
        self.ear_threshold = ear_threshold
        self.drowsy_seconds = drowsy_seconds
        self.away_seconds = away_seconds
        self.object_use_seconds = object_use_seconds

        self._away_timer = HoldTimer(grace_seconds)
        self._object_timer = HoldTimer(grace_seconds)
        self._last_detections = []

        

        self.current_state = "Normal"
        self.current_reason = "Attentive"
        self.behavior = "Normal"
        self.confidence = 1.0

    @staticmethod
    def _strength(duration, threshold):
        """How strongly a time rule is met: 0.5 right when the time limit is
        reached, growing to 1.0 when the behavior lasts twice the limit."""
        extra = min(1.0, max(0.0, (duration - threshold) / threshold))
        return 0.5 + 0.5 * extra

    @staticmethod
    def _object_behavior(detections):
        """Picks Using Phone / Drinking / Eating from the detected objects.
        The confidence is the YOLO confidence of the matching object."""
        def best(names):
            confs = [d["conf"] for d in detections if d["label"].lower() in names]
            return max(confs) if confs else None

        phone, drink, food = best(PHONE_LABELS), best(DRINK_LABELS), best(FOOD_LABELS)
        if phone is not None:
            return "Using Phone", float(phone)
        if drink is not None:
            return "Drinking", float(drink)
        if food is not None:
            return "Eating", float(food)
        return "Distracted", float(max(d["conf"] for d in detections))

    def update(self, ear_state, head_label, detections):
        """Takes output from EarTracker, HeadPoseEstimator, and ObjectDetector.
        Priority: Drowsy > objects (phone/drink/food) > head turned away > Normal."""
        now = time.monotonic()
        obj_labels = [d["label"] for d in detections]

        # Head turned away, looking down, or face completely missing
        away_dur = self._away_timer.update(head_label in ("Side", "Down", None), now)

        # Distraction objects (phone, cup, bottle, food items)
        if obj_labels:
            self._last_detections = detections
        obj_dur = self._object_timer.update(bool(obj_labels), now)

        closed_dur = ear_state["closed_duration"]

        new_state, reason = "Normal", "Attentive"
        behavior = "Normal"
        # Normal confidence = how far we are from triggering any rule
        progress = max(closed_dur / self.drowsy_seconds,
                       away_dur / self.away_seconds,
                       obj_dur / self.object_use_seconds)
        confidence = min(1.0, max(0.0, 1.0 - progress))

        if ear_state["drowsy"]:
            new_state, behavior = "Drowsy", "Drowsy"
            reason = f"Eyes closed ({closed_dur:.1f}s)"
            confidence = self._strength(closed_dur, self.drowsy_seconds)
        elif obj_dur >= self.object_use_seconds:
            new_state = "Distracted"
            # During a short detection gap 'detections' is empty, so use the last ones
            held = detections or self._last_detections
            behavior, confidence = self._object_behavior(held)
            unique_objs = ", ".join(sorted(set(d["label"] for d in held)))
            reason = f"Using/Holding {unique_objs} ({obj_dur:.1f}s)"
        elif away_dur >= self.away_seconds:
            new_state, behavior = "Distracted", "Distracted"
            pose_desc = "Face missing" if head_label is None else f"Head {head_label}"
            reason = f"{pose_desc} ({away_dur:.1f}s)"
            confidence = self._strength(away_dur, self.away_seconds)

        self.current_state = new_state
        self.current_reason = reason
        self.behavior = behavior
        self.confidence = confidence
        return self.current_state, self.current_reason