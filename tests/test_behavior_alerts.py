import tempfile
import unittest
from pathlib import Path

from backend.core.alert_manager import AlertManager
from backend.core.behavior_classifier import BehaviorClassifier


class BehaviorClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.classifier = BehaviorClassifier(
            log_csv=str(Path(self.temp_dir.name) / "events.csv")
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detects_drowsiness_from_ear_values(self):
        self.assertEqual(self.classifier.update(0.15, "Forward", now=0)[0], "Normal")
        state, reason = self.classifier.update(0.15, "Forward", now=1.5)

        self.assertEqual(state, "Drowsy")
        self.assertIn("Eyes closed", reason)

    def test_detects_object_distraction_from_object_detector_output(self):
        detections = [
            {"label": "cell phone", "conf": 0.9, "box": (10, 20, 40, 80)}
        ]
        self.assertEqual(
            self.classifier.update(0.3, "Forward", detections, now=0)[0],
            "Normal",
        )
        state, reason = self.classifier.update(0.3, "Forward", detections, now=1.5)

        self.assertEqual(state, "Distracted")
        self.assertIn("cell phone", reason)

    def test_missing_face_starts_away_timer(self):
        self.assertEqual(self.classifier.update(None, None, now=0)[0], "Normal")
        self.assertEqual(self.classifier.update(None, None, now=2)[0], "Distracted")

    def test_drowsiness_has_priority_over_other_conditions(self):
        detections = [{"label": "cup", "conf": 0.8, "box": (1, 2, 3, 4)}]
        self.classifier.update(0.1, "Forward", detections, now=0)
        state, _ = self.classifier.update(0.1, "Down", detections, now=2)

        self.assertEqual(state, "Drowsy")

    def test_config_uses_behavior_classifier_section(self):
        classifier = BehaviorClassifier.from_config(
            {
                "behavior_classifier": {
                    "ear_threshold": 0.2,
                    "drowsy_seconds": 1.0,
                    "away_seconds": 3.0,
                    "object_use_seconds": 2.0,
                    "log_csv": str(Path(self.temp_dir.name) / "configured.csv"),
                }
            }
        )

        self.assertEqual(classifier.ear_threshold, 0.2)
        self.assertEqual(classifier.away_seconds, 3.0)
        self.assertTrue(Path(classifier.log_csv).is_file())


class AlertManagerTests(unittest.TestCase):
    def test_alert_manager_accepts_classifier_tuple_and_enforces_cooldown(self):
        manager = AlertManager({"enable_visual": False})
        played = []
        manager._play_sound = lambda behavior: played.append(behavior) or True
        state = ("Drowsy", "Eyes closed (1.5s)")

        self.assertTrue(manager.update(state, now=0)["alert_on"])
        self.assertFalse(manager.update(state, now=1)["sound_played"])
        self.assertTrue(manager.update(state, now=5)["sound_played"])
        self.assertEqual(played, ["Drowsy", "Drowsy"])


if __name__ == "__main__":
    unittest.main()
