import csv
import re
import time
from datetime import datetime
from pathlib import Path

COLUMNS = ["Session_ID", "Timestamp", "Behavior", "EAR",
           "Head_Pose", "Confidence", "Alert"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def next_session_id(sessions_dir):
    """Looks at S01_events.csv, S02_events.csv ... and returns the next free ID."""
    numbers = []
    for f in Path(sessions_dir).glob("S*_events.csv"):
        m = re.match(r"S(\d+)_events\.csv$", f.name)
        if m:
            numbers.append(int(m.group(1)))
    return f"S{(max(numbers) + 1 if numbers else 1):02d}"


class CsvLogger:
    """Writes one row per detection to data/sessions/Sxx_events.csv.
    A row is written immediately when the behavior changes, and otherwise
    at most once every 'interval_seconds'."""

    def __init__(self, sessions_dir="data/sessions", interval_seconds=1.0):
        self.dir = PROJECT_ROOT / sessions_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = next_session_id(self.dir)
        self.path = self.dir / f"{self.session_id}_events.csv"
        self.interval = interval_seconds
        self._last_behavior = None
        self._last_write = 0.0
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self._file.flush()

    def log(self, behavior, ear, head_pose, confidence, alert):
        """alert: True/False (or 'ON'/'OFF'). Returns True if a row was written."""
        now = time.monotonic()
        changed = behavior != self._last_behavior
        if not changed and now - self._last_write < self.interval:
            return False
        if isinstance(alert, bool):
            alert = "ON" if alert else "OFF"
        self._writer.writerow([
            self.session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            behavior,
            "" if ear is None else f"{ear:.3f}",
            head_pose,
            f"{confidence:.3f}",
            alert,
        ])
        self._file.flush()   # so nothing is lost if the program is stopped
        self._last_behavior = behavior
        self._last_write = now
        return True

    def close(self):
        self._file.close()


if __name__ == "__main__":
    # Quick test: writes 3 fake rows, then tells you where the file is.
    logger = CsvLogger()
    logger.log("Normal", 0.31, "Forward", 0.96, False)
    logger.log("Drowsy", 0.15, "Forward", 0.92, True)
    logger.log("Using Phone", 0.29, "Down", 0.88, True)
    logger.close()
    print("Wrote", logger.path)