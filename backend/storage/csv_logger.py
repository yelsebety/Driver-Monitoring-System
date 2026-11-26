import csv
import re

from datetime import datetime
from pathlib import Path


COLUMNS = [
    "Session_ID",
    "Timestamp",
    "Behavior",
    "EAR",
    "Head_Pose",
    "Confidence",
    "Alert"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def next_session_id(sessions_dir):
    """
    Looks at S01_events.csv, S02_events.csv, ...
    and returns the next available session ID.
    """

    numbers = []

    for f in Path(sessions_dir).glob("S*_events.csv"):

        m = re.match(r"S(\d+)_events\.csv$", f.name)

        if m:
            numbers.append(int(m.group(1)))

    return f"S{(max(numbers) + 1 if numbers else 1):02d}"


class CsvLogger:
    """
    Writes one row per detection to:

        data/sessions/Sxx_events.csv

    A row is written immediately when the behavior changes,
    and otherwise at most once every interval_seconds.
    """

    def __init__(
        self,
        sessions_dir="data/sessions",
        interval_seconds=1.0
    ):

        self.dir = PROJECT_ROOT / sessions_dir
        self.dir.mkdir(parents=True, exist_ok=True)

        self.session_id = next_session_id(self.dir)

        self.path = self.dir / f"{self.session_id}_events.csv"

        self.interval = interval_seconds

        self._last_behavior = None
        self._last_write = 0.0

        self._closed = False

        self._file = open(
            self.path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self._writer = csv.writer(self._file)

        self._writer.writerow(COLUMNS)

        self._file.flush()

    # -------------------------------------------------
    # LOG A DETECTION
    # -------------------------------------------------

    def log(
        self,
        behavior,
        ear,
        head_pose,
        confidence,
        alert
    ):
        """
        Saves a detection to the session CSV.

        Returns True if a row was written.
        """

        if self._closed:
            raise RuntimeError("Cannot log to a closed CSV file.")

        import time

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

        # Flush immediately so data is not lost if the program stops.
        self._file.flush()

        self._last_behavior = behavior
        self._last_write = now

        return True

    # -------------------------------------------------
    # REMOVE TERMINAL NO-FACE ROWS
    # -------------------------------------------------

    def trim_terminal_no_face(self):
        """
        Removes consecutive no-face rows at the END of the session.

        A no-face row is identified by:

            EAR = empty
            Head_Pose = Away

        Only terminal rows are removed.

        No-face periods in the middle of the session
        are preserved.

        Returns the number of removed rows.
        """

        if self._closed:
            raise RuntimeError(
                "Cannot trim a CSV file that is already closed."
            )

        # Make sure all buffered data is written.
        self._file.flush()

        # Close the file before rewriting it.
        self._file.close()

        self._closed = True

        temp_path = self.path.with_name(
            self.path.name + ".tmp"
        )

        removed_rows = 0

        try:

            # Read the existing session CSV.
            with open(
                self.path,
                "r",
                newline="",
                encoding="utf-8"
            ) as f:

                rows = list(csv.reader(f))

            if len(rows) <= 1:
                return 0

            header = rows[0]
            data = rows[1:]

            ear_index = header.index("EAR")
            pose_index = header.index("Head_Pose")

            # Walk backwards from the final row.
            while data:

                last_row = data[-1]

                ear_missing = (
                    len(last_row) > ear_index
                    and last_row[ear_index].strip() == ""
                )

                head_missing = (
                    len(last_row) > pose_index
                    and last_row[pose_index].strip().lower() == "away"
                )

                # Stop when we reach a row where the face was detected.
                if not (ear_missing and head_missing):
                    break

                data.pop()

                removed_rows += 1

            # Write the cleaned CSV to a temporary file.
            with open(
                temp_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                writer.writerow(header)

                writer.writerows(data)

            # Replace the original CSV with the cleaned version.
            temp_path.replace(self.path)

        finally:

            # Remove temporary file if something went wrong.
            if temp_path.exists():
                temp_path.unlink()

            # Reopen the CSV in append mode.
            # This keeps close() safe when main.py calls it afterward.
            self._file = open(
                self.path,
                "a",
                newline="",
                encoding="utf-8"
            )

            self._writer = csv.writer(self._file)

            self._closed = False

        return removed_rows

    # -------------------------------------------------
    # CLOSE LOGGER
    # -------------------------------------------------

    def close(self):

        if not self._closed:

            self._file.flush()
            self._file.close()

            self._closed = True


# -------------------------------------------------
# QUICK TEST
# -------------------------------------------------

if __name__ == "__main__":

    logger = CsvLogger()

    logger.log(
        "Normal",
        0.31,
        "Forward",
        0.96,
        False
    )

    logger.log(
        "Drowsy",
        0.15,
        "Forward",
        0.92,
        True
    )

    logger.log(
        "Using Phone",
        0.29,
        "Down",
        0.88,
        True
    )

    logger.close()

    print("Wrote", logger.path)