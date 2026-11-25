"""Guided ground-truth recorder (does NOT use the camera).

Run it in a SECOND terminal while the monitoring system is recording a session.
It tells you what to do and when ("NOW: close your eyes"), remembers the real
clock time of every action, and writes data/ground_truth.csv for the evaluation.

Usage (from the project root, venv active):

    1. Start the monitoring system so it begins logging session S05.
    2. In another terminal:   python scripts/record_ground_truth.py --session S05
    3. Follow the instructions on screen.
    4. When it finishes, stop the monitoring system so S05_events.csv is complete.
       If the events file was not ready, convert afterwards with:
           python scripts/record_ground_truth.py --session S05 --convert-only

How the numbers are produced
    The evaluation measures seconds from the FIRST row of data/sessions/S05_events.csv.
    This script stores real clock times (data/ground_truth_raw.csv) and converts them
    to those seconds using that first row's Timestamp. Both use this computer's clock,
    so there is no manual stopwatch work.

Options:
    --repeat N     repeat the whole plan N times (default 2 -> 2 examples per behavior)
    --scale F      shrink all durations (e.g. 0.2 for a quick test run)
    --events-dir   where the session CSVs are (default data/sessions)
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

BEHAVIORS = ["Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking"]

# (behavior, seconds, instruction shown on screen)
PLAN = [
    ("Normal",      10, "Drive normally, look at the road"),
    ("Drowsy",      15, "Close your eyes / let them droop for the whole time"),
    ("Normal",      10, "Back to normal"),
    ("Using Phone", 15, "Hold your phone to your ear or in front of your face"),
    ("Normal",      10, "Back to normal"),
    ("Drinking",    15, "Drink from a bottle or cup (hold it at your mouth)"),
    ("Normal",      10, "Back to normal"),
    ("Eating",      15, "Eat something (or hold food at your mouth)"),
    ("Normal",      10, "Back to normal"),
    ("Distracted",  15, "Turn your head away / look down or to the side"),
    ("Normal",      10, "Back to normal"),
]

RAW_COLS = ["Session_ID", "Start_Time", "End_Time", "True_Behavior"]
GT_COLS = ["Session_ID", "Start_s", "End_s", "True_Behavior"]
TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"


def parse_events_time(text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot read Timestamp: {text!r}")


def first_event_time(events_path: str) -> datetime:
    with open(events_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            return parse_events_time(row["Timestamp"])
    raise ValueError(f"{events_path} has no rows yet")


def read_rows(path: str) -> list:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, cols: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def record(session: str, repeat: int, scale: float, raw_path: str) -> None:
    steps = [(b, max(1.0, s * scale), msg) for _ in range(repeat) for b, s, msg in PLAN]
    total = sum(s for _, s, _ in steps)
    print(f"\nSession {session}: {len(steps)} steps, about {total / 60:.1f} minutes.")
    print("Make sure the monitoring system is already recording this session.")
    input("Press Enter when you are in position and ready to start... ")

    new_rows = []
    for n in range(3, 0, -1):
        print(f"Starting in {n}...")
        time.sleep(1)

    for i, (behavior, seconds, msg) in enumerate(steps, 1):
        start = datetime.now()
        print(f"\a\n[{i}/{len(steps)}] NOW: {behavior.upper()} for {seconds:.0f} s -> {msg}")
        time.sleep(seconds)
        end = datetime.now()
        if behavior != "Normal":   # anything not listed counts as Normal
            new_rows.append({"Session_ID": session,
                             "Start_Time": start.strftime(TIME_FMT),
                             "End_Time": end.strftime(TIME_FMT),
                             "True_Behavior": behavior})
        # save after every step, so an interruption loses nothing
        kept = [r for r in read_rows(raw_path) if r["Session_ID"] != session]
        write_rows(raw_path, RAW_COLS, kept + new_rows)

    print("\nDone. Now stop the monitoring system so the session file is complete.")


def convert(session: str, raw_path: str, events_dir: str, gt_path: str) -> None:
    raw = [r for r in read_rows(raw_path) if r["Session_ID"] == session]
    if not raw:
        sys.exit(f"No recorded actions for {session} in {raw_path}. Run the recording first.")

    events_path = os.path.join(events_dir, f"{session}_events.csv")
    if not os.path.isfile(events_path):
        sys.exit(f"{events_path} not found. Check --events-dir and the session name, "
                 f"then run again with --convert-only.")
    t0 = first_event_time(events_path)

    out = []
    for r in raw:
        s = (datetime.strptime(r["Start_Time"], TIME_FMT) - t0).total_seconds()
        e = (datetime.strptime(r["End_Time"], TIME_FMT) - t0).total_seconds()
        out.append({"Session_ID": session, "Start_s": round(max(s, 0), 1),
                    "End_s": round(e, 1), "True_Behavior": r["True_Behavior"]})

    # keep other sessions' rows, replace this session's
    existing = [r for r in read_rows(gt_path) if r.get("Session_ID") != session]
    write_rows(gt_path, GT_COLS, existing + out)

    print(f"\nWrote {len(out)} rows for {session} to {gt_path} "
          f"(first event in {events_path}: {t0}).")
    for r in out:
        print(f"  {r['Session_ID']}  {r['Start_s']:>6} - {r['End_s']:<6} {r['True_Behavior']}")
    if out and out[0]["Start_s"] < 0:
        print("Warning: an action started before the first logged event.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--session", required=True, help="Session_ID, e.g. S05 (must match S05_events.csv)")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--events-dir", default="data/sessions")
    p.add_argument("--ground-truth", default="data/ground_truth.csv")
    p.add_argument("--raw", default="data/ground_truth_raw.csv")
    p.add_argument("--convert-only", action="store_true", help="skip recording, just convert")
    a = p.parse_args()

    if not a.convert_only:
        record(a.session, a.repeat, a.scale, a.raw)
    convert(a.session, a.raw, a.events_dir, a.ground_truth)


if __name__ == "__main__":
    main()
