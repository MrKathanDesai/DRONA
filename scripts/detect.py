#!/usr/bin/env python3
"""Run the audio detector on any audio file.

    python scripts/detect.py path/to/file.wav
    python scripts/detect.py data/raw_svanstrom/Audio/DRONE_001.wav --hop 0.5
    python scripts/detect.py file.wav --head tier3      # old indoor-only head

Slides a 1 s window over the file and prints a timeline. By default it uses
the tier-4 detector (indoor + outdoor training) with uncertainty flags from
scripts/eval_uncertainty.py, so each window is "drone", "no drone",
"unknown drone" or "low confidence". The encoder's top AudioSet tags are
shown as a sanity check of what it hears. A drone is reported only when
`--min-hits` consecutive windows exceed the threshold, which suppresses
one-off false alarms (the rule evaluated in the reports).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet import live as L
from dronedet.live import longest_run, windows  # noqa: F401  (used by tests)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--hop", type=float, default=0.5, help="window hop, s")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="consecutive positive windows needed to alarm")
    ap.add_argument("--head", choices=("detector", "tier4", "tier3"),
                    default="detector",
                    help="detector = tier 4 with uncertainty flags (default)")
    args = ap.parse_args()

    try:
        det = L.LiveDetector(args.head)
    except FileNotFoundError as e:
        print(e)
        return 1
    y = L.load(args.audio)
    df, _ = det.analyse(y, args.hop)

    print(f"\n  {args.audio}  ({len(y) / L.SR:.1f} s, {len(df)} windows, "
          f"{det.name}, threshold {det.threshold:.3f})\n")
    for r in df.itertuples():
        bar = "#" * int(round(r.p * 20))
        note = f" ({r.reason})" if r.reason else ""
        print(f"  {r.t:6.1f}s  {r.p:.3f} {bar:20s} {r.state:15s}{note:30s} {r.tags}")

    run = longest_run(df.hit)
    verdict = "DRONE DETECTED" if run >= args.min_hits else "no drone"
    flagged = int((df.state == "low confidence").sum())
    print(f"\n  {verdict}: {int(df.hit.sum())}/{len(df)} windows above threshold, "
          f"longest run {run} (need {args.min_hits}); max p {df.p.max():.3f}; "
          f"{flagged} windows flagged low confidence\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
