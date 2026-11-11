#!/usr/bin/env python3
"""Run the pretrained detector on any audio file.

    python scripts/detect.py path/to/file.wav
    python scripts/detect.py data/raw_svanstrom/Audio/DRONE_001.wav --hop 0.5

Slides a 1 s window over the file, scores each window with the frozen
encoder + trained head (scripts/train_pretrained.py), and prints a timeline
with the encoder's top AudioSet tags as a sanity check of what it hears.
A drone is reported only when `--min-hits` consecutive windows fire, which
suppresses one-off false alarms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from dronedet.features import audio as A
from dronedet.models import pretrained as PT
from dronedet.paths import DATA_DIR

HEAD = DATA_DIR / "models" / "pretrained_head.joblib"


def windows(y: np.ndarray, sr: int, hop_s: float) -> tuple[list, list]:
    n, hop = sr, int(hop_s * sr)
    starts = list(range(0, max(len(y) - n, 0) + 1, hop))
    return [A.fix_length(y[s:s + n], n) for s in starts], [s / sr for s in starts]


def longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--hop", type=float, default=0.5, help="window hop, s")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="consecutive positive windows needed to alarm")
    args = ap.parse_args()

    if not HEAD.exists():
        print(f"{HEAD} missing; run scripts/train_pretrained.py first")
        return 1
    head = joblib.load(HEAD)
    enc, labels = PT.Encoder(), PT.audioset_labels()

    y = A.load_mono(args.audio, A.DEFAULT.sample_rate)
    clips, starts = windows(y, A.DEFAULT.sample_rate, args.hop)
    emb, tags = enc(clips)
    p = head["model"].predict_proba(emb)[:, 1]
    hit = p >= head["threshold"]

    print(f"\n  {args.audio}  ({len(y) / A.DEFAULT.sample_rate:.1f} s, "
          f"{len(clips)} windows, threshold {head['threshold']:.3f})\n")
    for t, pi, h, tg in zip(starts, p, hit, tags):
        bar = "#" * int(round(pi * 20))
        top = ", ".join(labels[i] for i in np.argsort(tg)[::-1][:2])
        print(f"  {t:6.1f}s  {pi:.3f} {bar:20s} {'DRONE' if h else '     '}  {top}")

    run = longest_run(hit)
    verdict = "DRONE DETECTED" if run >= args.min_hits else "no drone"
    print(f"\n  {verdict}: {int(hit.sum())}/{len(hit)} windows above threshold, "
          f"longest run {run} (need {args.min_hits}); max p {p.max():.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
