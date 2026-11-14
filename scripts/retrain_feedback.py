#!/usr/bin/env python3
"""Retrain the audio detector on operator-labelled clips, behind a safety gate.

    python scripts/retrain_feedback.py            # retrain, promote only if the gate passes
    python scripts/retrain_feedback.py --status   # labelled clips and version history
    python scripts/retrain_feedback.py --rollback # back to the previous version

The live page (python -m console) does the same from its Model panel. The
labelled clips come from the incident queue there (Confirm drone / False alarm
/ Report missed drone). See src/dronedet/feedback.py for the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib

from dronedet import feedback as FB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(FB.status(), indent=2))
        return 0
    if a.rollback:
        print("now using", FB.rollback() or "the original detector")
        return 0
    from dronedet import live as L
    enc = L.LiveDetector("detector").enc
    champion = joblib.load(FB.current_model_path())
    res = FB.retrain(enc, champion)
    res.pop("model", None)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
