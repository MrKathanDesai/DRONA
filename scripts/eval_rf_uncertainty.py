#!/usr/bin/env python3
"""RF uncertainty flags and open-set "unknown signature" detection.

    python scripts/prepare_rf.py && python scripts/train_rf.py   # once
    python scripts/eval_rf_uncertainty.py

The same machinery as the audio detector (`models.uncertainty`), on the RF
forest's 128 sub-band features:

- review band: windows whose score is within the band holding 10% of
  validation windows nearest the threshold are "low confidence";
- open set: hold each drone out of training in turn; the Mahalanobis distance
  to the *known* drones' features, cut at the 95th percentile of known-drone
  validation windows, should flag the held-out drone as an unknown signature
  more often than held-out time from known drones.

All cut-offs come from validation. Saves the four-state RF detector to
data/models/rf_detector_dronerf.joblib (v1; the deployed RF detector is v2,
scripts/train_rf_v2.py) and writes
reports/rf_uncertainty.{json,md}. The RF caveat stands: DroneRF scores look
like session artefacts (reports/rf_detection.md), so these flags inherit it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dronedet.data import rf as R
from dronedet.eval import metrics as E
from dronedet.eval import protocol as P
from dronedet.models import rf as M
from dronedet.models import uncertainty as U
from dronedet.paths import DATA_DIR

FEATURES = DATA_DIR / "rf_features"
OUT = DATA_DIR / "models" / "rf_detector_dronerf.joblib"   # v1; v2 is rf_detector.joblib
REPORTS = ROOT / "reports"
TARGET_FPR = 0.01
REVIEW_RATE = 0.10
FAMILIAR_Q = 0.95
DRONES = ("bebop", "ar", "phantom")
# Forest probabilities hit exactly 0 and 1; clip before taking logits so the
# margin stays finite (U.logit clips at 1e-7 already; this is for clarity).


def load():
    X = np.load(FEATURES / "X.npy")
    df = pd.read_parquet(FEATURES / "index.parquet")
    df["bg_category"] = np.where(df.label == "background", "rf_background", None)
    return M.MODELS[M.DEPLOY][0](X), P.with_category(df)


def build_detector(F, df) -> U.Detector:
    y = df.y.values
    tr, va = (df.split_tier1 == "train").values, (df.split_tier1 == "val").values
    head = M.forest().fit(F[tr], y[tr])
    p_va = head.predict_proba(F[va])[:, 1]
    thr = E.threshold_at_fpr(p_va[y[va] == 0], TARGET_FPR)
    margin = U.margin_for_review_rate(U.logit(p_va) - U.logit(thr), REVIEW_RATE)
    fam = U.Familiarity().fit(F[tr]).calibrate(F[va], FAMILIAR_Q)
    fam_d = U.Familiarity().fit(F[tr & (y == 1)]).calibrate(F[va & (y == 1)], FAMILIAR_Q)
    return U.Detector(head, thr, margin, fam, fam_d, None,
                      info={"target_fpr": TARGET_FPR, "review_rate": REVIEW_RATE,
                            "familiar_quantile": FAMILIAR_Q, "modality": "rf",
                            "fingerprint": R.DEFAULT.fingerprint()})


def tier1_flags(det, F, df) -> dict:
    te = (df.split_tier1 == "test").values
    a = det.assess(F[te])
    y = df.y.values[te]
    err = (a["p"] >= det.threshold) != (y == 1)
    flagged = a["state"] == "low confidence"
    return {
        "windows": int(te.sum()),
        "error_rate": float(err.mean()),
        "flagged_rate": float(flagged.mean()),
        "errors_caught": float(err[flagged].sum() / max(err.sum(), 1)),
        "states": pd.Series(a["state"]).value_counts().to_dict(),
        "risk_coverage_margin": U.risk_coverage(err, -np.abs(a["margin"])),
    }


def open_set(F, df) -> dict:
    """Hold out one drone; is it farther from the known drones than held-out
    time from the known drones is?"""
    y = df.y.values
    out = {}
    for d in DRONES:
        known = (y == 1) & (df.drone_model != d).values
        tr = known & (df.split_tier1 == "train").values
        va = known & (df.split_tier1 == "val").values
        te_known = known & (df.split_tier1 == "test").values
        unknown = (df.drone_model == d).values
        fam = U.Familiarity().fit(F[tr]).calibrate(F[va], FAMILIAR_Q)
        dk, du = fam.distance(F[te_known]), fam.distance(F[unknown])
        out[d] = {
            "auroc": float(roc_auc_score(np.r_[np.zeros(len(dk)), np.ones(len(du))],
                                         np.r_[dk, du])),
            "flagged_unknown": float((du > fam.cutoff).mean()),
            "flagged_known": float((dk > fam.cutoff).mean()),
            "unknown_windows": int(len(du)), "known_windows": int(len(dk)),
        }
    return out


def main() -> int:
    F, df = load()
    det = build_detector(F, df)
    flags = tier1_flags(det, F, df)
    osr = open_set(F, df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(det, OUT)

    report = {"target_fpr": TARGET_FPR, "review_rate": REVIEW_RATE,
              "familiar_quantile": FAMILIAR_Q, "threshold": det.threshold,
              "margin_logit": det.margin, "tier1": flags, "open_set": osr}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "rf_uncertainty.json").write_text(json.dumps(report, indent=2))

    rc = flags["risk_coverage_margin"]["error_at_coverage"]
    md = [
        "# RF uncertainty and open-set flags (DroneRF)", "",
        "**Inherits the RF caveat:** DroneRF scores look like recording-session "
        "artefacts (see `rf_detection.md`), so low error rates here do not show "
        "real-world reliability. Generated by `scripts/eval_rf_uncertainty.py`.", "",
        f"Forest on 128 level-free sub-band features, threshold at {TARGET_FPR:.0%} "
        f"FPR on validation; review band = the {REVIEW_RATE:.0%} of validation "
        "windows nearest the threshold; familiarity cut at the "
        f"{FAMILIAR_Q:.0%} quantile of validation distances.", "",
        "## Tier 1 (held-out time)", "",
        f"- test windows: {flags['windows']}, error rate {flags['error_rate']:.3f}",
        f"- flagged low confidence: {flags['flagged_rate']:.1%}, holding "
        f"{flags['errors_caught']:.0%} of the errors",
        "- states: " + ", ".join(f"{k} {v}" for k, v in flags["states"].items()),
        "- error rate on the most certain windows (margin): "
        + ", ".join(f"{k} coverage {v:.3f}" for k, v in rc.items()), "",
        "## Open set: is a drone never trained on flagged as an unknown signature?", "",
        "Distance to the known drones' features; known = held-out time of the "
        "other two drones.", "",
        "| held-out drone | AUROC unknown vs known | unknown flagged | known flagged |",
        "|---|---|---|---|",
        *[f"| {d} | {r['auroc']:.3f} | {r['flagged_unknown']:.1%} ({r['unknown_windows']}) "
          f"| {r['flagged_known']:.1%} ({r['known_windows']}) |" for d, r in osr.items()], "",
        f"Detector saved to `data/models/{OUT.name}`.", "",
    ]
    (REPORTS / "rf_uncertainty.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
