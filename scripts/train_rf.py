#!/usr/bin/env python3
"""RF drone detection on DroneRF: classical models vs a small 1D-CNN.

    python scripts/prepare_rf.py     # once
    python scripts/train_rf.py

Same metrics and 1%-FPR operating point as the audio models. Tier 1 is held-out
time; tier 2 holds out each drone in turn, since Phantom alone is one reading.
There is no tier 3: DroneRF is one lab session. Writes
reports/rf_detection.{json,md} and data/models/rf_forest.joblib.
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
from sklearn.metrics import confusion_matrix, roc_auc_score

from dronedet.data import rf as R
from dronedet.eval import protocol as P
from dronedet.models import rf as M
from dronedet.paths import DATA_DIR

FEATURES = DATA_DIR / "rf_features"
REPORTS = ROOT / "reports"
MODEL_OUT = DATA_DIR / "models" / "rf_forest.joblib"
TARGET_FPR = 0.01
DRONES = ("bebop", "ar", "phantom")
MIN_HITS = 2
QUIET_DB = 15.0     # window level; the noise floor sits near 8 dB


def fmt(r) -> str:
    lo, hi = r["pr_auc_ci95"]
    return (f"| {r['pr_auc']:.3f} [{lo:.3f}, {hi:.3f}] | {r['roc_auc']:.3f} "
            f"| {r['recall']:.3f} | {r['fpr']:.4f} ({r['false_alarms']}/{r['negatives']}) |")


def segment_level(df, scores, thr, tier="split_tier1") -> dict:
    """Decision per 0.25 s segment: alarm on MIN_HITS consecutive windows."""
    t = df[df[tier] == "test"].assign(hit=scores >= thr)
    rows = []
    for _, r in t.sort_values("start_s").groupby(["bui", "segment"]):
        run = best = 0
        for h in r.hit:
            run = run + 1 if h else 0
            best = max(best, run)
        rows.append((r.label.iloc[0], best >= MIN_HITS))
    s = pd.DataFrame(rows, columns=["label", "alarm"]).groupby("label").alarm
    return {k: [int(v.sum()), len(v)] for k, v in s}


def quiet_probe(F, df) -> dict:
    """Skill with nothing visibly transmitting. Train and test only on windows
    at the noise floor in both bands: a score near 1 here cannot come from a
    drone's bursts, so it is the session or receiver, not the drone."""
    q = (df.level_db_low < QUIET_DB) & (df.level_db_high < QUIET_DB)
    tr, te = q & (df.split_tier1 == "train"), q & (df.split_tier1 == "test")
    model = M.forest().fit(F[tr], df.y[tr])
    return {"auc": float(roc_auc_score(df.y[te], model.predict_proba(F[te])[:, 1])),
            "train": int(tr.sum()), "test": int(te.sum()),
            "quiet_drone": int((q & (df.y == 1)).sum()),
            "quiet_background": int((q & (df.y == 0)).sum())}


def identify(F, df) -> tuple[list[str], np.ndarray]:
    """Which drone? Four classes, tier 1 split, forest."""
    names = ["background", *DRONES]
    y = df.drone_model.fillna("background")
    tr, te = df.split_tier1 == "train", df.split_tier1 == "test"
    model = M.forest().fit(F[tr], y[tr])
    return names, confusion_matrix(y[te], model.predict(F[te]), labels=names)


def main() -> int:
    X = np.load(FEATURES / "X.npy")
    df = pd.read_parquet(FEATURES / "index.parquet")
    cfg = json.loads((FEATURES / "config.json").read_text())
    if len(cfg["archives"]) != 23:
        print(f"  only {len(cfg['archives'])} of 23 archives processed; "
              "run python scripts/prepare_rf.py")
        return 1
    df["bg_category"] = np.where(df.label == "background", "rf_background", None)
    df = P.with_category(df)
    print(f"  PSD {X.shape}, {df.group_id.nunique()} blocks")

    probes = {}
    for col in ("level_db_low", "level_db_high"):
        probes[col] = float(roc_auc_score(df.y, df[col]))

    report = {"target_fpr": TARGET_FPR, "level_probe_auc": probes, "models": {}}
    for name, (features, make) in M.MODELS.items():
        F = features(X)
        print(f"  {name}: tier 1 ...", flush=True)
        t1, _ = P.tier1(F, df, make, TARGET_FPR)
        row = {"tier1_test": t1}
        for drone in DRONES:
            print(f"  {name}: tier 2, unseen {drone} ...", flush=True)
            d = R.assign_splits(df.drop(columns=["block", "group_id"]), held_out=drone)
            row[f"tier2_{drone}"] = P.tier2(F, d, make, TARGET_FPR)
        report["models"][name] = row

    F = M.MODELS[M.DEPLOY][0](X)
    model, thr = P.fit_deployable(F, df, M.MODELS[M.DEPLOY][1], TARGET_FPR, "split_tier1")
    te = (df.split_tier1 == "test").values
    seg = segment_level(df, model.predict_proba(F[te])[:, 1], thr)
    names, conf = identify(F, df)
    quiet = quiet_probe(F, df)
    report["quiet_probe"] = quiet
    report["segment_level"] = seg
    report["identify"] = {"classes": names, "confusion": conf.tolist()}
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "target_fpr": TARGET_FPR,
                 "fingerprint": cfg["fingerprint"]}, MODEL_OUT)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "rf_detection.json").write_text(json.dumps(report, indent=2))

    head = ("| model | split | PR-AUC [95% CI] | ROC-AUC | recall @ val-1%-FPR "
            "| test FPR (false alarms) |\n|---|---|---|---|---|---|")
    rows = [f"| {n} | tier 1 " + fmt(r["tier1_test"]) for n, r in report["models"].items()]
    for drone in DRONES:
        rows += [f"| {n} | tier 2, unseen {drone} " + fmt(r[f"tier2_{drone}"])
                 for n, r in report["models"].items()]
    conf_rows = [f"| {n} | " + " | ".join(str(v) for v in c) + " |"
                 for n, c in zip(names, conf)]
    md = [
        "# RF drone detection (DroneRF)", "",
        f"**Do not read these scores as detection performance.** On windows at the "
        f"noise floor, with nothing visibly transmitting, the forest still separates "
        f"drone from background at AUC {quiet['auc']:.3f} ({quiet['train']} train, "
        f"{quiet['test']} test windows). That points to the recording session or "
        "receiver, and background is one session, so DroneRF alone cannot remove "
        "it. A weak drone signal below the level cut is not ruled out.", "",
        f"Log-PSD per 25 ms window, both halves of the 2.4 GHz band. Threshold fixed "
        f"on held-out negatives at {TARGET_FPR:.0%} FPR, applied unchanged to test. "
        "CIs resample blocks of 3 consecutive segments. Generated by "
        "`scripts/train_rf.py`.", "",
        "- tier 1: last blocks of each recording held out, same drones",
        "- tier 2: one drone never trained on, plus the last background blocks",
        "- no tier 3: DroneRF is a single lab session, so nothing here measures "
        "another place, range or receiver", "",
        "`forest+level` keeps absolute level and is a shortcut probe, not a "
        "candidate. Every other row is level-invariant.", "",
        f"Level alone, AUC: low band {probes['level_db_low']:.3f}, "
        f"high band {probes['level_db_high']:.3f}.", "",
        head, *rows, "",
        f"## Per 0.25 s segment, tier 1 test ({M.DEPLOY}, {MIN_HITS} consecutive windows)", "",
        "| class | alarms | segments |", "|---|---|---|",
        *[f"| {k} | {a} | {n} |" for k, (a, n) in seg.items()], "",
        "## Which drone? Tier 1 test windows, true class in rows", "",
        "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|",
        *conf_rows, "",
        "Background was recorded separately from the drones, so a model can also "
        "separate the sessions rather than the signals. Test negatives are few, "
        "so the FPR is coarse.", "",
    ]
    (REPORTS / "rf_detection.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
