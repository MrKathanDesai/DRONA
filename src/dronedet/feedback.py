"""Operator feedback: labelled clips -> retrained detector, behind a safety gate.

Every incident an operator labels ("drone" / "not drone"), every drone they
report as missed, and every stretch they mark on the console's session review
strip, is saved as a 16 kHz clip in data/feedback/. Retraining
rebuilds the detector head on the original tier-4 training data plus every
labelled clip (the pretrained encoder stays frozen), then compares the new
model (challenger) with the one in use (champion):

1. Outdoor held-out benchmark (Svanström test, never used in training or
   feedback): drone recall may drop by at most RECALL_SLACK and the false-alarm
   rate must stay within max(champion's, TARGET_FPR).
2. The labelled clips themselves: the challenger is scored by cross-validation
   with each incident kept whole (it is never scored on a clip it trained on)
   and must do at least as well as the champion. Needs MIN_INCIDENTS labelled
   incidents; fewer, and only the benchmark check applies.

Only a challenger that passes both replaces the champion. Every promoted model
is kept as data/models/versions/vN.joblib, with its gate results in
data/models/registry.json, so the history can be shown and rolled back.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from dronedet.eval import metrics as E
from dronedet.eval import protocol as P
from dronedet.models import pretrained as PT
from dronedet.models import uncertainty as U
from dronedet.paths import DATA_DIR

SR = 16000
FEED = DATA_DIR / "feedback"
CLIPS = FEED / "clips"
LABELS = FEED / "labels.csv"
VERSIONS = DATA_DIR / "models" / "versions"
REGISTRY = DATA_DIR / "models" / "registry.json"
ORIGINAL = DATA_DIR / "models" / "detector.joblib"
BASE_POINTER = DATA_DIR / "models" / "base.json"          # written by scripts/eval_aircraft.py

TARGET_FPR = 0.01
REVIEW_RATE = 0.10
FAMILIAR_Q = 0.95
N_ENSEMBLE = 20
RECALL_SLACK = 0.02
MIN_INCIDENTS = 4
LABEL_Y = {"drone": 1, "not_drone": 0}
DRONE_STATES = ("drone", "unknown drone")


# --- storage -----------------------------------------------------------------

def save_clip(clip_id: str, label: str, audio: np.ndarray, input_name: str = "") -> dict:
    if label not in LABEL_Y:
        raise ValueError(label)
    if len(audio) < SR // 2:
        raise ValueError("clip shorter than half a second")
    CLIPS.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in clip_id if ch.isalnum() or ch in "-_")[:64]
    path = CLIPS / f"{safe}.wav"
    sf.write(path, np.clip(audio, -1, 1), SR)
    row = {"id": safe, "label": label, "seconds": round(len(audio) / SR, 2),
           "input": input_name[:120], "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    labels = load_labels()
    labels = pd.concat([labels[labels.id != safe], pd.DataFrame([row])], ignore_index=True)
    FEED.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS, index=False)
    return row


def load_labels() -> pd.DataFrame:
    if not LABELS.exists():
        return pd.DataFrame(columns=["id", "label", "seconds", "input", "saved_utc"])
    return pd.read_csv(LABELS, dtype={"id": str})


def registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"current": None, "history": []}


def base_path() -> Path:
    """The detector feedback builds on: the original, or a replacement that
    passed its own deployment rule (data/models/base.json)."""
    if BASE_POINTER.exists():
        p = DATA_DIR / "models" / json.loads(BASE_POINTER.read_text())["model"]
        if p.exists():
            return p
    return ORIGINAL


def current_model_path() -> Path:
    cur = registry().get("current")
    return VERSIONS / f"{cur}.joblib" if cur else base_path()


def status() -> dict:
    reg, labels = registry(), load_labels()
    used = set(reg["history"][-1]["clips"]) if reg["history"] else set()
    return {"version": reg.get("current") or "v0 (original)",
            "labelled": {k: int((labels.label == k).sum()) for k in LABEL_Y},
            "new": int((~labels.id.isin(used)).sum()),
            "last": reg["history"][-1] if reg["history"] else None}


# --- training ----------------------------------------------------------------

def _windows(path: Path) -> list[np.ndarray]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = y if y.ndim == 1 else y.mean(axis=1)
    n = max(1, len(y) // SR)
    y = np.pad(y, (0, max(0, n * SR - len(y))))
    return [y[i * SR:(i + 1) * SR] for i in range(n)]


def feedback_embeddings(enc) -> tuple[np.ndarray, pd.DataFrame]:
    """One row per 1 s window of every labelled clip (embeddings cached)."""
    labels = load_labels()
    F, rows = [], []
    for r in labels.itertuples():
        cache = CLIPS / f"{r.id}.npy"
        if cache.exists():
            e = np.load(cache)
        else:
            e = enc(_windows(CLIPS / f"{r.id}.wav"))[0].astype(np.float32)
            np.save(cache, e)
        F.append(e)
        rows += [{"group_id": f"fb_{r.id}", "y": LABEL_Y[r.label], "source": "feedback",
                  "label": "drone" if LABEL_Y[r.label] else "background", "split_tier4": "train"}] * len(e)
    if not F:
        return np.zeros((0, 1024), np.float32), pd.DataFrame(rows)
    return np.concatenate(F), pd.DataFrame(rows)


def _head(F, df, tr):
    w = P.source_weights(df[tr])
    return PT.head(balanced=False).fit(F[tr], df.y.values[tr], logisticregression__sample_weight=w), w


def build(F, df) -> U.Detector:
    """Same recipe as the deployed tier-4 detector (scripts/eval_uncertainty.py)."""
    y, g = df.y.values, df.group_id.values
    tr, va = (df.split_tier4 == "train").values, (df.split_tier4 == "val").values
    head, w = _head(F, df, tr)
    p_va = head.predict_proba(F[va])[:, 1]
    thr = E.threshold_at_fpr(p_va[y[va] == 0], TARGET_FPR)
    margin = U.margin_for_review_rate(U.logit(p_va) - U.logit(thr), REVIEW_RATE)
    fam = U.Familiarity().fit(F[tr]).calibrate(F[va], FAMILIAR_Q)
    fam_d = U.Familiarity().fit(F[tr & (y == 1)]).calibrate(F[va & (y == 1)], FAMILIAR_Q)
    ens = U.HeadEnsemble(lambda: PT.head(balanced=False), N_ENSEMBLE).fit(F[tr], y[tr], g[tr], w)
    return U.Detector(head, thr, margin, fam, fam_d, ens,
                      info={"target_fpr": TARGET_FPR, "encoder": PT.CHECKPOINT.name,
                            "trained_on": "tier 4 + operator-labelled clips"})


def _rates(hit: np.ndarray, y: np.ndarray) -> dict:
    return {"recall": float(hit[y == 1].mean()) if (y == 1).any() else None,
            "false_alarms": float(hit[y == 0].mean()) if (y == 0).any() else None}


def _balanced(hit, y) -> float | None:
    parts = [hit[y == 1].mean()] if (y == 1).any() else []
    parts += [1 - hit[y == 0].mean()] if (y == 0).any() else []
    return float(np.mean(parts)) if parts else None


def _cv_feedback(F, df, n_base: int) -> np.ndarray:
    """Out-of-fold drone calls for every feedback window, folds = whole incidents."""
    fb = np.arange(n_base, len(df))
    groups = df.group_id.values[fb]
    uniq = np.array(sorted(set(groups)))
    k = min(5, len(uniq))
    fold_of = {g: i % k for i, g in enumerate(np.random.default_rng(0).permutation(uniq))}
    hit = np.zeros(len(fb), bool)
    va = (df.split_tier4 == "val").values
    y = df.y.values
    for f in range(k):
        held = np.array([fold_of[g] == f for g in groups])
        tr = (df.split_tier4 == "train").values.copy()
        tr[fb[held]] = False
        head, _ = _head(F, df, tr)
        thr = E.threshold_at_fpr(head.predict_proba(F[va])[:, 1][y[va] == 0], TARGET_FPR)
        hit[held] = head.predict_proba(F[fb[held]])[:, 1] >= thr
    return hit


def retrain(enc, champion: U.Detector) -> dict:
    """Build a challenger from base data + feedback, gate it, promote if it passes."""
    t0 = time.time()
    st = status()
    if st["labelled"]["drone"] + st["labelled"]["not_drone"] == 0:
        return {"promoted": False, "reason": "No labelled clips yet."}
    if st["new"] == 0:
        return {"promoted": False, "reason": "Nothing new has been labelled since the last retrain."}
    F0 = np.load(DATA_DIR / "embeddings" / "E.npy")
    df0 = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))
    Ffb, dffb = feedback_embeddings(enc)
    F = np.concatenate([F0, Ffb])
    df = pd.concat([df0, dffb], ignore_index=True)
    challenger = build(F, df)

    te = (df0.split_tier4 == "test").values                     # outdoor, never trained on
    y_te = df0.y.values[te]
    bench = {name: _rates(np.isin(d.assess(F0[te])["state"], DRONE_STATES), y_te)
             for name, d in (("champion", champion), ("challenger", challenger))}
    ok_bench = (bench["challenger"]["recall"] >= bench["champion"]["recall"] - RECALL_SLACK
                and bench["challenger"]["false_alarms"] <= max(bench["champion"]["false_alarms"], TARGET_FPR))

    y_fb = dffb.y.values
    n_inc = dffb.group_id.nunique()
    fb = {"incidents": int(n_inc),
          "champion": _balanced(np.isin(champion.assess(Ffb)["state"], DRONE_STATES), y_fb)}
    if n_inc >= MIN_INCIDENTS:
        fb["challenger"] = _balanced(_cv_feedback(F, df, len(df0)), y_fb)
        ok_fb = fb["challenger"] >= fb["champion"]
    else:
        fb["challenger"], ok_fb = None, True

    promoted = bool(ok_bench and ok_fb)
    reg = registry()
    version = f"v{len([h for h in reg['history'] if h['promoted']]) + 1}"
    entry = {"version": version if promoted else None, "promoted": promoted,
             "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "clips": sorted(load_labels().id.tolist()), "benchmark_outdoor": bench, "feedback": fb,
             "seconds": round(time.time() - t0, 1)}
    if not ok_bench:
        entry["reason"] = "It would miss more drones or raise more false alarms on the held-out benchmark."
    elif not ok_fb:
        entry["reason"] = "It did not judge the labelled clips better than the current model."
    else:
        entry["reason"] = ("Passed the benchmark and judged the labelled clips at least as well."
                           if fb["challenger"] is not None else
                           f"Passed the benchmark (fewer than {MIN_INCIDENTS} labelled incidents, so not yet measured on them).")
    if promoted:
        VERSIONS.mkdir(parents=True, exist_ok=True)
        joblib.dump(challenger, VERSIONS / f"{version}.joblib")
        reg["current"] = version
    reg["history"].append(entry)
    REGISTRY.write_text(json.dumps(reg, indent=2))
    entry["model"] = challenger if promoted else None
    return entry


def rollback() -> str | None:
    """Go back to the previous promoted version (or the original)."""
    reg = registry()
    promoted = [h["version"] for h in reg["history"] if h["promoted"]]
    if reg.get("current") in promoted:
        i = promoted.index(reg["current"])
        reg["current"] = promoted[i - 1] if i > 0 else None
        REGISTRY.write_text(json.dumps(reg, indent=2))
    return reg["current"]
