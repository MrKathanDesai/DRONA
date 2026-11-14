#!/usr/bin/env python3
"""Outdoor aircraft: does the drone detector false-alarm on real aircraft?

    python scripts/eval_aircraft.py

Data: AeroSonicDB v1 from Zenodo record 8371595 (audio.zip and
sample_meta.csv), unpacked into data/raw_aerosonic/.

AeroSonicDB (YPAD-0523; Downward & Nordby 2023; CC BY-NC 4.0; Zenodo 8371595):
ADS-B-labelled outdoor recordings of low-flying aircraft near Adelaide airport,
human-annotated with when each aircraft is audible, plus "silence" clips of the
same places with no aircraft. Its own train/test split keeps every recording
session (time, place, microphone) on one side.

1. Held out: the deployed tier-4 detector on the AeroSonicDB test sessions.
   Aircraft windows come from the audible interval only.
2. Retrained: the same tier-4 recipe with AeroSonicDB training sessions added
   as negatives (folds 1-4 train, fold 5 validation), judged on the same
   held-out sessions and on the outdoor Svanström test set (drones,
   helicopters, background), which it never trains on.

Deployment rule, fixed before any result was seen: the retrained detector
becomes the live page's base detector only if it lowers aircraft false alarms
on the held-out sessions, loses at most 0.02 drone recall on Svanström, and
does not raise helicopter false alarms there.

Writes reports/aircraft.{json,md}; if the rule is met, also
data/models/detector_aircraft.joblib and data/models/base.json, which makes it
the base detector of the live page and of operator-feedback retraining.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from dronedet import feedback as FB
from dronedet import live as L
from dronedet.eval import protocol as P
from dronedet.features import audio as A
from dronedet.paths import DATA_DIR

RAW = DATA_DIR / "raw_aerosonic"
SR = L.SR
MAX_WINDOWS = 20                # per recording, so long overflights do not dominate
DRONE_STATES = ("drone", "unknown drone")
REPORTS = ROOT / "reports"
OUT = DATA_DIR / "models" / "detector_aircraft.joblib"


def load_meta() -> pd.DataFrame:
    m = pd.read_csv(RAW / "sample_meta.csv")
    paths = {p.name: p for p in RAW.rglob("*.wav") if "env_audio" not in p.parts}
    m["path"] = m.filename.map(paths)
    missing = m.path.isna().sum()
    if missing:
        raise FileNotFoundError(f"{missing} AeroSonicDB clips missing under {RAW}")
    m["kind"] = np.where(m["class"] == 1, "aircraft", "outdoor silence")
    m["group_id"] = "aero_" + m.session.astype(str) + "_" + m.location.astype(str) + "_" + m.mic.astype(str)
    return m


def windows(row) -> list[np.ndarray]:
    y = A.load_mono(row.path, SR)
    if row["class"] == 1:                                   # audible interval only
        a = int(row.offset * SR)
        y = y[a:a + int(row.duration * SR)]
    n = min(MAX_WINDOWS, len(y) // SR)
    if n == 0:
        return []
    starts = np.linspace(0, len(y) - SR, n).astype(int)
    return [y[s:s + SR] for s in starts]


def embed_rows(enc, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    F, rows = [], []
    for _, r in meta.iterrows():
        ws = windows(r)
        if not ws:
            continue
        F.append(enc(ws)[0].astype(np.float32))
        rows += [{"rec": r.filename, "kind": r.kind, "group_id": r.group_id,
                  "fold": r["fold"], "split": r["train-test"]}] * len(ws)
    return np.concatenate(F), pd.DataFrame(rows)


def rates(det, F, df) -> dict:
    hit = np.isin(det.assess(F)["state"], DRONE_STATES)
    out = {}
    for kind, sub in df.assign(hit=hit).groupby("kind"):
        out[kind] = {"windows": int(len(sub)), "window_rate": float(sub.hit.mean()),
                     "recordings": int(sub.rec.nunique()),
                     "recording_rate": float(sub.groupby("rec").hit.any().mean())}
    return out


def svanstrom(det, F0, df0) -> dict:
    te = (df0.split_tier4 == "test").values
    d = df0[te].assign(hit=np.isin(det.assess(F0[te])["state"], DRONE_STATES))
    cat = np.where(d.y == 1, "drone", d.bg_category.fillna("background"))
    return {k: float(v) for k, v in d.assign(cat=cat).groupby("cat").hit.mean().items()}


def main() -> int:
    t0 = time.time()
    meta = load_meta()
    live = L.LiveDetector("detector")
    enc, base = live.enc, live.det
    print(f"embedding {len(meta)} AeroSonicDB recordings ...", flush=True)
    Fa, da = embed_rows(enc, meta)
    test = (da.split == "test").values

    F0 = np.load(DATA_DIR / "embeddings" / "E.npy")
    df0 = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))

    # retrained: tier 4 + AeroSonicDB training sessions as negatives
    tr_a = (da.split == "train").values
    extra = pd.DataFrame({"group_id": da.group_id[tr_a].values, "y": 0, "source": "aerosonic",
                          "label": "background",
                          "split_tier4": np.where(da.fold[tr_a].values == 5, "val", "train")})
    F = np.concatenate([F0, Fa[tr_a]])
    df = pd.concat([df0, extra], ignore_index=True)
    retrained = FB.build(F, df)
    retrained.info["trained_on"] = "tier 4 + AeroSonicDB training sessions (aircraft, outdoor silence) as negatives"

    res = {"original": {"aerosonic_test": rates(base, Fa[test], da[test]), "svanstrom": svanstrom(base, F0, df0)},
           "retrained": {"aerosonic_test": rates(retrained, Fa[test], da[test]),
                         "svanstrom": svanstrom(retrained, F0, df0)}}
    o, r = res["original"], res["retrained"]
    deploy = bool(r["aerosonic_test"]["aircraft"]["window_rate"] < o["aerosonic_test"]["aircraft"]["window_rate"]
                  and r["svanstrom"]["drone"] >= o["svanstrom"]["drone"] - 0.02
                  and r["svanstrom"].get("helicopter", 0) <= o["svanstrom"].get("helicopter", 0))
    if deploy:
        joblib.dump(retrained, OUT)
        FB.BASE_POINTER.write_text(json.dumps({"model": OUT.name, "source": "scripts/eval_aircraft.py"}, indent=2) + "\n")
    rep = {"n": {"test_aircraft_recordings": int(da[test & (da.kind == "aircraft")].rec.nunique()),
                 "test_silence_recordings": int(da[test & (da.kind != "aircraft")].rec.nunique()),
                 "train_negative_windows": int(tr_a.sum())},
           "results": res, "deployed": deploy}
    (REPORTS / "aircraft.json").write_text(json.dumps(rep, indent=2))
    (REPORTS / "aircraft.md").write_text(markdown(rep))
    print(markdown(rep))
    print(f"total {time.time() - t0:.0f}s")
    return 0


def markdown(rep) -> str:
    n, res = rep["n"], rep["results"]
    md = ["# Outdoor aircraft: false alarms on real overflights", "",
          "AeroSonicDB (YPAD-0523, CC BY-NC 4.0): ADS-B-labelled recordings of low-flying aircraft near "
          "Adelaide airport, annotated with when each aircraft is audible, plus outdoor clips of the same "
          "places with no aircraft. Its own split keeps each recording session on one side. Test sessions: "
          f"{n['test_aircraft_recordings']} aircraft and {n['test_silence_recordings']} no-aircraft recordings. "
          "1 s windows from the audible interval (at most 20 per recording). Generated by `scripts/eval_aircraft.py`.", "",
          "- **original**: the deployed tier-4 detector, never trained on aircraft overflights.",
          f"- **retrained**: the same recipe with AeroSonicDB *training* sessions as negatives "
          f"({n['train_negative_windows']} windows; fold 5 used for the threshold).", "",
          "## Held-out AeroSonicDB sessions: share called drone", "",
          "| detector | aircraft windows | aircraft recordings | no-aircraft windows | no-aircraft recordings |",
          "|---|---|---|---|---|"]
    for k, v in res.items():
        a, s = v["aerosonic_test"]["aircraft"], v["aerosonic_test"]["outdoor silence"]
        md.append(f"| {k} | {a['window_rate']:.3f} | {a['recording_rate']:.2f} | "
                  f"{s['window_rate']:.3f} | {s['recording_rate']:.2f} |")
    md += ["", "Recordings: share with at least one drone window, which is what the live page's alarm "
           "rule (first drone window) would turn into an alarm.", "",
           "## Svanström outdoor test (never trained on): share of windows called drone", "",
           "| detector | drone (recall) | helicopter | outdoor background |", "|---|---|---|---|"]
    for k, v in res.items():
        s = v["svanstrom"]
        md.append(f"| {k} | {s.get('drone', 0):.3f} | {s.get('helicopter', 0):.3f} | "
                  f"{s.get('outdoor_background', 0):.3f} |")
    md += ["", "**Deployment rule** (fixed before the run): lower aircraft false alarms on the held-out "
           "sessions, at most 0.02 drone recall lost on Svanström, no more helicopter false alarms. "
           + ("**Met: the live page now uses the retrained detector as its base.**" if rep["deployed"]
              else "**Not met: the live page keeps the original detector.**"), ""]
    return "\n".join(md)


if __name__ == "__main__":
    raise SystemExit(main())
