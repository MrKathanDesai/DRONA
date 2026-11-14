#!/usr/bin/env python3
"""Does the DroneRF detector see drones, or the DroneRF session?

    python scripts/prepare_rf.py && python scripts/train_rf.py   # once
    python scripts/eval_rf_tampere.py

Applies the saved DroneRF forest (data/models/rf_forest.joblib), at its frozen
1%-FPR threshold, to the Tampere anechoic-chamber recordings
(data/raw_tampere/*_2G*.bin): another receiver, another session, other drones.
Nothing is trained or tuned here.

Two readings:

- recall on windows where a transmission is visible: does it detect drones
  it was never trained on, recorded on another receiver?
- scores on windows at the noise floor in both bands: a session-blind
  detector should call these "no drone". If it calls them "drone", it has
  learned "not the DroneRF background session" (the quiet-window AUC of 1.000
  in reports/rf_detection.md points that way).

Tampere has no drone-off recordings, so no false-alarm rate on real
background is measured. Writes reports/rf_tampere.{json,md} and caches the
PSD in data/rf_features_tampere/.
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

from dronedet.data import rf as R
from dronedet.data import rf_tampere as T
from dronedet.models import rf as M
from dronedet.paths import DATA_DIR

RAW = DATA_DIR / "raw_tampere"
CACHE = DATA_DIR / "rf_features_tampere"
MODEL = DATA_DIR / "models" / "rf_forest.joblib"
REPORTS = ROOT / "reports"
QUIET_ABOVE_FLOOR_DB = 7.0   # DroneRF: noise floor ~8 dB, quiet cut 15 dB
N_NOISE = 40                 # pure-noise control windows


def noise_control(n: int = N_NOISE, std: float = 50.0, seed: int = 0) -> np.ndarray:
    """PSDs of complex white Gaussian noise run through the same conversion:
    no drone, no session, nothing but the receiver chain's maths. Level does
    not matter (the forest's features remove each band's mean)."""
    rng = np.random.default_rng(seed)
    per_win = int(round(R.DEFAULT.window_s * T.FS_IN))
    iq = (rng.normal(0, std, (n * per_win, 2))).round().astype("<i2")
    return T.to_dronerf_psd(iq)[0]


def verdict(report) -> list[str]:
    nc = report["noise_control"]["called_drone"]
    if nc < 0.5:
        return ["Pure noise is mostly called \"no drone\", so on this receiver the "
                "forest responds to signal content."]
    return [f"**Verdict: the DroneRF forest does not detect drones.** It calls "
            f"{nc:.0%} of pure-noise windows \"drone\", so its rule is \"not the "
            "DroneRF background recording\". The 100% recall above says nothing "
            "about detection, and the same shortcut explains the near-perfect "
            "DroneRF scores and the quiet-window AUC of 1.000 in `rf_detection.md`. "
            "A usable RF detector needs training negatives from more than one "
            "session and receiver."]


def build_cache() -> tuple[np.ndarray, pd.DataFrame]:
    files = sorted(RAW.glob("*_2G*.bin"))
    if not files:
        raise SystemExit(f"no 2.4 GHz files in {RAW}; download Zenodo 4264467")
    key = [f"{f.name}:{f.stat().st_size}" for f in files] + [R.DEFAULT.fingerprint()]
    if (CACHE / "key.json").exists() and json.loads((CACHE / "key.json").read_text()) == key:
        return np.load(CACHE / "X.npy"), pd.read_parquet(CACHE / "index.parquet")
    Xs, rows = [], []
    for f in files:
        drone, part = T.parse_name(f)
        print(f"  {f.name} ...", flush=True)
        psd, level = T.to_dronerf_psd(T.load_iq(f))
        Xs.append(psd)
        rows += [dict(file=f.name, drone=drone, part=part, window=i,
                      level_db_low=float(level[i, 0]), level_db_high=float(level[i, 1]))
                 for i in range(len(psd))]
    X, df = np.concatenate(Xs), pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "X.npy", X)
    df.to_parquet(CACHE / "index.parquet")
    (CACHE / "key.json").write_text(json.dumps(key))
    return X, df


def main() -> int:
    if not MODEL.exists():
        raise SystemExit(f"{MODEL} missing; run scripts/train_rf.py")
    saved = joblib.load(MODEL)
    if saved["fingerprint"] != R.DEFAULT.fingerprint():
        raise SystemExit("saved forest was built with another RFConfig; retrain")
    X, df = build_cache()
    features = M.MODELS[M.DEPLOY][0]
    df["p"] = saved["model"].predict_proba(features(X))[:, 1]
    thr = saved["threshold"]
    df["hit"] = df.p >= thr

    # Noise floor per band from the quietest windows of the whole corpus;
    # receiver gain differs from DroneRF, so the cut is relative, not 15 dB.
    floor = {b: float(np.percentile(df[f"level_db_{b}"], 5)) for b in ("low", "high")}
    df["quiet"] = ((df.level_db_low < floor["low"] + QUIET_ABOVE_FLOOR_DB)
                   & (df.level_db_high < floor["high"] + QUIET_ABOVE_FLOOR_DB))

    per = (df.groupby("drone")
             .agg(windows=("hit", "size"), recall=("hit", "mean"),
                  median_p=("p", "median"), quiet=("quiet", "sum"))
             .reset_index())
    active, quiet = df[~df.quiet], df[df.quiet]
    X_noise = noise_control()
    np.save(CACHE / "noise_X.npy", X_noise)        # reused by eval_fusion.py
    p_noise = saved["model"].predict_proba(features(X_noise))[:, 1]
    report = {
        "threshold": thr,
        "noise_floor_db": floor,
        "quiet_above_floor_db": QUIET_ABOVE_FLOOR_DB,
        "windows": int(len(df)),
        "recall_active": float(active.hit.mean()) if len(active) else None,
        "active_windows": int(len(active)),
        "quiet_windows": int(len(quiet)),
        "quiet_called_drone": float(quiet.hit.mean()) if len(quiet) else None,
        "quiet_median_p": float(quiet.p.median()) if len(quiet) else None,
        "per_drone": per.to_dict(orient="records"),
        "noise_control": {"windows": N_NOISE,
                          "called_drone": float((p_noise >= thr).mean()),
                          "median_p": float(np.median(p_noise))},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "rf_tampere.json").write_text(json.dumps(report, indent=2))

    def pct(v):
        return "n/a" if v is None else f"{v:.3f}"

    md = [
        "# RF cross-dataset check: DroneRF forest on Tampere recordings", "",
        f"Saved DroneRF forest, threshold {thr:.3f} (1% FPR on DroneRF validation), "
        "applied unchanged to the Tampere anechoic-chamber recordings (Zenodo "
        "4264467): new receiver, new session, drones never trained on. Each "
        "2.44 GHz / 120 MS/s file is mixed to DroneRF's two 40 MHz bands and "
        "decimated to 40 MS/s; the real part stands in for DroneRF's single "
        "channel (an assumption). Generated by `scripts/eval_rf_tampere.py`.", "",
        f"- windows: {report['windows']} of 25 ms",
        f"- recall on windows with a visible transmission: **{pct(report['recall_active'])}** "
        f"({report['active_windows']} windows)",
        f"- windows at the noise floor in both bands (< floor + "
        f"{QUIET_ABOVE_FLOOR_DB:.0f} dB): {report['quiet_windows']}; called drone: "
        f"**{pct(report['quiet_called_drone'])}**, median score {pct(report['quiet_median_p'])}", "",
        f"- **pure white noise** through the same conversion ({N_NOISE} windows, "
        "no drone at all): called drone "
        f"**{report['noise_control']['called_drone']:.3f}**, median score "
        f"{report['noise_control']['median_p']:.3f}", "",
        *verdict(report), "",
        "No drone-off recordings exist in this corpus, so no false-alarm rate is "
        "measured. Quiet windows are the closest stand-in: a detector that reads "
        "drone signals should call them \"no drone\".", "",
        "| drone | windows | recall @ frozen threshold | median score | quiet windows |",
        "|---|---|---|---|---|",
        *[f"| {r.drone} | {r.windows} | {r.recall:.3f} | {r.median_p:.3f} | {r.quiet} |"
          for r in per.itertuples()], "",
    ]
    (REPORTS / "rf_tampere.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
