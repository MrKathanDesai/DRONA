#!/usr/bin/env python3
"""Audio detector that also works through a speaker and a microphone.

    python scripts/train_mic_detector.py      # ~15-25 min on CPU

The deployed detector (scripts/eval_uncertainty.py) was trained on recordings
as they were captured. Played from a laptop or phone speaker and picked up by
a laptop microphone, a drone loses its low rotor hum (small speakers cannot
reproduce it) and gains room echo and hiss; the detector then misses it.

This keeps everything else the same (frozen pretrained encoder, tier-4 split,
source balancing, 1% false alarms on validation negatives, uncertainty flags)
and adds one copy of each training clip passed through a simulated
speaker -> room -> microphone channel (`speaker_mic`). Validation also gets a
channel copy, so the threshold holds for both clean and played-back sound.
Test clips get their own channel copies with a different seed.

Writes data/models/detector_mic.joblib and reports/mic_robustness.{json,md}.
Opt-in on the live page: python -m console --mic-model speaker.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

from dronedet import live as L
from dronedet.eval import protocol as P
from dronedet.features import audio as A
from dronedet.paths import DATA_DIR

spec = importlib.util.spec_from_file_location("unc", ROOT / "scripts" / "eval_uncertainty.py")
UNC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(UNC)

OUT = DATA_DIR / "models" / "detector_mic.joblib"
REPORTS = ROOT / "reports"
SR = L.SR
BG_PER_SPLIT = {"train": 3000, "val": 600}       # background clips to augment (all drones are)
COPIES = 3                                        # channel copies per augmented clip


def speaker_mic(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A small speaker in a room, heard by a laptop microphone.

    - speaker: loses the low end (high-pass 150-1200 Hz) and the top (4-7.5 kHz)
    - room: two early echoes
    - microphone: hiss and room noise 5-30 dB below the sound
    Level is irrelevant (the encoder normalises loudness)."""
    z = sosfilt(butter(2, rng.uniform(150, 1200), "highpass", fs=SR, output="sos"), y)
    z = sosfilt(butter(2, rng.uniform(4000, 7500), "lowpass", fs=SR, output="sos"), z)
    for lo, hi, g in ((.003, .012, rng.uniform(.2, .45)), (.015, .045, rng.uniform(.1, .3))):
        dly = int(rng.uniform(lo, hi) * SR)
        z = z + g * np.concatenate([np.zeros(dly), z[:-dly]])
    p = np.sqrt(np.mean(z ** 2)) + 1e-9
    noise = rng.normal(0, 1, len(z))
    noise = sosfilt(butter(1, rng.uniform(200, 3000), "lowpass", fs=SR, output="sos"), noise)
    noise *= p / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * 10 ** (-rng.uniform(5, 30) / 20)
    return (z + noise).astype(np.float32)


def clips_for(rows: pd.DataFrame) -> list[np.ndarray]:
    return [A.load_clip(DATA_DIR / p, s) for p, s in zip(rows.path, rows.start_s)]


def embed(enc, clips, batch=64):
    out = []
    for i in range(0, len(clips), batch):
        e, _ = enc(clips[i:i + batch])
        out.append(e)
    return np.concatenate(out).astype(np.float32)


def rates(det, F, y) -> dict:
    a = det.assess(F)
    hit = np.isin(a["state"], ["drone", "unknown drone"])
    return {"drone_recall": float(hit[y == 1].mean()) if (y == 1).any() else None,
            "false_alarm_rate": float(hit[y == 0].mean()) if (y == 0).any() else None}


def main() -> int:
    t0 = time.time()
    F = np.load(DATA_DIR / "embeddings" / "E.npy")
    df = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))
    det_old = joblib.load(DATA_DIR / "models" / "detector.joblib")
    enc = L.LiveDetector("detector").enc
    rng = np.random.default_rng(7)

    # --- channel copies of training and validation clips -------------------
    aug_idx = []
    for split, n_bg in BG_PER_SPLIT.items():
        s = df[df.split_tier4 == split]
        if split == "val":             # indoor validation clips are the evaluation set below
            s = s[s.source != "droneaudio"]
        aug_idx += list(s[s.y == 1].index)
        aug_idx += list(s[s.y == 0].sample(min(n_bg, (s.y == 0).sum()), random_state=1).index)
    base = df.loc[aug_idx]
    rows = pd.concat([base] * COPIES)
    print(f"channel copies: {len(rows)} ({COPIES} per clip) ...", flush=True)
    clean = clips_for(base)
    F_aug = embed(enc, [speaker_mic(c, rng) for _ in range(COPIES) for c in clean])
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    F2 = np.concatenate([F, F_aug])
    df2 = pd.concat([df, rows.assign(group_id=rows.group_id)], ignore_index=True)
    tr = (df2.split_tier4 == "train").values
    va = (df2.split_tier4 == "val").values
    w = P.source_weights(df2[tr])
    det_new = UNC.build(F2, df2, tr, va, UNC.head_weighted, w)
    det_new.info["trained_on"] = "tier 4 + simulated speaker->room->microphone copies"

    # --- test: held-out clean, and held-out through the channel ------------
    test = {}
    rng_t = np.random.default_rng(99)
    # tier-1 test is tier-4 training data; indoor validation is the only indoor
    # audio the detector weights never saw (it does set the threshold)
    t1 = df[(df.split_tier4 == "val") & (df.source == "droneaudio")]
    t1 = pd.concat([t1[t1.y == 1], t1[t1.y == 0].sample(600, random_state=2)])
    sv = df[(df.split_tier4 == "test")]
    for name, rows_t in (("indoor validation", t1), ("outdoor held-out (Svanström)", sv)):
        y = rows_t.y.values
        Fc = F[rows_t.index.values]
        Fm = embed(enc, [speaker_mic(c, rng_t) for c in clips_for(rows_t)])
        test[name] = {
            "clean": {"before": rates(det_old, Fc, y), "after": rates(det_new, Fc, y)},
            "through speaker + mic": {"before": rates(det_old, Fm, y), "after": rates(det_new, Fm, y)},
        }
        print(name, json.dumps(test[name]), flush=True)

    joblib.dump(det_new, OUT)
    rep = {"augmented_clips": int(len(rows)), "test": test}
    (REPORTS / "mic_robustness.json").write_text(json.dumps(rep, indent=2))
    md = ["# Microphone robustness: detector retrained with speaker -> mic copies", "",
          "Played from a speaker and heard by a laptop microphone, a drone loses its low "
          "rotor hum and gains echo and hiss. The deployed detector was never trained on that. "
          "`scripts/train_mic_detector.py` adds simulated speaker -> room -> microphone copies "
          f"x{COPIES} of each training clip ({len(rows)} copies; all drones, a sample of backgrounds) and "
          "rebuilds the same tier-4 detector (same split, source balancing, 1% false alarms on "
          "validation negatives, uncertainty flags). Test clips get their own channel copies. Indoor rows "
          "use validation recordings: the tier-1 test split is tier-4 training data, so it cannot be used.", "",
          "| test set | input | drone recall before | after | false alarms before | after |",
          "|---|---|---|---|---|---|"]
    for name, r in test.items():
        for inp, v in r.items():
            md.append(f"| {name} | {inp} | {v['before']['drone_recall']:.2f} | "
                      f"{v['after']['drone_recall']:.2f} | {v['before']['false_alarm_rate']:.3f} | "
                      f"{v['after']['false_alarm_rate']:.3f} |")
    md += ["", "The channel is simulated, not measured on a real speaker and microphone, so "
           "real playback may differ. It is lower on clean audio, so the live page keeps the original detector by default; `python -m console --mic-model speaker` switches the microphone to this one.", ""]
    (REPORTS / "mic_robustness.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
