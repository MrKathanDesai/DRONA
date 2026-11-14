#!/usr/bin/env python3
"""Music masking: can a drone hidden under loud music still be detected?

    python scripts/eval_masking.py

Smugglers are reported to play loud music to hide a drone's sound. This builds
a masking benchmark and tests two defences, alone and together:

- front-end: stationary-component extraction (`features/unmask.py`), which
  keeps steady rotor tones and drops changing music, applied before the
  unchanged detector;
- training: a music-aware detector, the same tier-4 detector rebuilt with
  drone + music mixtures as positives and music alone as negatives.

Benchmark: DroneAudioDataset recordings from the tier-4 *validation* split
(consecutive 1 s slices joined back into 4-5 s recordings). The deployed
tier-4 detector trains on every other indoor recording, including the tier-1
test split, so validation is the only indoor audio its weights never saw; it
does set each detector's threshold, so these are validation-level numbers.
Drones are mixed with GTZAN test-split music at SNR clean, +10, +5, 0, -5, -10 dB (drone power
over music power); negatives are music alone, background recordings, and
background + music at 0 dB. GTZAN excerpts alternate between test and train
as in `features/noise.py`, so no excerpt is in both. Windows are 1 s, judged
by each detector at its own threshold (1% false alarms on its validation
negatives).

Deployment rule for the live page, fixed before any result was seen: use the
front-end if it raises drone recall by >= 0.10 at both 0 dB and -5 dB, keeps
false alarms on music alone <= 0.02, and costs <= 0.05 recall on clean drones.

Writes reports/masking.{json,md} and reports/figures/12_masking.png.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dronedet import live as L
from dronedet.eval import protocol as P
from dronedet.features import audio as A
from dronedet.features import noise as N
from dronedet.features.unmask import stationary
from dronedet.paths import DATA_DIR

spec = importlib.util.spec_from_file_location("unc", ROOT / "scripts" / "eval_uncertainty.py")
UNC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(UNC)

SR = L.SR
SNRS = [None, 10, 5, 0, -5, -10]            # None = clean
DRONE_STATES = ("drone", "unknown drone")
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures" / "12_masking.png"


# --- data ---------------------------------------------------------------------

def gtzan_files(split: str) -> list[str]:
    files = N._external_files(DATA_DIR / "raw_noise", "music_real")
    groups = sorted({g for _, g in files})
    keep = set(groups[0::2] if split == "test" else groups[1::2])
    return [f for f, g in files if g in keep]


def music_segments(files, n: int, seconds: float, rng) -> list[np.ndarray]:
    """n non-silent segments, spread over the files (several per file if needed)."""
    need = int(seconds * SR)
    per_file = max(1, -(-n // max(1, len(files))) + 1)
    out = []
    for f in rng.permutation(files):
        try:
            y = A.load_mono(f, SR)
        except Exception:                 # GTZAN has one known corrupt file
            continue
        starts = np.arange(0, len(y) - need + 1, need)
        for a in rng.permutation(starts)[:per_file]:
            seg = y[a:a + need]
            if N.rms(seg) >= N.MIN_RMS:
                out.append(seg.astype(np.float32))
        if len(out) >= n:
            break
    return out[:n]


def recordings(df: pd.DataFrame, label: str, n: int | None, rng, min_slices: int = 4):
    """Consecutive 1 s DroneAudioDataset slices joined back into recordings."""
    d = df[(df.source == "droneaudio") & (df.label == label)]
    groups = [g for g, s in d.groupby("group_id") if len(s) >= min_slices]
    if n is not None and len(groups) > n:
        groups = list(rng.choice(groups, n, replace=False))
    out = []
    for g in groups:
        paths = sorted(d[d.group_id == g].path)[:5]
        out.append(np.concatenate([A.fix_length(A.load_mono(DATA_DIR / p, SR), SR) for p in paths]))
    return out


def windows(y: np.ndarray) -> list[np.ndarray]:
    return [y[a:a + SR] for a in range(0, len(y) - SR + 1, SR)]


# --- detection ----------------------------------------------------------------

def embed(enc, clips, batch=64):
    out = []
    for i in range(0, len(clips), batch):
        out.append(enc(clips[i:i + batch])[0])
    return np.concatenate(out).astype(np.float32)


def hits(det, F) -> np.ndarray:
    return np.isin(det.assess(F)["state"], DRONE_STATES)


def build_music_detector(enc, rng) -> object:
    """Tier-4 detector rebuilt with music mixtures (train/val music only)."""
    F = np.load(DATA_DIR / "embeddings" / "E.npy")
    df = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))
    music = gtzan_files("train")
    rng.shuffle(music)
    music_tr, music_va = music[: len(music) * 3 // 4], music[len(music) * 3 // 4:]
    new_F, new_rows = [], []
    for split, files, n_mix, n_music in (("train", music_tr, 3000, 1500), ("val", music_va, 600, 300)):
        s = df[df.split_tier4 == split]
        if split == "val":                 # keep the benchmark's indoor drones out of its threshold
            s = s[s.source != "droneaudio"]
        drones = s[s.y == 1].sample(min(n_mix, (s.y == 1).sum()), random_state=3)
        clips = [A.load_clip(DATA_DIR / p, st) for p, st in zip(drones.path, drones.start_s)]
        m1 = music_segments(files, len(clips) + n_music, 1.0, rng)
        mixes = [N.mix(c, m, rng.uniform(-10, 10)) for c, m in zip(clips, m1)]
        new_F.append(embed(enc, mixes))
        new_rows.append(drones.assign(group_id=drones.group_id + "_music"))
        mus = m1[len(clips):]
        new_F.append(embed(enc, mus))
        new_rows.append(pd.DataFrame({
            "path": "gtzan", "label": "background", "group_id": [f"gtzan_{split}_{i}" for i in range(len(mus))],
            "source": "gtzan", "y": 0, "split_tier4": split}))
    F2 = np.concatenate([F, *new_F])
    df2 = pd.concat([df, *new_rows], ignore_index=True)
    tr, va = (df2.split_tier4 == "train").values, (df2.split_tier4 == "val").values
    det = UNC.build(F2, df2, tr, va, UNC.head_weighted, P.source_weights(df2[tr]))
    det.info["trained_on"] = "tier 4 + drone/music mixtures (SNR -10..+10 dB) + music alone"
    return det


# --- main -----------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(12)
    live = L.LiveDetector("detector")
    enc = live.enc
    det_base = live.det
    print("building music-aware detector ...", flush=True)
    det_music = build_music_detector(enc, np.random.default_rng(5))

    m = pd.read_parquet(DATA_DIR / "manifest.parquet")
    test = m[m.split_tier4 == "val"]       # tier-1 test is tier-4 training data
    drones = recordings(test, "drone", None, rng)
    backgrounds = recordings(test, "background", 60, rng, min_slices=5)
    music_test = gtzan_files("test")
    music = music_segments(music_test, 60, 5.0, rng)
    print(f"benchmark: {len(drones)} drone recordings, {len(backgrounds)} background, "
          f"{len(music)} music excerpts", flush=True)

    # every test signal, as (condition, label, signal)
    signals = []
    for snr in SNRS:
        for y in drones:
            if snr is None:
                signals.append(("clean", 1, y))
            else:
                mu = music[rng.integers(len(music))]
                signals.append((f"{snr:+d} dB", 1, N.mix(y, mu[: len(y)], snr)))
    for mu in music:
        signals.append(("music alone", 0, mu))
    for y in backgrounds:
        signals.append(("background", 0, y))
        mu = music[rng.integers(len(music))]
        signals.append(("background + music 0 dB", 0, N.mix(y, mu[: len(y)], 0)))

    res = {}
    for front in (False, True):
        clips, cond, lab, rec = [], [], [], []
        for i, (c, yl, y) in enumerate(signals):
            z = stationary(y) if front else y
            for w in windows(z):
                clips.append(w); cond.append(c); lab.append(yl); rec.append(i)
        F = embed(enc, clips)
        cond, lab, rec = np.array(cond), np.array(lab), np.array(rec)
        for name, det in (("original detector", det_base), ("music-aware detector", det_music)):
            h = hits(det, F)
            key = f"{name}{' + stationary front-end' if front else ''}"
            rows = {}
            for c in pd.unique(cond):
                sel = cond == c
                r_hit = pd.Series(h[sel]).groupby(rec[sel]).any()
                rows[c] = {"window_rate": float(h[sel].mean()), "recording_rate": float(r_hit.mean()),
                           "is_drone": bool(lab[sel][0])}
            res[key] = rows
            print(key, {c: round(v["window_rate"], 3) for c, v in rows.items()}, flush=True)

    base = res["original detector"]
    fe = res["original detector + stationary front-end"]
    gain = lambda c: fe[c]["window_rate"] - base[c]["window_rate"]
    use_front = (gain("+0 dB") >= 0.10 and gain("-5 dB") >= 0.10
                 and fe["music alone"]["window_rate"] <= 0.02
                 and base["clean"]["window_rate"] - fe["clean"]["window_rate"] <= 0.05)
    rep = {"snrs_db": [s for s in SNRS if s is not None], "results": res,
           "deploy_front_end": bool(use_front),
           "n": {"drone_recordings": len(drones), "background_recordings": len(backgrounds),
                 "music_excerpts": len(music)}}
    (REPORTS / "masking.json").write_text(json.dumps(rep, indent=2))
    plot(res)
    (REPORTS / "masking.md").write_text(markdown(rep))
    print(markdown(rep))
    print(f"total {time.time() - t0:.0f}s")
    return 0


def plot(res):
    order = ["clean", "+10 dB", "+5 dB", "+0 dB", "-5 dB", "-10 dB"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    styles = {"original detector": ("#8a8984", "-"),
              "original detector + stationary front-end": ("#2a78d6", "-"),
              "music-aware detector": ("#8a8984", "--"),
              "music-aware detector + stationary front-end": ("#d6452a", "-")}
    for k, (col, ls) in styles.items():
        ax.plot(order, [res[k][c]["window_rate"] for c in order], ls, color=col, marker="o", ms=4, lw=1.8, label=k)
    ax.set_ylabel("drone windows caught")
    ax.set_xlabel("drone over music (SNR)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e4e3df", lw=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)


def markdown(rep) -> str:
    res, n = rep["results"], rep["n"]
    order = ["clean", "+10 dB", "+5 dB", "+0 dB", "-5 dB", "-10 dB"]
    neg = ["music alone", "background", "background + music 0 dB"]
    md = ["# Music masking: detecting a drone under loud music", "",
          f"Indoor drone recordings from the tier-4 validation split ({n['drone_recordings']}; the detector's "
          "weights never saw them, its threshold did; the tier-1 test split is tier-4 training data) "
          f"mixed with held-out GTZAN music "
          f"({n['music_excerpts']} excerpts) at falling SNR (drone power over music power; "
          f"-10 dB = music ten times stronger). Negatives: music alone, {n['background_recordings']} "
          "background recordings, and background + music. 1 s windows, each detector at its own "
          "threshold (1% false alarms on its validation negatives). Generated by "
          "`scripts/eval_masking.py`; figure `figures/12_masking.png`.", "",
          "- **stationary front-end**: keeps sound that stays at one frequency for seconds "
          "(rotor tones), drops what changes (music), then runs the unchanged detector "
          "(`src/dronedet/features/unmask.py`).",
          "- **music-aware detector**: the tier-4 detector rebuilt with drone + music mixtures "
          "(-10 to +10 dB) and music alone in training; train/val music never overlaps the test music.", "",
          "## Drone windows caught", "",
          "| method | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for k, rows in res.items():
        md.append(f"| {k} | " + " | ".join(f"{rows[c]['window_rate']:.2f}" for c in order) + " |")
    md += ["", "## False alarms (share of windows called drone)", "",
           "| method | " + " | ".join(neg) + " |", "|---|" + "---|" * len(neg)]
    for k, rows in res.items():
        md.append(f"| {k} | " + " | ".join(f"{rows[c]['window_rate']:.3f}" for c in neg) + " |")
    md += ["", "## Recordings with at least one alarm (live rule: first drone window)", "",
           "| method | clean | 0 dB | -5 dB | -10 dB | music alone | background + music |",
           "|---|---|---|---|---|---|---|"]
    for k, rows in res.items():
        md.append(f"| {k} | " + " | ".join(f"{rows[c]['recording_rate']:.2f}" for c in
                                            ["clean", "+0 dB", "-5 dB", "-10 dB", "music alone",
                                             "background + music 0 dB"]) + " |")
    md += ["", f"**Live page:** the deployment rule (fixed before the run: +0.10 recall at 0 and -5 dB, "
           f"music-alone false alarms <= 0.02, clean recall cost <= 0.05) "
           f"{'is met, so the live page applies the front-end.' if rep['deploy_front_end'] else 'is not met, so the live page does not use it.'}", "",
           "## Reading the results", "",
           "- **The stationary front-end does not work on real recordings.** It removes enough of a "
           "real drone (rotor speed is not perfectly steady) that clean recall collapses, and the "
           "steady parts of music (sustained bass and pads) survive it and look drone-like, so music "
           "alone raises false alarms. The synthetic test in `tests/test_unmask.py` (a perfectly "
           "steady rotor) overstates what it can do. A detector trained on front-end output might "
           "recover some of this; that is untested.",
           "- **Training with music works.** The music-aware detector keeps clean recall close to the "
           "original and catches over twice as many drone windows at 0 and -5 dB; music alone "
           "still raises no alarms. The cost is more false alarms on background noise mixed with "
           "music. It is an experiment and is not used on the live page.",
           "- Below -5 dB (music more than three times stronger than the drone) nothing here works.", ""]
    return "\n".join(md)


if __name__ == "__main__":
    raise SystemExit(main())
