"""Run the audio detector over a whole recording, window by window, and
optionally fuse it with an RF feed.

Shared by scripts/detect.py and the Streamlit dashboard so both show the
same numbers. Uses the tier-4 detector with uncertainty flags
(data/models/detector.joblib, from scripts/eval_uncertainty.py) when it
exists; otherwise falls back to a plain head, which gives only "drone" /
"no drone".

RF has no live receiver here either: `RFFeed` replays held-out RF windows of a
chosen class (DroneRF, Tampere drones never trained on, or receiver noise),
one 25 ms snapshot per audio window, and `fuse` combines the two with
data/models/fusion.joblib.
Audio and RF were never recorded together, so a fused replay is a
what-if, not a recording.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd

from dronedet.features import audio as A
from dronedet.models import explain as X
from dronedet.models import pretrained as PT
from dronedet.paths import DATA_DIR

MODELS = DATA_DIR / "models"
DETECTOR = MODELS / "detector.joblib"
HEADS = {"tier4": MODELS / "outdoor_head.joblib",
         "tier3": MODELS / "pretrained_head.joblib"}
SR = A.DEFAULT.sample_rate
AUDIO_FLAGS = ("low_margin", "unfamiliar", "far_from_drones")


def windows(y: np.ndarray, sr: int, hop_s: float) -> tuple[list, list]:
    n, hop = sr, int(hop_s * sr)
    starts = list(range(0, max(len(y) - n, 0) + 1, hop))
    return [A.fix_length(y[s:s + n], n) for s in starts], [s / sr for s in starts]


def longest_run(flags) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


class LiveDetector:
    def __init__(self, head: str = "detector", device: str | None = None):
        if head == "detector" and DETECTOR.exists():
            self.det = joblib.load(DETECTOR)
            self.model, self.threshold = self.det.head, self.det.threshold
            self.name = "tier 4 + uncertainty flags"
        else:
            path = HEADS["tier4" if head == "detector" else head]
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run the training scripts first")
            h = joblib.load(path)
            self.det, self.model, self.threshold = None, h["model"], h["threshold"]
            self.name = f"{head} head (no uncertainty flags)"
        self.enc = PT.Encoder(device=device)
        self.labels = PT.audioset_labels()
        self.w, self.b = X.linear_head(self.model)

    def analyse(self, y: np.ndarray, hop_s: float = 0.5) -> tuple[pd.DataFrame, list]:
        """16 kHz mono signal -> one row per 1 s window, plus the clips."""
        clips, starts = windows(y, SR, hop_s)
        emb, tags = self.enc(clips)
        if self.det is not None:
            a = self.det.assess(emb)
            df = pd.DataFrame({k: a[k] for k in
                               ("p", "state", "reason", "margin", "distance", "spread")
                               if k in a})
            for k in AUDIO_FLAGS:          # kept per window for fusion
                df[k] = a[k]
        else:
            p = self.model.predict_proba(emb)[:, 1]
            df = pd.DataFrame({"p": p, "state": np.where(p >= self.threshold,
                                                         "drone", "no drone"),
                               "reason": ""})
        df.insert(0, "t", starts)
        df["hit"] = df.p >= self.threshold
        df["tags"] = [" | ".join(self.labels[i] for i in np.argsort(t)[::-1][:3])
                      for t in tags]
        return df, clips

    def explain(self, clip: np.ndarray, layer: int = 11):
        return X.gradcam(self.enc, clip, self.w, self.b, layer)


def load(path: str | Path | BinaryIO) -> np.ndarray:
    return A.load_mono(path, SR)


RF_DETECTOR = MODELS / "rf_detector.joblib"
FUSION = MODELS / "fusion.joblib"
FUSION_MODELS = MODELS / "fusion_models.joblib"
RF_V2 = DATA_DIR / "rf_features_v2"


class RFFeed:
    """Held-out RF windows to replay beside an audio recording: the test split
    of the RF v2 dataset (scripts/train_rf_v2.py), scored by the deployed RF
    detector with the features it was trained on."""

    def __init__(self):
        from dronedet.models import rf as M
        for p in (RF_DETECTOR, RF_V2 / "X.npy"):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing; run scripts/prepare_rf.py, "
                                        "eval_rf_tampere.py and train_rf_v2.py")
        if FUSION_MODELS.exists():
            saved_fusions = joblib.load(FUSION_MODELS)
            self.fusions = saved_fusions["methods"]
        elif FUSION.exists():
            self.fusions = {"sum": joblib.load(FUSION)}
        else:
            raise FileNotFoundError(
                f"{FUSION_MODELS} missing; run scripts/eval_fusion.py"
            )
        if not self.fusions:
            raise ValueError(f"No fusion models found in {FUSION_MODELS}")
        self.det = joblib.load(RF_DETECTOR)
        self.fusion = self.fusions.get("sum", next(iter(self.fusions.values())))
        self.features = M.MODELS[self.det.info.get("features", M.DEPLOY)][0]
        X = np.load(RF_V2 / "X.npy")
        df = pd.read_parquet(RF_V2 / "index.parquet")
        te = df[df.split == "test"]
        self.sources: dict[str, np.ndarray] = {}
        for (src, drone, cat), sub in te.groupby(["source", "drone", "category"]):
            if src == "dronerf":
                label = (f"DroneRF drone: {drone}" if drone
                         else "DroneRF background (no drone)")
            elif src == "tampere":
                label = f"Tampere drone, never trained on: {drone}"
            else:
                label = f"Receiver noise, no drone: {cat.removeprefix('noise_')} " \
                        f"({src.removeprefix('noise_')} pipeline)"
            self.sources[label] = X[sub.index.values]

    def windows(self, source: str, n: int) -> tuple[np.ndarray, dict]:
        """n RF windows from `source`, in time order, wrapping around; returns
        their PSDs (n, 2, bins) and the RF detector's assessment."""
        X = self.sources[source]
        psd = X[np.arange(n) % len(X)]
        return psd, self.det.assess(self.features(psd))


def fuse(audio_df: pd.DataFrame, fusion, rf_p: np.ndarray | None,
         rf_flags: dict | None) -> tuple[pd.DataFrame, float]:
    """Fused state per audio window, and the fused threshold in use.
    `rf_p=None` means no RF feed (fusion then uses audio evidence alone)."""
    n = len(audio_df)
    audio_flags = ({k: audio_df[k].values.astype(bool) for k in AUDIO_FLAGS}
                   if all(k in audio_df for k in AUDIO_FLAGS) else None)
    out = fusion.assess({"audio": audio_df.p.values, "rf": rf_p}, n,
                        {"audio": audio_flags, "rf": rf_flags})
    return pd.DataFrame({"fused": out["fused"], "llr_audio": out["llr_audio"],
                         "llr_rf": out["llr_rf"], "fused_state": out["state"],
                         "fused_reason": out["reason"],
                         "fused_hit": out["hit"]}), float(out["threshold"])
