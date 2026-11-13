"""Run the audio detector over a whole recording, window by window.

Shared by scripts/detect.py and the Streamlit dashboard so both show the
same numbers. Uses the tier-4 detector with uncertainty flags
(data/models/detector.joblib, from scripts/eval_uncertainty.py) when it
exists; otherwise falls back to a plain head, which gives only "drone" /
"no drone".
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
                               ("p", "state", "reason", "margin", "distance", "spread")})
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
