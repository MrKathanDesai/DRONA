"""Frozen AudioSet-pretrained encoder: PANNs MobileNetV1 (Kong et al. 2020).

4.2 M parameters, 24 MB checkpoint, no training here: each 1 s clip becomes
a 1024-d embedding and only a small sklearn head is fitted on top. The
embedding is the pooled conv output, before `fc1`: `fc1` is specialised to
AudioSet's 527 tags, and on the unseen-drone split (tier 2) the pooled
features transfer clearly better (recall 0.65 vs 0.41 at 1% FPR). Weights:
https://zenodo.org/records/3987831 (`MobileNetV1_mAP=0.389.pth`).

The front end (STFT -> mel -> dB) is done in numpy with the checkpoint's own
mel filterbank, so the torchlibrosa dependency of the original code is not
needed. Clips are RMS-normalised first: loudness alone separates the classes
in this corpus (0.79), so the encoder must not see it.
"""
from __future__ import annotations

import csv
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch import nn

from dronedet.paths import DATA_DIR

CHECKPOINT = DATA_DIR / "pretrained" / "MobileNetV1_mAP=0.389.pth"
LABELS_CSV = DATA_DIR / "pretrained" / "class_labels_indices.csv"
SAMPLE_RATE = 32000      # what the encoder was trained on
N_FFT, HOP = 1024, 320
TARGET_RMS = 0.05        # ~ -26 dBFS
EMBED_DIM = 1024


def _conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.AvgPool2d(stride),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def _conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.AvgPool2d(stride),
        nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class MobileNetV1(nn.Module):
    """Layer layout identical to PANNs, minus the torchlibrosa front end."""

    def __init__(self, mel_bins=64, classes_num=527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        widths = [(1, 32, 2), (32, 64, 1), (64, 128, 2), (128, 128, 1),
                  (128, 256, 2), (256, 256, 1), (256, 512, 2)] \
            + [(512, 512, 1)] * 5 + [(512, 1024, 2), (1024, 1024, 1)]
        self.features = nn.Sequential(
            _conv_bn(*widths[0]), *[_conv_dw(*w) for w in widths[1:]])
        self.fc1 = nn.Linear(1024, 1024)
        self.fc_audioset = nn.Linear(1024, classes_num)

    def forward(self, logmel):
        """logmel (B, 1, T, mels) -> (pooled embedding (B, 1024), tags (B, 527))."""
        x = self.bn0(logmel.transpose(1, 3)).transpose(1, 3)
        x = self.features(x).mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        tags = torch.sigmoid(self.fc_audioset(F.relu(self.fc1(x))))
        return x, tags


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder:
    def __init__(self, checkpoint: Path = CHECKPOINT, device: str | None = None):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"{checkpoint} missing; run scripts/download_data.py")
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        self.mel_w = sd["logmel_extractor.melW"].numpy()          # (513, 64)
        self.device = device or pick_device()
        self.net = MobileNetV1()
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        stray = [k for k in unexpected
                 if not k.startswith(("spectrogram_extractor", "logmel_extractor"))]
        if missing or stray:
            raise RuntimeError(f"checkpoint mismatch: {missing} {stray}")
        self.net.eval().to(self.device)

    def logmel(self, y16k: np.ndarray) -> np.ndarray:
        """1-D 16 kHz clip -> (T, 64) dB at the encoder's 32 kHz front end.
        Everything is band-limited to 8 kHz first, so both corpora look the
        same to the encoder whatever their native rate."""
        rms = float(np.sqrt(np.mean(y16k ** 2)))
        y = y16k * (TARGET_RMS / rms) if rms > 1e-8 else y16k
        y = resample_poly(y, 2, 1).astype(np.float32)
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann",
                                center=True, pad_mode="reflect")) ** 2
        return (10.0 * np.log10(np.maximum(S.T @ self.mel_w, 1e-10))).astype(np.float32)

    @torch.no_grad()
    def __call__(self, clips: list[np.ndarray], batch: int = 128):
        """List of equal-length 16 kHz clips -> (embeddings, AudioSet tags)."""
        embs, tags = [], []
        for i in range(0, len(clips), batch):
            M = np.stack([self.logmel(c) for c in clips[i:i + batch]])
            e, t = self.net(torch.from_numpy(M[:, None]).to(self.device))
            embs.append(e.cpu().numpy())
            tags.append(t.cpu().numpy())
        return np.concatenate(embs), np.concatenate(tags)


def audioset_labels(path: Path = LABELS_CSV) -> list[str]:
    with open(path) as fh:
        return [r["display_name"] for r in csv.DictReader(fh)]


HEAD_C = 0.01   # picked from a small grid; see scripts/train_pretrained.py


def head(balanced: bool = True):
    """The trained part of the detector: standardise, then logistic regression.
    `balanced=False` is for tier 4, whose weights come from
    `protocol.source_weights` (pass `logisticregression__sample_weight`)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=HEAD_C, class_weight="balanced" if balanced else None, max_iter=5000))
