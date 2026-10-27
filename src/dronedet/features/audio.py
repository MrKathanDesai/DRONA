from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_fft: int = 1024          # 64 ms: resolves rotor harmonics ~16 Hz apart
    hop_length: int = 256      # 16 ms -> 63 frames per second
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        return 1 + self.clip_samples // self.hop_length

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


DEFAULT = LogMelConfig()


def load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != sample_rate:
        g = gcd(sr, sample_rate)
        y = resample_poly(y, sample_rate // g, sr // g).astype(np.float32)
    return y


def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    """Centre-crop long clips; loop short ones.

    Never zero-pad. In DroneAudioDataset each clip duration belongs to a
    single class (drone 1.024 s, background 1.0217 / 0.975 / 0.960 s ...),
    so a silent tail of class-dependent length would hand the model the
    label.
    """
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    return np.resize(y, n)


def log_mel(y: np.ndarray, cfg: LogMelConfig = DEFAULT) -> np.ndarray:
    """(n_mels, n_frames) log-power mel spectrogram in dB, unnormalised.

    Absolute level is kept on purpose: the simulator scales clips by distance,
    so normalisation is a model decision, not a feature decision.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0)
    return (10.0 * np.log10(S + 1e-10)).astype(np.float32)


def load_clip(path: str | Path, start_s: float | None = None,
              cfg: LogMelConfig = DEFAULT) -> np.ndarray:
    """One fixed-length mono clip at cfg.sample_rate. `start_s` selects a
    window inside a long recording; None (NaN in the manifest) takes the
    whole clip and centre-crops it."""
    y = load_mono(path, cfg.sample_rate)
    if start_s is not None and not np.isnan(start_s):
        a = int(round(start_s * cfg.sample_rate))
        y = y[a:a + cfg.clip_samples]
    return fix_length(y, cfg.clip_samples)


def clip_features(path: str | Path, cfg: LogMelConfig = DEFAULT,
                  start_s: float | None = None) -> np.ndarray:
    return log_mel(load_clip(path, start_s, cfg), cfg)


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-mel-band standardisation; fit mean/std on the train split only."""
    return (X - mean[:, None]) / (std[:, None] + 1e-6)


def band_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=(0, 2)), X.std(axis=(0, 2))
