"""Stationary-component extraction: keep steady rotor tones, drop music.

A multirotor's sound is dominated by its blade-pass frequency and harmonics,
which stay put for seconds while the drone holds its throttle. Music changes:
notes and chords last a fraction of a second, drums are transient. A median
taken along time over a long horizon (default 2 s) per frequency therefore
keeps the rotor tones and rejects most of the music, which occupies any one
frequency for less than half of the horizon.

A narrow median across frequency (3 bins, ~47 Hz) tolerates slow rotor-speed
drift. The stationary estimate becomes a soft (Wiener-style) mask on the
complex STFT, and the masked signal is resynthesised, so the unchanged
detector can run on it.

This is the harmonic half of harmonic/percussive separation (Fitzgerald 2010)
with a much longer time kernel: HPSS keeps sustained musical notes as
"harmonic"; a horizon of seconds does not.

Result: on real recordings it does not help detection (reports/masking.md):
real rotors drift enough to be partly removed, and sustained parts of music
survive. Kept as a documented negative result.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

SR = 16000
N_FFT = 1024
HOP = 256


def stationary(y: np.ndarray, sr: int = SR, horizon_s: float = 2.0,
               freq_bins: int = 3, power: float = 2.0) -> np.ndarray:
    """Audio -> its stationary part, same length. Needs > horizon_s of context
    to do anything useful; shorter input is mirrored at the edges."""
    _, _, Z = stft(y, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
    M = np.abs(Z)
    k = max(3, int(round(horizon_s * sr / HOP)) | 1)          # odd number of frames
    H = median_filter(M, size=(freq_bins, k), mode="reflect")
    R = np.maximum(M - H, 0.0)
    Hp, Rp = H ** power, R ** power
    mask = Hp / (Hp + Rp + 1e-12)
    _, out = istft(Z * mask, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)
    return out[: len(y)].astype(np.float32)
