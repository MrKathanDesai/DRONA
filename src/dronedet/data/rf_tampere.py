"""Tampere drone RF recordings (Pärlin, Marin et al., Zenodo 4264467, CC-BY 4.0).

Complex IQ, interleaved little-endian int16, recorded in an anechoic chamber.
The 2.4 GHz files are centred on 2.44 GHz at 120 MS/s, so each spans
2.38-2.50 GHz and contains both DroneRF bands:

    DroneRF low band   2.400-2.440 GHz  ->  centre 2.420 GHz, offset -20 MHz
    DroneRF high band  2.440-2.480 GHz  ->  centre 2.460 GHz, offset +20 MHz

`to_dronerf_psd` mixes each band to baseband, low-pass filters and decimates by
3 to 40 MS/s, keeps the real part, and runs the DroneRF PSD
(`rf.segment_psd`) on 25 ms windows. Keeping only the real part assumes
DroneRF's single real channel is the in-phase output of a complex
downconverter; that is an assumption, since the DroneRF paper does not say so.

There are no drone-off recordings, so this corpus gives recall and
score behaviour on a new receiver and session, never a false-alarm rate.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from dronedet.data import rf as R

FS_IN = 120_000_000
DECIMATE = 3                                  # 120 -> 40 MS/s, DroneRF's rate
BAND_OFFSETS_HZ = (-20_000_000, 20_000_000)   # low band, high band
FILE_NAME = re.compile(r"^(.+?)_2G(?:_(\d)of\d)?\.bin$")

# Tampere file stem -> manufacturer, for reporting.
MAKERS = {"dji": "DJI", "parrot": "Parrot", "yuneec": "Yuneec"}


def parse_name(path: str | Path) -> tuple[str, int]:
    """'DJI_inspire_2_2G.bin' -> ('DJI_inspire_2', 1); '..._2G_2of2.bin' -> part 2."""
    m = FILE_NAME.match(Path(path).name)
    if not m:
        raise ValueError(f"not a 2.4 GHz Tampere file: {path}")
    return m[1], int(m[2] or 1)


def load_iq(path: str | Path) -> np.ndarray:
    """Memory-mapped complex view of one recording (no copy)."""
    raw = np.memmap(path, dtype="<i2", mode="r")
    if len(raw) % 2:
        raise ValueError(f"{path}: odd number of int16 values")
    return raw.reshape(-1, 2)


def to_dronerf_psd(iq: np.ndarray, cfg: R.RFConfig = R.DEFAULT,
                   fs_in: int = FS_IN) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) int16 IQ -> PSD (windows, 2 bands, bins) and level (windows, 2).

    Windows are as long as DroneRF's (cfg.window_s); a trailing partial window
    is dropped."""
    if fs_in != cfg.sample_rate * DECIMATE:
        raise ValueError("input rate must be 3x the DroneRF rate")
    per_win = int(round(cfg.window_s * fs_in))
    n_win = len(iq) // per_win
    n_out = per_win // DECIMATE
    psd = np.empty((n_win, 2, cfg.n_bins), dtype=np.float32)
    level = np.empty((n_win, 2), dtype=np.float32)
    phase_step = [-2j * np.pi * f / fs_in for f in BAND_OFFSETS_HZ]
    for w in range(n_win):
        a = w * per_win
        x = iq[a:a + per_win].astype(np.float64)
        z = x[:, 0] + 1j * x[:, 1]
        n = np.arange(a, a + per_win)       # absolute index keeps phase continuous
        for b, step in enumerate(phase_step):
            y = resample_poly(z * np.exp(step * n), 1, DECIMATE)[:n_out].real
            # One DroneRF segment's worth of windows is 10; here one window
            # at a time, so ask segment_psd for a single window.
            p, lv = R.segment_psd(y.astype(np.float32),
                                  R.RFConfig(cfg.sample_rate, 1, cfg.n_fft))
            psd[w, b], level[w, b] = p[0], lv[0]
    return psd, level
