"""Synthetic receiver noise: RF windows with no transmitter at all.

DroneRF has one background session, so a model trained on it learns "not
that session" rather than "a drone is transmitting" (reports/rf_tampere.md:
pure white noise is called "drone"). Noise windows from many simulated
receivers break that shortcut: each has its own noise-floor shape, and none
contains a signal, so the only way to call them "no drone" is to look for a
transmission.

Two pipelines, because the PSD of the same noise differs between them:

- "dronerf": real samples at 40 MS/s, as DroneRF's receivers deliver them.
- "tampere": complex IQ at 120 MS/s through `rf_tampere.to_dronerf_psd`, so the
  conversion's filter roll-off is seen in negatives too and cannot become a
  drone cue (every Tampere recording is a drone).

Noise families:

- "white": flat.
- "coloured": white noise through a random smooth gain curve (up to +-12 dB
  over the band, a few broad bumps): receiver front ends and cable losses.
- "pink": 1/f, never used for training; a held-out family for testing.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin2

from dronedet.data import rf as R
from dronedet.data import rf_tampere as T

FAMILIES = ("white", "coloured", "pink")
PIPELINES = ("dronerf", "tampere")


def _smooth_gain(rng, n_points: int = 9, max_db: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    f = np.linspace(0, 1, n_points)
    g_db = rng.uniform(-max_db, max_db, n_points)
    g_db -= g_db.mean()
    return f, 10 ** (g_db / 20)


def _shape(x: np.ndarray, family: str, rng) -> np.ndarray:
    if family == "white":
        return x
    if family == "coloured":
        f, g = _smooth_gain(rng)
        h = firwin2(129, f, g)
        return fftconvolve(x, h, mode="same")
    if family == "pink":
        X = np.fft.fft(x)
        k = np.abs(np.fft.fftfreq(len(x)))
        k[0] = k[1]
        return np.fft.ifft(X / np.sqrt(k / k.max())).real if np.isrealobj(x) \
            else np.fft.ifft(X / np.sqrt(k / k.max()))
    raise ValueError(family)


def noise_window(family: str, pipeline: str, rng: np.random.Generator,
                 cfg: R.RFConfig = R.DEFAULT) -> np.ndarray:
    """One window's PSD, (2 bands, bins), with no transmitter in it."""
    if pipeline == "dronerf":
        n = int(round(cfg.window_s * cfg.sample_rate))
        bands = []
        one = R.RFConfig(cfg.sample_rate, 1, cfg.n_fft)
        for _ in range(2):              # two independent receivers
            x = _shape(rng.normal(0, 1, n), family, rng)
            bands.append(R.segment_psd(x.astype(np.float32), one)[0][0])
        return np.stack(bands)
    if pipeline == "tampere":
        n = int(round(cfg.window_s * T.FS_IN))
        z = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        z = _shape(z, family, rng)
        z = z / np.std(z) * 300.0       # int16 range, well above quantisation
        iq = np.stack([z.real, z.imag], axis=1).round().astype("<i2")
        return T.to_dronerf_psd(iq, cfg)[0][0]
    raise ValueError(pipeline)


def noise_bank(n: int, families, pipelines, seed: int) -> tuple[np.ndarray, list[dict]]:
    """n windows per (family, pipeline), with a row of metadata each."""
    rng = np.random.default_rng(seed)
    X, rows = [], []
    for fam in families:
        for pipe in pipelines:
            for i in range(n):
                X.append(noise_window(fam, pipe, rng))
                rows.append({"family": fam, "pipeline": pipe, "index": i, "seed": seed})
    return np.stack(X).astype(np.float32), rows
