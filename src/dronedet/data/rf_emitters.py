"""Synthetic non-drone 2.4 GHz emitters: Wi-Fi, Bluetooth, microwave oven.

DroneRF's only real "no drone" recording is one lab session, so a detector has
never seen the everyday 2.4 GHz traffic it would meet in the field. These are
simple, physically motivated stand-ins (not standard-compliant waveforms):

- "wifi": 802.11 b/g/n on channel 1, 6 or 11 (2412 / 2437 / 2462 MHz), a
  ~17 MHz wide OFDM-like noise burst, sent in packets of 0.1-2 ms with gaps.
- "bluetooth": classic Bluetooth, 1 MHz FSK tone hopping over 79 channels
  (2402-2480 MHz) every 625 us slot, active in a random share of slots.
- "microwave": a magnetron near 2450 MHz, its frequency wobbling with the
  50 Hz mains and switched on for half of each mains cycle.

Each is placed on receiver noise at a random SNR and run through the same two
pipelines as `rf_noise` (DroneRF's real 40 MS/s bands, and the Tampere 120 MS/s
IQ conversion), so the conversion cannot become a cue.

Caveat: DroneRF's Bebop and AR drones are themselves controlled over Wi-Fi, so
"wifi" negatives are hard on purpose; some confusion there is expected and real.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from dronedet.data import rf as R
from dronedet.data import rf_tampere as T

FAMILIES = ("wifi", "bluetooth", "microwave")
CENTRE_HZ = 2.44e9
WIFI_CHANNELS_HZ = (2.412e9, 2.437e9, 2.462e9)


def emitter(family: str, fs: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Complex baseband (centred on 2.44 GHz) of one emitter, unit average power."""
    t = np.arange(n) / fs
    if family == "wifi":
        z = rng.normal(size=n) + 1j * rng.normal(size=n)
        z = lfilter(firwin(65, 8.5e6, fs=fs), 1.0, z)          # ~17 MHz wide
        z *= np.exp(2j * np.pi * (rng.choice(WIFI_CHANNELS_HZ) - CENTRE_HZ) * t)
        gate = np.zeros(n, bool)
        i = int(rng.uniform(0, 1e-3) * fs)
        while i < n:                                           # packets and gaps
            on = int(rng.uniform(0.1e-3, 2e-3) * fs)
            gate[i:i + on] = True
            i += on + int(rng.exponential(rng.uniform(0.2e-3, 3e-3)) * fs)
        z *= gate
    elif family == "bluetooth":
        slot = int(625e-6 * fs)
        duty = rng.uniform(0.3, 0.9)
        freq = np.zeros(n)
        gate = np.zeros(n, bool)
        for a in range(0, n, slot):
            ch = rng.integers(0, 79)
            freq[a:a + slot] = 2.402e9 + ch * 1e6 - CENTRE_HZ
            freq[a:a + slot] += rng.choice([-160e3, 160e3], size=len(freq[a:a + slot]))
            gate[a:a + int(366e-6 * fs)] = rng.random() < duty    # 366 us packet
        z = np.exp(2j * np.pi * np.cumsum(freq) / fs) * gate
    elif family == "microwave":
        phase0 = rng.uniform(0, 2 * np.pi)
        mains = np.sin(2 * np.pi * 50 * t + phase0)
        f_inst = 2.45e9 - CENTRE_HZ + rng.uniform(4e6, 10e6) * mains \
            + rng.normal(0, 0.5e6, n)
        z = np.exp(2j * np.pi * np.cumsum(f_inst) / fs) * (mains > 0)
    else:
        raise ValueError(family)
    p = np.mean(np.abs(z) ** 2)
    return z / np.sqrt(p) if p > 0 else z


def _scene(family, fs, n, rng, snr_db):
    """Emitter + complex receiver noise. Noise power 1, emitter at snr_db."""
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    return noise + emitter(family, fs, n, rng) * 10 ** (snr_db / 20)


def dronerf_bands(z80: np.ndarray, fs: float = 80e6) -> list[np.ndarray]:
    """80 MS/s complex scene -> DroneRF's two real 40 MS/s bands."""
    n = np.arange(len(z80))
    out = []
    for off in T.BAND_OFFSETS_HZ:
        y = resample_poly(z80 * np.exp(-2j * np.pi * off / fs * n), 1, 2)
        out.append(y.real.astype(np.float32))
    return out


def emitter_window(family: str, pipeline: str, rng: np.random.Generator,
                   cfg: R.RFConfig = R.DEFAULT, snr_db: float | None = None) -> np.ndarray:
    """One window's PSD, (2 bands, bins): a non-drone emitter on receiver noise."""
    snr_db = rng.uniform(0, 25) if snr_db is None else snr_db
    if pipeline == "dronerf":
        n = int(round(cfg.window_s * 80e6))
        bands = dronerf_bands(_scene(family, 80e6, n, rng, snr_db))
        one = R.RFConfig(cfg.sample_rate, 1, cfg.n_fft)
        return np.stack([R.segment_psd(b, one)[0][0] for b in bands])
    if pipeline == "tampere":
        n = int(round(cfg.window_s * T.FS_IN))
        z = _scene(family, T.FS_IN, n, rng, snr_db)
        z = z / np.std(z) * 300.0
        iq = np.stack([z.real, z.imag], axis=1).round().astype("<i2")
        return T.to_dronerf_psd(iq, cfg)[0][0]
    raise ValueError(pipeline)


def emitter_bank(n: int, families, pipelines, seed: int) -> tuple[np.ndarray, list[dict]]:
    """n windows per (family, pipeline), with a row of metadata each."""
    rng = np.random.default_rng(seed)
    X, rows = [], []
    for fam in families:
        for pipe in pipelines:
            for i in range(n):
                X.append(emitter_window(fam, pipe, rng))
                rows.append({"family": fam, "pipeline": pipe, "index": i, "seed": seed})
    return np.stack(X).astype(np.float32), rows


# --- time-frequency features (need raw samples, so used on Tampere only) ------

def tf_features(x: np.ndarray, fs: float = 40e6, nperseg: int = 256,
                n_sub: int = 32, max_frames: int = 2048) -> np.ndarray:
    """Timing cues the averaged PSD throws away, from one real band signal.

    Short frames (6.4 us at 40 MS/s) in n_sub sub-bands. A frame is "active"
    when its power is 6 dB over the window's quiet frames. Returns 8 numbers:
    duty cycle, mean and max burst length (frames, log), frame-power spread,
    hop rate (active frames whose strongest sub-band changed), mean and spread
    of occupied sub-bands per active frame, and share of sub-bands ever peaked.
    """
    m = min(len(x) // nperseg, max_frames)
    fr = x[: m * nperseg].reshape(m, nperseg) * np.hanning(nperseg)
    P = np.abs(np.fft.rfft(fr, axis=1))[:, 1:] ** 2
    S = np.stack([s.sum(1) for s in np.array_split(P, n_sub, axis=1)], 1)
    S_db = 10 * np.log10(S + 1e-12)
    tot = 10 * np.log10(S.sum(1) + 1e-12)
    floor = np.percentile(tot, 10)
    act = tot > floor + 6
    runs, r = [], 0
    for a in act:
        if a:
            r += 1
        elif r:
            runs.append(r)
            r = 0
    if r:
        runs.append(r)
    sub_floor = np.percentile(S_db, 10, axis=0)
    occ = (S_db > sub_floor + 6).sum(1)
    peak = S_db.argmax(1)
    hops = (np.diff(peak) != 0) & act[1:] & act[:-1]
    n_act = max(act.sum(), 1)
    return np.array([
        act.mean(),
        np.log1p(np.mean(runs)) if runs else 0.0,
        np.log1p(max(runs)) if runs else 0.0,
        np.std(tot),
        hops.sum() / max((act[1:] & act[:-1]).sum(), 1),
        occ[act].mean() / n_sub if act.any() else 0.0,
        occ[act].std() / n_sub if act.any() else 0.0,
        len(np.unique(peak[act])) / n_sub if act.any() else 0.0,
    ], dtype=np.float32) if n_act else np.zeros(8, np.float32)


def tampere_band_signals(z120: np.ndarray) -> list[np.ndarray]:
    """120 MS/s complex window -> DroneRF's two real 40 MS/s bands (as T does)."""
    n = np.arange(len(z120))
    return [resample_poly(z120 * np.exp(-2j * np.pi * off / T.FS_IN * n), 1, T.DECIMATE)
            .real.astype(np.float32) for off in T.BAND_OFFSETS_HZ]
