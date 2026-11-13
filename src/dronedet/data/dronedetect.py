"""Metadata and streaming PSD features for the documented DroneDetect IQ format."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch

INTERFERENCE = {
    "00": "clean",
    "01": "bluetooth",
    "10": "wifi",
    "11": "bluetooth_wifi",
}
MODES = {"00": "on", "01": "hovering", "10": "flying"}
NAME = re.compile(r"(?P<model_code>[A-Z0-9]+)_(?P<interference_code>\d{2})"
                  r"(?P<mode_code>\d{2})_(?P<take>\d{2})\.dat$")


@dataclass(frozen=True)
class DroneDetectConfig:
    sample_rate: int = 60_000_000
    samples_per_window: int = 1_200_000
    n_fft: int = 2048
    n_bands: int = 64

    @property
    def window_s(self) -> float:
        return self.samples_per_window / self.sample_rate


DEFAULT = DroneDetectConfig()


@dataclass(frozen=True)
class Recording:
    path: Path
    model_code: str
    interference_code: str
    mode_code: str
    take: str

    @property
    def interference(self) -> str:
        return INTERFERENCE[self.interference_code]

    @property
    def mode(self) -> str:
        return MODES[self.mode_code]


def parse_recording(path: Path | str) -> Recording:
    path = Path(path)
    match = NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected DroneDetect filename: {path.name}")
    fields = match.groupdict()
    if fields["interference_code"] not in INTERFERENCE:
        raise ValueError(f"unknown interference code in {path.name}")
    if fields["mode_code"] not in MODES:
        raise ValueError(f"unknown flight-mode code in {path.name}")
    return Recording(path=path, **fields)


def find_recordings(root: Path | str) -> list[Recording]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DroneDetect folder not found: {root}")
    paths = sorted(root.glob("*.dat"))
    if not paths:
        raise FileNotFoundError(f"No .dat recordings found in {root}")
    return [parse_recording(path) for path in paths]


def psd_windows(
    path: Path | str,
    config: DroneDetectConfig = DEFAULT,
) -> tuple[np.ndarray, int]:
    """Return normalized two-sided log-PSD for each complete 20 ms IQ window."""
    path = Path(path)
    bytes_per_sample = np.dtype(np.complex64).itemsize
    size = path.stat().st_size
    if size % bytes_per_sample:
        raise ValueError(f"{path} size is not a whole number of complex64 samples")

    sample_count = size // bytes_per_sample
    n_windows = sample_count // config.samples_per_window
    if not n_windows:
        raise ValueError(f"{path} has no complete {config.window_s:g} s window")

    iq = np.memmap(path, dtype=np.complex64, mode="r")
    features = np.empty((n_windows, config.n_fft), dtype=np.float32)
    try:
        for index in range(n_windows):
            start = index * config.samples_per_window
            window = iq[start:start + config.samples_per_window]
            if not np.isfinite(window).all():
                raise ValueError(f"{path} contains non-finite IQ values")
            _, power = welch(
                window,
                fs=config.sample_rate,
                nperseg=config.n_fft,
                noverlap=0,
                detrend=False,
                return_onesided=False,
            )
            features[index] = 10 * np.log10(np.maximum(power, 1e-20))
    finally:
        del iq

    features -= features.mean(axis=1, keepdims=True)
    tail_samples = sample_count - n_windows * config.samples_per_window
    return features, tail_samples
