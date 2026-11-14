from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.data.dronedetect import DroneDetectConfig, parse_recording, psd_windows


def test_parse_documented_filename_codes():
    recording = parse_recording("MA1_0100_04.dat")

    assert recording.model_code == "MA1"
    assert recording.interference == "bluetooth"
    assert recording.mode == "on"
    assert recording.take == "04"


def test_parse_rejects_unsupported_codes():
    with pytest.raises(ValueError, match="unknown interference"):
        parse_recording("MA1_0200_04.dat")
    with pytest.raises(ValueError, match="unexpected DroneDetect filename"):
        parse_recording("not-a-recording.dat")


def test_psd_windows_reads_complex_iq_and_drops_incomplete_tail(tmp_path):
    config = DroneDetectConfig(sample_rate=64, samples_per_window=16, n_fft=8)
    t = np.arange(35) / config.sample_rate
    signal = np.exp(2j * np.pi * 8 * t).astype(np.complex64)
    path = tmp_path / "MA1_0100_00.dat"
    signal.tofile(path)

    features, tail_samples = psd_windows(path, config)

    assert features.shape == (2, config.n_fft)
    assert tail_samples == 3
    assert np.isfinite(features).all()


def test_psd_windows_rejects_non_complex64_file_size(tmp_path):
    path = tmp_path / "MA1_0100_00.dat"
    path.write_bytes(b"not a whole complex sample")

    with pytest.raises(ValueError, match="whole number of complex64"):
        psd_windows(path, DroneDetectConfig(samples_per_window=1, n_fft=2))
