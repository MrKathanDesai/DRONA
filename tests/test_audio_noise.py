from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_audio_noise import add_noise


@pytest.mark.parametrize("noise_type", ["white", "pink"])
def test_add_noise_hits_requested_snr_and_is_reproducible(noise_type):
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.1, 16_000).astype(np.float32)

    first, metadata = add_noise(signal, 10, noise_type, 123, 0.1)
    second, second_metadata = add_noise(signal, 10, noise_type, 123, 0.1)

    assert np.array_equal(first, second)
    assert metadata == second_metadata
    assert metadata["achieved_snr_db"] == pytest.approx(10, abs=1e-6)
    assert not metadata["used_training_drone_reference_for_silence"]


def test_silent_clip_uses_explicit_training_reference():
    silence = np.zeros(16_000, dtype=np.float32)

    noisy, metadata = add_noise(silence, 20, "white", 42, 0.05)

    assert np.isfinite(noisy).all()
    assert np.any(noisy != 0)
    assert metadata["used_training_drone_reference_for_silence"]
    assert metadata["achieved_snr_db"] is None
    assert metadata["noise_rms"] == pytest.approx(0.005)


def test_add_noise_rejects_invalid_silence_reference():
    with pytest.raises(ValueError, match="Silence reference RMS"):
        add_noise(np.ones(100, dtype=np.float32), 10, "white", 42, 0)
