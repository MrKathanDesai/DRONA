from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# pyrefly: ignore [missing-import]
from dronedet.features import noise as N


def tone(f=300.0, n=16000):
    return (0.2 * np.sin(2 * np.pi * f * np.arange(n) / 16000)).astype(np.float32)


@pytest.mark.parametrize("snr", [20.0, 10.0, 0.0, -5.0])
def test_mix_hits_requested_snr(snr):
    clean = tone()
    noise = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    added = N.mix(clean, noise, snr) - clean
    got = 20 * np.log10(N.rms(clean) / N.rms(added))
    assert got == pytest.approx(snr, abs=0.01)


def test_mix_leaves_silent_input_alone():
    z = np.zeros(16000, np.float32)
    np.testing.assert_array_equal(N.mix(z, tone(), 0.0), z)
    np.testing.assert_array_equal(N.mix(tone(), z, 0.0), tone())


def test_synth_music_is_finite_tonal_and_seeded():
    a = N.synth_music(np.random.default_rng(1))
    b = N.synth_music(np.random.default_rng(1))
    assert a.shape == (16000,) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    spec = np.abs(np.fft.rfft(a))
    assert spec.max() > 20 * np.median(spec)       # peaky, not white


def make_bank():
    rng = np.random.default_rng(0)
    clips, cats, groups = [], [], []
    for cat in ("laughing", "engine", "wind", "rain", "speech", "music_real"):
        for g in range(3):
            clips.append(rng.standard_normal(16000).astype(np.float32) * 0.1)
            cats.append(cat)
            groups.append(f"{cat}_{g}")
    clips.append(np.zeros(16000, np.float32))           # silent ESC-50 padding
    cats.append("wind")
    groups.append("wind_silent")
    return N.NoiseBank(clips, cats, groups)


def test_bank_drops_silent_slices():
    bank = make_bank()
    assert all(N.rms(c) >= N.MIN_RMS for c, _ in bank.by_cond["wind"])
    assert len(bank.by_cond["wind"]) == 3


def test_bank_avoids_own_recording():
    bank = make_bank()
    rng = np.random.default_rng(0)
    own = bank.by_cond["voice"][0]
    for _ in range(50):
        assert bank.sample("voice", rng, avoid_group=own[1]) is not own[0]


def test_every_condition_corrupts():
    bank, rng = make_bank(), np.random.default_rng(0)
    for cond in N.CONDITIONS:
        out = bank.corrupt(tone(), cond, 0.0, rng)
        assert out.shape == (16000,) and np.isfinite(out).all()
        assert N.rms(out - tone()) > 0.5 * N.rms(tone())


def test_bank_requires_each_real_condition():
    with pytest.raises(ValueError):
        N.NoiseBank([tone()], ["laughing"], ["g"])
