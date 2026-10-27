from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
# pyrefly: ignore [missing-import]
from dronedet.models import pretrained as PT

needs_checkpoint = pytest.mark.skipif(
    not PT.CHECKPOINT.exists(),
    reason="run scripts/download_data.py first")


@pytest.fixture(scope="module")
def enc():
    return PT.Encoder(device="cpu")


def noisy_tone(freq, seed=0, sr=16000):
    t = np.arange(sr) / sr
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * freq * t)
            + 0.05 * rng.standard_normal(sr)).astype(np.float32)


@needs_checkpoint
def test_checkpoint_loads_every_layer(enc):
    """Encoder() raises on any missing or unexpected backbone weight, so
    constructing it is the test; check the shapes on top."""
    e, t = enc([noisy_tone(440)])
    assert e.shape == (1, PT.EMBED_DIM) and t.shape == (1, 527)


@needs_checkpoint
def test_embedding_ignores_overall_level(enc):
    """Loudness separates the classes at 0.79 in this corpus; gain and
    distance must not reach the head."""
    y = noisy_tone(300)
    e, _ = enc([y, 0.1 * y, 3.0 * y])
    np.testing.assert_allclose(e[1], e[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(e[2], e[0], rtol=1e-3, atol=1e-3)


@needs_checkpoint
def test_silence_is_finite(enc):
    e, t = enc([np.zeros(16000, dtype=np.float32)])
    assert np.isfinite(e).all() and np.isfinite(t).all()


@needs_checkpoint
def test_different_sounds_give_different_embeddings(enc):
    e, _ = enc([noisy_tone(200), noisy_tone(200, seed=1), noisy_tone(3000)])
    same = np.linalg.norm(e[0] - e[1])
    diff = np.linalg.norm(e[0] - e[2])
    assert diff > 2 * same


def test_detect_requires_consecutive_hits():
    import detect
    assert detect.longest_run(np.array([1, 0, 1, 0, 1], bool)) == 1
    assert detect.longest_run(np.array([0, 1, 1, 1, 0], bool)) == 3


def test_detect_windows_cover_the_file():
    import detect
    clips, starts = detect.windows(np.zeros(16000 * 3, np.float32), 16000, 0.5)
    assert starts == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert all(len(c) == 16000 for c in clips)
