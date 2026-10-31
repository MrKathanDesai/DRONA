from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_audio_baseline import best_f1_threshold, extract_features


def test_extract_features_returns_finite_fixed_size_vector(tmp_path):
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = 0.5 * np.sin(2 * np.pi * 440 * time)
    path = tmp_path / "tone.wav"
    sf.write(path, waveform, sample_rate)

    features = extract_features(path)

    assert features.shape == (128,)
    assert np.isfinite(features).all()


def test_best_f1_threshold_uses_validation_probabilities():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = best_f1_threshold(labels, probabilities)

    assert 0.4 <= threshold <= 0.6
