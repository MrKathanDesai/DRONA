from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_audio_cnn import AudioCNN, extract_log_mel


def test_cnn_returns_one_logit_per_spectrogram():
    model = AudioCNN()
    inputs = torch.zeros((2, 1, 64, 101))

    logits = model(inputs)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_log_mel_feature_has_fixed_shape_for_short_and_long_clips(tmp_path):
    sample_rate = 16_000
    for name, duration in (("short.wav", 0.5), ("long.wav", 1.5)):
        time = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
        path = tmp_path / name
        sf.write(path, 0.25 * np.sin(2 * np.pi * 440 * time), sample_rate)

        features = extract_log_mel(path)

        assert features.shape == (64, 101)
        assert np.isfinite(features).all()
