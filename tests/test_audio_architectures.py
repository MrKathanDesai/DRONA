from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dronedet.models.audio_architectures import (  # noqa: E402
    CNNRecurrent,
    LightweightTransformer,
    MODEL_FACTORIES,
)
from compare_audio_architectures import FeatureDataset  # noqa: E402


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_architecture_returns_one_logit_per_clip(name):
    model = MODEL_FACTORIES[name]()
    output = model(torch.randn(2, 64, 21))
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_cnn_recurrent_propagates_gradients(cell):
    model = CNNRecurrent(cell)
    output = model(torch.randn(2, 64, 21))
    output.sum().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_cnn_recurrent_rejects_unsupported_cell():
    with pytest.raises(ValueError, match="unsupported recurrent layer"):
        CNNRecurrent("rnn")


def test_lightweight_transformer_rejects_too_many_frames():
    model = LightweightTransformer(n_frames=10)
    with pytest.raises(ValueError, match="maximum is 10"):
        model(torch.randn(1, 64, 11))


def test_cached_features_pool_to_shared_64_by_21_input():
    features = torch.arange(128 * 63, dtype=torch.float32).numpy().reshape(1, 128, 63)
    sample = FeatureDataset(
        features,
        labels=[1.0],
        indices=[0],
        mean=torch.zeros(128).numpy(),
        std=torch.ones(128).numpy(),
    )[0][0]
    assert sample.shape == (64, 21)
    assert torch.isfinite(sample).all()
