from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_rf_trust as evaluation  # noqa: E402
from dronedet.eval.calibration import (  # noqa: E402
    calibration_metrics,
    fit_temperature,
    temperature_scale,
)
from dronedet.eval.open_set import open_set_metrics, unknownness_scores  # noqa: E402


def test_temperature_fit_uses_only_provided_calibration_examples():
    probabilities = np.array([
        [0.85, 0.15],
        [0.20, 0.80],
        [0.70, 0.30],
        [0.10, 0.90],
    ])
    labels = np.array([0, 1, 0, 1])
    temperature = fit_temperature(probabilities, labels)
    calibrated = temperature_scale(probabilities, temperature)
    assert temperature > 0
    assert calibrated.shape == probabilities.shape
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    assert calibration_metrics(calibrated, labels)["sample_count"] == 4


def test_calibration_metrics_validate_input():
    with pytest.raises(ValueError, match="n_bins"):
        calibration_metrics(np.array([0.2, 0.8]), np.array([0, 1]), n_bins=0)
    with pytest.raises(ValueError, match="greater than zero"):
        temperature_scale(np.array([0.2, 0.8]), 0)


def test_unknownness_ignores_background_confidence():
    probabilities = np.array([
        [0.80, 0.15, 0.05],
        [0.05, 0.90, 0.05],
        [0.10, 0.10, 0.80],
    ])
    scores = unknownness_scores(probabilities, [1, 2])
    np.testing.assert_allclose(scores, [0.85, 0.10, 0.20])


def test_open_set_metrics_reports_known_accept_and_unknown_reject_rates():
    scores = np.array([0.05, 0.10, 0.70, 0.80])
    is_unknown = np.array([False, False, True, True])
    result = open_set_metrics(scores, is_unknown, acceptance_threshold=0.2)
    assert result["unknown_roc_auc"] == 1.0
    assert result["known_accept_rate"] == 1.0
    assert result["unknown_reject_rate"] == 1.0


def _tiny_forest(monkeypatch):
    monkeypatch.setattr(
        evaluation.M,
        "forest",
        lambda: RandomForestClassifier(n_estimators=12, random_state=0, n_jobs=1),
    )


def test_binary_calibration_fits_separate_validation_thresholds(monkeypatch):
    _tiny_forest(monkeypatch)
    rng = np.random.default_rng(4)
    labels = np.tile([0, 1], 60)
    features = rng.normal(0, 0.1, size=(len(labels), 8))
    features[:, 0] += labels
    split = np.repeat(["train", "val", "test"], 40)
    groups = [f"{part}_{i}" for part in ("train", "val", "test") for i in range(40)]
    df = pd.DataFrame({
        "y": labels,
        "split_tier1": split,
        "group_id": groups,
    })

    result = evaluation._binary_calibration(features, df)

    assert result["temperature"] > 0
    assert result["raw_threshold_at_validation_fpr"] > 0
    assert result["threshold_at_validation_fpr"] > 0
    assert result["test"]["raw_operating_point"]["threshold"] == (
        result["raw_threshold_at_validation_fpr"]
    )
    assert result["test"]["temperature_scaled_operating_point"]["threshold"] == (
        result["threshold_at_validation_fpr"]
    )


def test_phantom_open_set_excludes_unknown_from_training_and_calibration(monkeypatch):
    _tiny_forest(monkeypatch)
    rng = np.random.default_rng(12)
    names = ("background", "bebop", "ar", "phantom")
    rows = []
    vectors = []
    for split in ("train", "val", "test"):
        for class_id, name in enumerate(names):
            if name == "phantom" and split != "test":
                continue
            for sample in range(15):
                feature = rng.normal(0, 0.08, size=8)
                feature[class_id] += 2.0
                vectors.append(feature)
                rows.append({
                    "drone_model": None if name == "background" else name,
                    "split_tier1": split,
                    "group_id": f"{split}_{name}_{sample}",
                })
    df = pd.DataFrame(rows)

    result = evaluation._multiclass_open_set(np.asarray(vectors), df)

    assert result["unknown_drone_model"] == "phantom"
    assert "phantom" not in result["known_classes"]
    assert result["open_set_test"]["known_count"] == 30
    assert result["open_set_test"]["unknown_count"] == 15
    assert 0 <= result["open_set_test"]["unknown_roc_auc"] <= 1


def test_missing_feature_cache_names_required_files(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "FEATURES", tmp_path / "rf_features")

    with pytest.raises(FileNotFoundError, match="prepared feature cache"):
        evaluation._load_features()
