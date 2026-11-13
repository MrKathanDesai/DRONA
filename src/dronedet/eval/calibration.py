from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import log_loss


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.column_stack((1 - probabilities, probabilities))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (samples, classes>=2)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("each probability row must have a positive sum")
    probabilities = probabilities / row_sums[:, None]
    return np.log(np.clip(probabilities, 1e-12, 1.0))


def temperature_scale(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Scale class probabilities using a positive scalar temperature."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    logits = _probabilities_to_logits(probabilities)
    scaled = logits / temperature
    scaled -= logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(scaled)


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit scalar temperature by minimizing multiclass NLL on calibration data."""
    labels = np.asarray(labels)
    logits = _probabilities_to_logits(probabilities)
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("labels must be one-dimensional and match probability rows")
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("labels must be finite, non-negative class indices")
    labels = labels.astype(int)
    if (labels >= logits.shape[1]).any() or np.unique(labels).size < 2:
        raise ValueError("calibration labels must contain at least two valid classes")

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        return float(logsumexp(scaled, axis=1).mean() - scaled[
            np.arange(len(labels)), labels
        ].mean())

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"temperature fitting failed: {result.message}")
    return float(np.exp(result.x))


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Report top-label ECE, multiclass Brier score, NLL, and reliability bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.column_stack((1 - probabilities, probabilities))
    probabilities = np.exp(_probabilities_to_logits(probabilities))
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels must be one-dimensional and match probability rows")
    if (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("labels must be valid class indices")
    labels = labels.astype(int)

    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(mask.sum())
        if not count:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        bins.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })

    one_hot = np.eye(probabilities.shape[1])[labels]
    return {
        "ece": float(ece),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
        "sample_count": int(len(labels)),
        "bins": bins,
    }
