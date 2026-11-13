from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def unknownness_scores(
    probabilities: np.ndarray,
    known_class_indices: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return one minus the largest probability among known in-scope classes."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    indices = np.asarray(known_class_indices, dtype=int)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if not len(indices) or (indices < 0).any() or (indices >= probabilities.shape[1]).any():
        raise ValueError("known class indices must select at least one valid class")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    known_confidence = probabilities[:, indices].max(axis=1)
    return 1.0 - known_confidence


def open_set_metrics(
    unknown_scores: np.ndarray,
    is_unknown: np.ndarray,
    acceptance_threshold: float,
) -> dict:
    """Evaluate unknown AUROC and known-accept/unknown-reject rates."""
    scores = np.asarray(unknown_scores, dtype=np.float64)
    labels = np.asarray(is_unknown, dtype=bool)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("scores and labels must be matching 1D arrays")
    if not np.isfinite(scores).all() or not np.isfinite(acceptance_threshold):
        raise ValueError("scores and threshold must be finite")
    if labels.all() or (~labels).all():
        raise ValueError("open-set evaluation requires known and unknown examples")
    accept = scores <= acceptance_threshold
    known = ~labels
    unknown = labels
    return {
        "unknown_roc_auc": float(roc_auc_score(labels, scores)),
        "known_accept_rate": float(accept[known].mean()),
        "unknown_reject_rate": float((~accept[unknown]).mean()),
        "known_count": int(known.sum()),
        "unknown_count": int(unknown.sum()),
        "acceptance_unknownness_threshold": float(acceptance_threshold),
    }
