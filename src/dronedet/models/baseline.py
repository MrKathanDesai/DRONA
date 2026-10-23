from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def summary_features(X: np.ndarray) -> np.ndarray:
    """(N, mels, frames) dB -> (N, 3 * mels), invariant to overall level.

    Per band: mean relative to the clip's own mean level (spectral shape),
    std over time, and mean |frame-to-frame change| (stationarity). Adding a
    constant dB offset to a clip changes none of them, so recording gain and
    distance cannot be used as a shortcut.
    """
    level = X.mean(axis=(1, 2), keepdims=True)
    shape = (X - level).mean(axis=2)
    spread = X.std(axis=2)
    flux = np.abs(np.diff(X, axis=2)).mean(axis=2)
    return np.concatenate([shape, spread, flux], axis=1).astype(np.float32)


def clip_level(X: np.ndarray) -> np.ndarray:
    """Mean dB per clip. Used only as a shortcut probe, never as a feature."""
    return X.mean(axis=(1, 2))


MODELS = {
    "logreg": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
    "hgb": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced",
        random_state=0),
}
