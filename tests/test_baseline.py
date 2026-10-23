from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# pyrefly: ignore [missing-import]
from dronedet.eval import metrics as E
# pyrefly: ignore [missing-import]
from dronedet.models import baseline as B


def test_summary_features_ignore_overall_level():
    """Gain and distance shift a clip's dB by a constant; the baseline must
    not see that (loudness alone separates the classes at ~0.73 AUC)."""
    X = np.random.default_rng(0).normal(-40, 10, (4, 128, 63)).astype(np.float32)
    np.testing.assert_allclose(B.summary_features(X),
                               B.summary_features(X + 17.0), atol=1e-4)


def test_threshold_respects_target_fpr():
    neg = np.random.default_rng(1).random(1000)
    for fpr in (0.0, 0.01, 0.05, 0.2):
        thr = E.threshold_at_fpr(neg, fpr)
        assert (neg >= thr).mean() <= fpr


def test_operating_point_counts():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = E.operating_point(y, s, 0.5)
    assert r["recall"] == 0.5 and r["false_alarms"] == 1 and r["negatives"] == 3


def test_bootstrap_resamples_groups_not_clips():
    """Ten identical copies of one recording are one piece of evidence: the
    CI must be as wide as with a single copy."""
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(20), np.zeros(20)].astype(int)
    s = np.r_[rng.normal(1, 1, 20), rng.normal(0, 1, 20)]
    g = np.arange(40)
    one = E.grouped_bootstrap(y, s, g, lambda a, b: b[a == 1].mean(), n=500)
    ten = E.grouped_bootstrap(np.repeat(y, 10), np.repeat(s, 10),
                              np.repeat(g, 10), lambda a, b: b[a == 1].mean(), n=500)
    assert abs((one[1] - one[0]) - (ten[1] - ten[0])) < 0.15


def test_false_alarms_by_category():
    df = pd.DataFrame({"label": ["background"] * 4 + ["drone"],
                       "category": ["a", "a", "b", "b", "drone"]})
    t = E.false_alarms_by_category(df, np.array([0.9, 0.1, 0.1, 0.1, 0.9]), 0.5)
    assert t.loc["a", "fa_rate"] == 0.5 and t.loc["b", "false_alarms"] == 0
    assert "drone" not in t.index
