import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronedet import fusion as FU
from dronedet.eval import metrics as E
from dronedet.models import deep as D


def _scores(rng, n, sep):
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    p = 1 / (1 + np.exp(-(rng.normal(0, 1, 2 * n) + sep * (2 * y - 1))))
    return p, y


def _fusion(rng, method="sum"):
    pa, ya = _scores(rng, 500, 1.0)
    pr, yr = _scores(rng, 500, 2.0)
    cal = {"audio": FU.Calibrator(5).fit(pa, ya), "rf": FU.Calibrator(5).fit(pr, yr)}
    f = FU.Fusion(cal, method)
    if method == "learned":
        f.learned = FU.fit_learned(FU.pair_llrs(f.llr({"audio": pa, "rf": pr}), len(ya)), ya)
    return f.fit_decision({"audio": pa, "rf": pr}, ya, 0.01, 0.1), pa, pr, ya


def test_calibrator_is_monotone_and_capped():
    rng = np.random.default_rng(0)
    p, y = _scores(rng, 500, 1.5)
    c = FU.Calibrator(cap=3).fit(p, y)
    grid = np.linspace(1e-6, 1 - 1e-6, 50)
    llr = c.llr(grid)
    assert np.all(np.diff(llr) >= 0)
    assert llr.min() >= -3 and llr.max() <= 3


def test_missing_sensor_gives_zero_evidence_and_own_threshold():
    rng = np.random.default_rng(1)
    f, pa, pr, y = _fusion(rng)
    only_audio = f.fused({"audio": pa, "rf": None}, len(y))
    assert np.allclose(only_audio, f.calibrators["audio"].llr(pa))
    assert set(f.decision) == {"audio+rf", "audio", "rf"}
    # each sensor set holds its false-alarm target on the data it was fitted on
    for key, sc in (("audio", {"audio": pa}), ("rf", {"rf": pr})):
        fpr = (f.fused(sc, len(y))[y == 0] >= f.threshold(sc)).mean()
        assert fpr <= 0.011, key


def test_assess_states_and_unknown_drone():
    rng = np.random.default_rng(2)
    f, pa, pr, y = _fusion(rng)
    n = len(y)
    no = np.zeros(n, bool)
    flags = {"audio": {"low_margin": no, "unfamiliar": no, "far_from_drones": no},
             "rf": {"low_margin": no, "unfamiliar": no, "far_from_drones": ~no}}
    out = f.assess({"audio": pa, "rf": pr}, n, flags)
    assert set(out["state"]) <= {"drone", "no drone", "unknown drone", "low confidence"}
    # RF says "unlike any known drone" everywhere; confident drone calls with
    # positive RF evidence must therefore be unknown drones, never plain drones
    assert "drone" not in set(out["state"][out["llr_rf"] > 0])


def test_operating_point_reports_proposal_metrics():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = E.operating_point(y, s, 0.5)
    assert r["confusion"] == [[2, 1], [1, 1]]
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["precision"] == pytest.approx(0.5) and r["f1"] == pytest.approx(0.5)


@pytest.mark.parametrize("arch", list(D.ARCHS))
def test_deep_archs_shape_and_tiny_fit(arch):
    net = D.ARCHS[arch]()
    assert net(torch.randn(2, 128, 63)).shape == (2,)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (32, 128, 63)).astype(np.float32)
    y = np.r_[np.ones(16, int), np.zeros(16, int)]
    X[y == 1, 40:50] += 3.0                     # a band only drones have
    m = D.DeepClassifier(arch, epochs=2, batch=16, device="cpu").fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (32, 2) and np.allclose(p.sum(1), 1)


def test_deep_is_level_invariant():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (16, 128, 63)).astype(np.float32)
    y = np.r_[np.ones(8, int), np.zeros(8, int)]
    m = D.DeepClassifier("cnn", epochs=1, batch=8, device="cpu").fit(X, y)
    assert np.allclose(m.decision_function(X), m.decision_function(X + 20.0), atol=1e-4)
