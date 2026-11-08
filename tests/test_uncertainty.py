from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# pyrefly: ignore [missing-import]
from dronedet.models import explain as X
from dronedet.models import pretrained as PT
from dronedet.models import uncertainty as U


def head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def toy(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    F = rng.standard_normal((n, d)) + y[:, None] * 1.5
    groups = np.array([f"g{i // 4}_{y[i]}" for i in range(n)])
    return F, y, groups


def test_familiarity_flags_shifted_data():
    F, _, _ = toy()
    fam = U.Familiarity(n_components=8).fit(F[:300]).calibrate(F[300:], 0.95)
    assert fam.unfamiliar(F[300:]).mean() == pytest.approx(0.05, abs=0.02)
    shifted = F[300:] + 6.0
    assert fam.unfamiliar(shifted).mean() > 0.9


def test_margin_band_holds_requested_share():
    z = np.random.default_rng(0).standard_normal(10000) * 3
    m = U.margin_for_review_rate(z, 0.10)
    assert (np.abs(z) < m).mean() == pytest.approx(0.10, abs=0.005)


def test_ensemble_members_see_both_classes_and_pickle():
    F, y, g = toy()
    ens = U.HeadEnsemble(head, n=5).fit(F, y, g)
    L = ens.member_logits(F)
    assert L.shape == (5, len(F)) and np.isfinite(L).all()
    back = pickle.loads(pickle.dumps(ens))           # no make_head in the pickle
    np.testing.assert_allclose(back.member_logits(F), L)


def test_ensemble_accepts_sample_weight():
    F, y, g = toy()
    U.HeadEnsemble(head, n=2).fit(F, y, g, sample_weight=np.ones(len(y)))


def test_decide_priorities():
    hit = np.array([True, False, True, True, True])
    low = np.array([False, False, True, False, False])
    unf = np.array([False, False, False, True, False])
    far = np.array([False, False, False, False, True])
    state, reason = U.decide(hit, low, unf, far)
    assert list(state) == ["drone", "no drone", "low confidence",
                           "low confidence", "unknown drone"]
    assert reason[3] == "input unlike training data"


def test_risk_coverage_rewards_informative_uncertainty():
    err = np.r_[np.zeros(90), np.ones(10)]
    good = np.r_[np.zeros(90), np.ones(10)]           # errors are most uncertain
    bad = -good
    rg, rb = U.risk_coverage(err, good), U.risk_coverage(err, bad)
    assert rg["error_at_coverage"]["90%"] == 0.0
    assert rg["error_at_coverage"]["100%"] == pytest.approx(0.1)
    assert rg["aurc"] < rb["aurc"]


def test_linear_head_matches_pipeline():
    F, y, _ = toy()
    h = head().fit(F, y)
    w, b = X.linear_head(h)
    np.testing.assert_allclose(F @ w + b, h.decision_function(F), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not PT.CHECKPOINT.exists(), reason="run scripts/download_data.py first")
def test_gradcam_logit_matches_head_and_map_is_normalised():
    enc = PT.Encoder(device="cpu")
    rng = np.random.default_rng(0)
    clip = (0.1 * np.sin(2 * np.pi * 400 * np.arange(16000) / 16000)
            + 0.02 * rng.standard_normal(16000)).astype(np.float32)
    w = rng.standard_normal(PT.EMBED_DIM).astype(np.float32) * 0.01
    e, _ = enc([clip])
    for layer in X.LAYERS.values():
        M, cam, z = X.gradcam(enc, clip, w, 0.3, layer)
        assert M.shape == cam.shape == (101, 64)
        assert 0.0 <= cam.min() and cam.max() <= 1.0
        assert z == pytest.approx(float(e[0] @ w + 0.3), rel=1e-3, abs=1e-3)
