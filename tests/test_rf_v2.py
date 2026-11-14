import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dronedet.data import rf as R
from dronedet.data import rf_noise as RN
from dronedet.models import rf as M


@pytest.mark.parametrize("family", RN.FAMILIES)
def test_noise_window_shape_and_no_carrier(family):
    psd = RN.noise_window(family, "dronerf", np.random.default_rng(0))
    assert psd.shape == (2, R.DEFAULT.n_bins) and np.isfinite(psd).all()
    # no transmitter: no narrow line sticks far out of the local floor
    for b in range(2):
        local = np.convolve(psd[b], np.ones(31) / 31, mode="same")
        assert (psd[b] - local)[20:-20].max() < 10.0


def test_tampere_pipeline_noise_runs():
    psd = RN.noise_window("white", "tampere", np.random.default_rng(1))
    assert psd.shape == (2, R.DEFAULT.n_bins) and np.isfinite(psd).all()


def test_noise_bank_rows_match():
    X, rows = RN.noise_bank(2, ("white",), ("dronerf",), seed=3)
    assert len(X) == len(rows) == 2 and rows[0]["family"] == "white"


def test_tilt_feature_ignores_common_gain_but_sees_band_difference():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (4, 2, R.DEFAULT.n_bins)).astype(np.float32)
    a, b = M.tilt_features(X), M.tilt_features(X + 17.0)
    np.testing.assert_allclose(a, b, atol=1e-4)            # same gain on both bands
    X2 = X.copy()
    X2[:, 0] += 10.0                                       # one band louder
    np.testing.assert_allclose(M.tilt_features(X2)[:, -1] - a[:, -1], 10.0, atol=1e-4)


def test_tampere_split_is_deterministic_and_by_drone():
    import train_rf_v2 as V
    assert V.tampere_group("Parrot_mambo_control") == V.tampere_group("Parrot_mambo_video")
    assert V.tampere_split("DJI_mavic_pro") == V.tampere_split("DJI_mavic_pro")
    assert {V.tampere_split(f"d{i}") for i in range(40)} == {"train", "val", "test"}


def test_choose_rule():
    import train_rf_v2 as V

    def res(noise, lodo, unseen):
        return {"noise_max_fpr": noise, "tampere_lodo": [{"recall": lodo}],
                "dronerf_unseen_drone_recall": {"x": unseen}}
    assert V.choose({"v2": res(0.05, 0.9, 0.5), "v2b": res(0.05, 0.95, 0.5)}) == "v2b"
    assert V.choose({"v2": res(0.05, 0.9, 0.5), "v2b": res(0.10, 0.99, 0.9)}) == "v2"
    assert V.choose({"v2": res(0.05, 0.9, 0.5), "v2b": res(0.01, 0.80, 0.5)}) == "v2"
