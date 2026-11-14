import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronedet.data import rf as R
from dronedet.data import rf_emitters as RE


@pytest.mark.parametrize("family", RE.FAMILIES)
@pytest.mark.parametrize("pipeline", ["dronerf", "tampere"])
def test_emitter_window_shape_and_signal(family, pipeline):
    w = RE.emitter_window(family, pipeline, np.random.default_rng(0), snr_db=20)
    assert w.shape == (2, R.DEFAULT.n_bins)
    assert np.isfinite(w).all()
    # a 20 dB emitter must stand out of the noise floor in at least one band
    assert (w.max(axis=1) - np.percentile(w, 10, axis=1)).max() > 10


def test_tf_features_separate_bursty_emitter_from_noise():
    rng = np.random.default_rng(1)
    n = 3_000_000
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    wifi = RE._scene("wifi", 120e6, n, rng, 20)
    f_noise = [RE.tf_features(b) for b in RE.tampere_band_signals(noise)]
    f_wifi = [RE.tf_features(b) for b in RE.tampere_band_signals(wifi)]
    assert all(f.shape == (8,) for f in f_noise + f_wifi)
    # duty cycle: noise has almost no active frames; Wi-Fi bursts in the band
    # holding its channel (either half of 2.4 GHz)
    assert max(f[0] for f in f_noise) < 0.05
    assert max(f[0] for f in f_wifi) > 0.05
