import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronedet.data import rf as R
from dronedet.data import rf_tampere as T


def _tone_iq(f_hz: float, n: int, amp: float = 3000.0) -> np.ndarray:
    t = np.arange(n) / T.FS_IN
    z = amp * np.exp(2j * np.pi * f_hz * t)
    rng = np.random.default_rng(0)
    z = z + rng.normal(0, 5, n) + 1j * rng.normal(0, 5, n)
    return np.stack([z.real, z.imag], axis=1).round().astype("<i2")


@pytest.mark.parametrize("rf_hz, band", [(2.430e9, 0), (2.470e9, 1)])
def test_tone_lands_in_right_band_and_bin(rf_hz, band):
    offset = rf_hz - 2.44e9                          # relative to the file centre
    per_win = int(round(R.DEFAULT.window_s * T.FS_IN))
    psd, level = T.to_dronerf_psd(_tone_iq(offset, 2 * per_win))
    assert psd.shape == (2, 2, R.DEFAULT.n_bins)
    other = 1 - band
    # 10 MHz from the band centre, after folding by the real part
    expect_bin = round(10e6 / (R.DEFAULT.sample_rate / R.DEFAULT.n_fft))
    assert abs(int(np.argmax(psd[0, band])) - expect_bin) <= 2
    assert level[0, band] > level[0, other] + 20


def test_parse_name():
    assert T.parse_name("DJI_inspire_2_2G.bin") == ("DJI_inspire_2", 1)
    assert T.parse_name("Yuneec_typhoon_h_2G_2of2.bin") == ("Yuneec_typhoon_h", 2)
    with pytest.raises(ValueError):
        T.parse_name("DJI_inspire_2_5G_1of2.bin")
