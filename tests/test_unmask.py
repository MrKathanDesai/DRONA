import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronedet.features.unmask import stationary

SR = 16000


def rotor_and_music(seconds=5, seed=0):
    t = np.arange(seconds * SR) / SR
    rotor = sum(np.sin(2 * np.pi * 180 * h * t) / h for h in range(1, 6))
    rng = np.random.default_rng(seed)
    notes = np.repeat(rng.choice([220, 262, 330, 392, 440, 523], int(seconds / 0.3) + 1),
                      int(0.3 * SR))[: len(t)]
    music = np.sin(2 * np.pi * np.cumsum(notes) / SR)
    return rotor / rotor.std(), music / music.std()


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_keeps_steady_rotor_and_drops_changing_music():
    rotor, music = rotor_and_music()
    mix = (rotor + 2 * music).astype(np.float32)          # music 6 dB louder
    out = stationary(mix)
    assert out.shape == mix.shape
    assert corr(out, rotor) > 0.9 > corr(mix, rotor)
    assert abs(corr(out, music)) < 0.3


def test_steady_signal_passes_almost_unchanged():
    rotor, _ = rotor_and_music()
    out = stationary(rotor.astype(np.float32))
    assert corr(out, rotor) > 0.98
