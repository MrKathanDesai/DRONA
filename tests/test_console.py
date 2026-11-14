import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from console.server import CALIBRATION, Session, to_16k


def run(states):
    s = Session()
    return [s.step(x) for x in states]


def test_two_drone_windows_confirm_and_hold_briefly():
    out = run(["drone", "drone", "no drone", "no drone", "no drone", "no drone"])
    assert out == ["maybe", "drone", "drone", "drone", "drone", "clear"]


def test_single_hit_or_repeated_doubt_is_maybe():
    assert run(["no drone", "drone", "no drone"]) == ["clear", "maybe", "clear"]
    assert run(["low confidence", "low confidence"]) == ["clear", "maybe"]


def test_unknown_drone_is_flagged():
    s = Session()
    s.step("unknown drone")
    assert s.step("unknown drone") == "drone" and s.unfamiliar


def test_resample_to_16k():
    for rate in (48000, 44100, 16000):
        assert len(to_16k(np.zeros(rate, np.float32), rate)) == 16000


def test_calibration_file_is_valid():
    assert set(CALIBRATION) <= {"confirm", "hold"}
    s = Session(**CALIBRATION)
    states = ["drone"] * s.confirm
    assert [s.step(x) for x in states][-1] == "drone"
