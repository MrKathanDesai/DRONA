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


def test_session_audio_keeps_exact_time_at_48k():
    from console.server import SessionAudio
    s = SessionAudio(48000)
    t = np.arange(48000 * 5) / 48000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    for i in range(0, len(x), 4096):                  # browser-sized chunks
        s.add(x[i:i + 4096])
    assert s.end_s == 5.0
    clip = s.slice(1.0, 3.5)
    assert len(clip) == int(2.5 * 16000)
    assert 0.6 < float(np.sqrt(np.mean(clip ** 2))) < 0.8   # the tone survives resampling


def test_session_audio_drops_old_sound_and_clamps_slices():
    from console.server import SessionAudio
    s = SessionAudio(16000, max_s=3)
    for k in range(5):
        s.add(np.full(16000, k, np.float32))          # second k has value k
    assert s.end_s == 5.0 and s.offset == 2 * 16000
    assert len(s.slice(0, 2)) == 0                    # already dropped
    assert set(np.unique(s.slice(2.0, 3.0))) == {2.0}
    assert len(s.slice(4.5, 9)) == 8000               # clamped to what exists
