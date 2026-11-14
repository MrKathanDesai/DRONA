"""Any audio or video file -> 16 kHz mono WAV (src/dronedet/media.py)."""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet import media as M


def _tone(seconds=2.0, sr=44_100, channels=2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    return np.stack([y] * channels, axis=1).astype(np.float32)


def _wav_bytes(y, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def _check(clip: M.Clip, seconds: float):
    y, sr = sf.read(io.BytesIO(clip.wav))
    assert sr == M.SR and y.ndim == 1
    assert abs(len(y) / sr - seconds) < 0.15
    assert np.abs(y).max() > 0.1                  # the tone survived


def test_wav_stereo_44k_becomes_16k_mono():
    clip = M.decode_bytes(_wav_bytes(_tone(), 44_100), "tone.wav")
    _check(clip, 2.0)
    assert clip.title == "tone.wav" and not clip.truncated


def test_long_input_is_truncated(monkeypatch):
    monkeypatch.setattr(M, "MAX_SECONDS", 1)
    clip = M.decode_bytes(_wav_bytes(_tone(3.0), 44_100), "long.wav")
    assert clip.truncated and abs(clip.seconds - 1.0) < 1e-6


def test_empty_and_garbage_inputs_give_a_clear_error():
    with pytest.raises(M.MediaError):
        M.decode_bytes(b"", "x.wav")
    if M.ffmpeg_exe():
        with pytest.raises(M.MediaError):
            M.decode_bytes(b"not audio at all" * 100, "x.mp4")


@pytest.fixture(scope="module")
def ffmpeg():
    exe = M.ffmpeg_exe()
    if not exe:
        pytest.skip("ffmpeg not available (pip install imageio-ffmpeg)")
    return exe


@pytest.mark.parametrize("ext, args", [
    ("mp3", ["-c:a", "libmp3lame"]),
    ("m4a", ["-c:a", "aac"]),
    ("webm", ["-c:a", "libopus"]),
])
def test_compressed_audio_formats(ffmpeg, tmp_path, ext, args):
    src = tmp_path / "tone.wav"
    src.write_bytes(_wav_bytes(_tone(), 44_100))
    out = tmp_path / f"tone.{ext}"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), *args, str(out)],
                   check=True)
    _check(M.decode_bytes(out.read_bytes(), out.name), 2.0)


def test_video_file_keeps_only_its_sound(ffmpeg, tmp_path):
    out = tmp_path / "clip.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
                    "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(out)], check=True)
    _check(M.decode_bytes(out.read_bytes(), out.name), 2.0)


def test_video_without_sound_is_reported(ffmpeg, tmp_path):
    out = tmp_path / "silent.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:d=1", "-c:v", "mpeg4", str(out)], check=True)
    with pytest.raises(M.MediaError):
        M.decode_bytes(out.read_bytes(), out.name)


@pytest.mark.parametrize("url, kind", [
    ("https://www.youtube.com/watch?v=abc", "site"),
    ("https://youtu.be/abc", "site"),
    ("https://example.org/sounds/drone.mp3", "direct"),
    ("http://example.org/a/b/clip.MP4?x=1", "direct"),
])
def test_classify_url(url, kind):
    assert M.classify_url(url) == kind


@pytest.mark.parametrize("bad", ["", "youtube.com/watch?v=x", "ftp://x/y.wav", "file:///C:/a.wav"])
def test_bad_links_are_rejected(bad):
    with pytest.raises(M.MediaError):
        M.classify_url(bad)


def test_upload_types_cover_audio_and_video():
    for ext in ("wav", "mp3", "m4a", "flac", "ogg", "mp4", "webm", "mov", "mkv"):
        assert ext in M.UPLOAD_TYPES
