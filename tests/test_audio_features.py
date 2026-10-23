from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
# pyrefly: ignore [missing-import]
from dronedet.features import audio as A
# pyrefly: ignore [missing-import]
from dronedet.paths import DATA_DIR, resolve

CFG = A.DEFAULT
FEATS = DATA_DIR / "features"
needs_cache = pytest.mark.skipif(
    not (FEATS / "config.json").exists(),
    reason="run scripts/make_features.py first")


def tone(freq, seconds, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- pure unit tests: no data needed -------------------------------------

def test_shape_matches_config():
    assert A.log_mel(tone(440, 1.0)).shape == (CFG.n_mels, CFG.n_frames)


def test_long_clips_are_centre_cropped():
    y = np.arange(20000, dtype=np.float32)
    out = A.fix_length(y, 16000)
    assert len(out) == 16000 and out[0] == 2000


def test_short_clips_loop_instead_of_zero_padding():
    y = np.ones(9000, dtype=np.float32)
    out = A.fix_length(y, 16000)
    assert len(out) == 16000
    assert (out != 0).all()


def test_clip_length_is_not_a_feature():
    """Clip duration predicts the class in this corpus. The same stationary
    sound at two lengths must give the same per-band profile, and the short
    one must have no quiet tail."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(16384).astype(np.float32) * 0.1
    short = A.log_mel(A.fix_length(noise[:15360], 16000))   # 0.960 s
    full = A.log_mel(A.fix_length(noise, 16000))            # 1.024 s
    frame_energy = short.mean(axis=0)
    assert frame_energy[-3:].min() > np.median(frame_energy) - 3.0
    assert np.abs(short.mean(axis=1) - full.mean(axis=1)).max() < 1.0


def test_silence_is_finite():
    X = A.log_mel(np.zeros(16000, dtype=np.float32))
    assert np.isfinite(X).all()


def test_resampling_preserves_pitch(tmp_path):
    t = np.arange(44100) / 44100
    sf.write(tmp_path / "a.wav", 0.5 * np.sin(2 * np.pi * 1000 * t), 44100)
    from_44k = A.clip_features(tmp_path / "a.wav")
    native = A.log_mel(tone(1000, 1.0))
    assert from_44k.mean(axis=1).argmax() == native.mean(axis=1).argmax()


def test_features_are_deterministic():
    y = tone(300, 1.0)
    assert np.array_equal(A.log_mel(y), A.log_mel(y))


# --- cache tests: the committed code must reproduce the cached file ------

@needs_cache
def test_cache_was_made_with_current_config():
    meta = json.loads((FEATS / "config.json").read_text())
    assert meta["fingerprint"] == CFG.fingerprint(), \
        "feature config changed; rerun scripts/make_features.py"


@needs_cache
def test_cache_matches_current_manifest():
    import make_features
    df = pd.read_parquet(DATA_DIR / "manifest.parquet")
    df = df[df.exclude_reason == ""].reset_index(drop=True)
    meta = json.loads((FEATS / "config.json").read_text())
    assert meta["manifest_hash"] == make_features.manifest_hash(df), \
        "manifest changed since features were cached"
    idx = pd.read_parquet(FEATS / "index.parquet")
    assert list(idx.path) == list(df.path)


@needs_cache
def test_cache_reproduces_from_code():
    X = np.load(FEATS / "X.npy", mmap_mode="r")
    idx = pd.read_parquet(FEATS / "index.parquet")
    assert X.shape == (len(idx), CFG.n_mels, CFG.n_frames)
    # every distinct duration (incl. the short, looped ones) plus a random draw
    rows = set(idx.groupby(idx.duration.round(3)).head(1).index)
    rows |= set(np.random.default_rng(0).choice(len(idx), 20, replace=False))
    for i in sorted(rows):
        again = A.clip_features(resolve(idx.path[i]))
        np.testing.assert_allclose(again, X[i], atol=1e-4,
                                   err_msg=idx.path[i])


@needs_cache
def test_cache_is_finite():
    X = np.load(FEATS / "X.npy", mmap_mode="r")
    for s in range(0, len(X), 2048):
        assert np.isfinite(X[s:s + 2048]).all()
