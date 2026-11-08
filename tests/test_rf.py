from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# pyrefly: ignore [missing-import]
from dronedet.data import rf as R
# pyrefly: ignore [missing-import]
from dronedet.data.manifest import LeakageError, validate
# pyrefly: ignore [missing-import]
from dronedet.models import rf as M


def segments() -> pd.DataFrame:
    """The real shape of the dataset: one row per (bui, segment)."""
    rows = [("00000", s) for s in range(41)]
    for bui in ("10000", "10001", "10010", "10011",
                "10100", "10101", "10110", "11000"):
        rows += [(bui, s) for s in range(21)]
    rows += [("10111", s) for s in range(18)]
    return pd.DataFrame([dict(bui=b, segment=s, **R.decode_bui(b)) for b, s in rows])


def as_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["path"] = df.bui + "_" + df.segment.astype(str)
    df["start_s"], df["exclude_reason"] = 0.0, ""
    return df


def test_dataset_shape_matches_table_1():
    counts = segments().drone_model.fillna("background").value_counts().to_dict()
    assert counts == R.EXPECT_SEGMENTS


def test_decode_bui():
    assert R.decode_bui("00000") == dict(label="background", drone_model=None, mode=None)
    assert R.decode_bui("10011") == dict(label="drone", drone_model="bebop",
                                         mode="flying_video")
    assert R.decode_bui("10110")["drone_model"] == "ar"
    assert R.decode_bui("11000") == dict(label="drone", drone_model="phantom",
                                         mode="on_connected")


def test_names_including_the_misspelt_archive():
    assert R.archive_key("RF Data_11000_L1.rar") == "11000_L1"
    assert R.archive_key("FR Data_00000_H2.rar") == "00000_H2"
    assert R.parse_member("11000L_9.csv") == ("11000", "L", 9)
    assert R.parse_member("folder/10111H_17.csv") == ("10111", "H", 17)
    with pytest.raises(ValueError):
        R.parse_member("readme.txt")


def test_splits_pass_the_leakage_validator():
    validate(as_manifest(R.assign_splits(segments())))


def test_validator_catches_injected_leak():
    df = as_manifest(R.assign_splits(segments()))
    group = df[df.split_tier1 == "train"].group_id.iloc[0]
    df.loc[df[df.group_id == group].index[0], "split_tier1"] = "test"
    with pytest.raises(LeakageError):
        validate(df)


def test_held_out_time_comes_after_training_time():
    df = R.assign_splits(segments())
    for _, sub in df.groupby("bui"):
        tr, va, te = (sub[sub.split_tier1 == s].segment
                      for s in ("train", "val", "test"))
        assert tr.max() < va.min() and va.max() < te.min()


def test_every_recording_is_in_every_tier1_split():
    df = R.assign_splits(segments())
    assert (df.groupby("bui").split_tier1.nunique() == 3).all()


def test_tier2_never_trains_on_the_held_out_drone():
    df = R.assign_splits(segments())
    drones = df[df.label == "drone"]
    assert set(drones[drones.split_tier2 == "test"].drone_model) == {R.HELD_OUT_DRONE}
    assert not drones[(drones.drone_model == R.HELD_OUT_DRONE)
                      & (drones.split_tier2 == "train")].shape[0]
    assert set(df[df.label == "background"].split_tier2) == {"train", "test"}


def test_psd_shape_and_tone_frequency():
    cfg = R.RFConfig(windows_per_segment=2)
    f = 5e6
    t = np.arange(R.SEGMENT_SAMPLES // 10) / cfg.sample_rate
    x = np.resize(np.sin(2 * np.pi * f * t), R.SEGMENT_SAMPLES).astype(np.float32)
    psd, level = R.segment_psd(x, cfg)
    assert psd.shape == (2, cfg.n_bins) and level.shape == (2,)
    assert abs(psd[0].argmax() * cfg.sample_rate / cfg.n_fft - f) < cfg.sample_rate / cfg.n_fft


def test_silence_is_finite():
    psd, level = R.segment_psd(np.zeros(R.SEGMENT_SAMPLES, dtype=np.float32))
    assert np.isfinite(psd).all() and np.isfinite(level).all()


def test_archive_stream_keeps_members_apart(tmp_path, monkeypatch):
    """Members must not run together, and short files must be refused."""
    monkeypatch.setattr(R, "SEGMENT_SAMPLES", 4)
    (tmp_path / "10000L_0.csv").write_text("1.000000,2.000000,3.000000,4.000000\r\n")
    (tmp_path / "10000L_1.csv").write_text("5.000000,6.000000,7.000000,8.000000\r\n")
    arc = tmp_path / "a.zip"
    subprocess.run(["bsdtar", "-a", "-cf", str(arc), "-C", str(tmp_path),
                    "10000L_0.csv", "10000L_1.csv"], check=True)
    got = dict(R.iter_archive(arc))
    assert got["10000L_0.csv"].tolist() == [1, 2, 3, 4]
    assert got["10000L_1.csv"].tolist() == [5, 6, 7, 8]

    monkeypatch.setattr(R, "SEGMENT_SAMPLES", 5)
    with pytest.raises(ValueError):
        list(R.iter_archive(arc))


def test_every_drone_can_be_the_held_out_one():
    for drone in ("bebop", "ar", "phantom"):
        df = R.assign_splits(segments(), held_out=drone)
        validate(as_manifest(df))
        drones = df[df.label == "drone"]
        assert set(drones[drones.split_tier2 == "test"].drone_model) == {drone}


def test_shape_features_ignore_each_bands_level():
    X = np.random.default_rng(0).normal(-40, 5, (4, 2, 1025)).astype(np.float32)
    louder = X.copy()
    louder[:, 0] += 17.0
    louder[:, 1] -= 9.0
    np.testing.assert_allclose(M.shape_features(X), M.shape_features(louder), atol=1e-4)


def test_cnn_is_small_learns_and_survives_sklearn_clone():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 2, 257)).astype(np.float32)
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    X[60:, 0, 100:110] += 4.0
    model = clone(M.TinyCNN(epochs=15)).fit(X, y)
    assert sum(p.numel() for p in model.net_.parameters()) < 15_000
    assert (model.predict(X) == y).mean() > 0.95
    assert model.predict_proba(X).shape == (120, 2)
