"""Tests for the manifest. These are the ones that catch the bugs that
silently inflate results rather than crashing.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.data import manifest as M          # noqa: E402

MANIFEST = ROOT / "data" / "manifest.parquet"
pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="run scripts/build_manifest.py first")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(MANIFEST)


# ------------------------------------------------------------- leakage
def test_no_group_spans_a_split(df):
    """The headline guarantee. If this fails, every reported number is wrong."""
    for tier in ("split_tier1", "split_tier2"):
        sub = df[df[tier] != "excluded"]
        spanning = sub.groupby("group_id")[tier].nunique().loc[lambda s: s > 1]
        assert spanning.empty, f"{tier}: {list(spanning.index[:5])}"


def test_validate_catches_injected_leak(df):
    """Prove the check actually works — a guard nobody has seen fail is not
    a guard."""
    bad = df.copy()
    g = bad.loc[bad.split_tier1 == "train", "group_id"].iloc[0]
    idx = bad.index[bad.group_id == g][0]
    bad.loc[idx, "split_tier1"] = "test"
    with pytest.raises(M.LeakageError):
        M.validate(bad)


def test_tier2_holds_out_a_whole_drone_model(df):
    """Train on bebop, test on membo — no membo clip may appear in training."""
    train = df[(df.split_tier2 == "train") & (df.label == "drone")]
    test = df[(df.split_tier2 == "test") & (df.label == "drone")]
    assert set(train.drone_model) == {"bebop"}
    assert set(test.drone_model) == {"membo"}


def test_every_split_has_both_classes(df):
    for tier in ("split_tier1", "split_tier2"):
        for sp in set(df[tier]) - {"excluded"}:
            assert df[df[tier] == sp].label.nunique() == 2, f"{tier}/{sp}"


# ------------------------------------------------------------- grouping
def test_positive_slices_group_by_parent_recording():
    for name, want in [
        ("B_S2_D1_067-bebop_000_.wav", "B_S2_D1_067"),
        ("B_S2_D1_067-bebop_004_.wav", "B_S2_D1_067"),
        ("mixed_membo_34-membo_2_.wav", "mixed_membo_34"),
    ]:
        assert M.positive_group(name) == want


def test_negative_slices_group_by_source_clip():
    """ESC-50 clips were cut into segments too — they leak the same way."""
    esc = M.load_esc50(ROOT / "data" / "esc50_meta.csv")
    g1, c1 = M.negative_info("1-172649-A-403.wav", esc)
    g2, c2 = M.negative_info("1-172649-A-401.wav", esc)
    assert g1 == g2, "segments of one ESC-50 clip must share a group"
    assert c1 == c2 == "helicopter"


# ------------------------------------------------------------- labels
def test_categories_come_from_metadata_not_filename_arithmetic(df):
    """Decoding the category as last_field // 10 disagrees with ESC-50's own
    metadata on ~20 files. We join instead, so we must match exactly."""
    import re
    esc = M.load_esc50(ROOT / "data" / "esc50_meta.csv")
    checked = mismatched = 0
    for p, cat in zip(df.path, df.bg_category):
        name = Path(p).name
        m = M.ESC_NAME.match(name)
        if not m or cat is None:
            continue
        _, src, take, tail = m.groups()
        checked += 1
        if esc[(src, take)]["category"] != cat:
            mismatched += 1
        if int(tail) // 10 != int(esc[(src, take)]["target"]):
            pass          # the arithmetic shortcut is wrong here; we don't use it
    assert checked > 9000
    assert mismatched == 0


def test_hard_confusers_are_present(df):
    """The negatives must contain rotor and machine sounds, or a high score
    means nothing."""
    for c in ("helicopter", "chainsaw", "vacuum_cleaner",
              "washing_machine", "engine"):
        assert (df.bg_category == c).sum() >= 100, c


# ------------------------------------------------------------- hygiene
def test_splits_are_deterministic():
    """Same group id must land in the same split on every machine, forever."""
    a = [M.assign_tier1(f"group_{i}") for i in range(500)]
    b = [M.assign_tier1(f"group_{i}") for i in range(500)]
    assert a == b
    assert set(a) == {"train", "val", "test"}


def test_short_clips_are_excluded_not_dropped(df):
    short = df[df.duration < M.MIN_DURATION]
    assert len(short) > 0
    assert (short.exclude_reason != "").all()
    assert (short.split_tier1 == "excluded").all()


def test_no_duplicate_paths(df):
    assert not df.path.duplicated().any()
