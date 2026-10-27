from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# pyrefly: ignore [missing-import]
from dronedet.data import manifest as M

MANIFEST = ROOT / "data" / "manifest.parquet"
pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="run scripts/build_manifest.py first")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(MANIFEST)


def test_no_group_spans_a_split(df):
    for tier in M.TIERS:
        sub = df[df[tier] != "excluded"]
        spanning = sub.groupby("group_id")[tier].nunique().loc[lambda s: s > 1]
        assert spanning.empty, f"{tier}: {list(spanning.index[:5])}"


def test_validate_catches_injected_leak(df):
    bad = df.copy()
    g = bad.loc[bad.split_tier1 == "train", "group_id"].iloc[0]
    idx = bad.index[bad.group_id == g][0]
    bad.loc[idx, "split_tier1"] = "test"
    with pytest.raises(M.LeakageError):
        M.validate(bad)


def test_tier2_holds_out_a_whole_drone_model(df):
    train = df[(df.split_tier2 == "train") & (df.label == "drone")]
    test = df[(df.split_tier2 == "test") & (df.label == "drone")]
    assert set(train.drone_model) == {"bebop"}
    assert set(test.drone_model) == {"membo"}


def test_every_split_has_both_classes(df):
    for tier in ("split_tier1", "split_tier2"):
        for sp in set(df[tier]) - {"excluded"}:
            assert df[df[tier] == sp].label.nunique() == 2, f"{tier}/{sp}"


def test_positive_slices_group_by_parent_recording():
    for name, want in [
        ("B_S2_D1_067-bebop_000_.wav", "B_S2_D1_067"),
        ("B_S2_D1_067-bebop_004_.wav", "B_S2_D1_067"),
        ("mixed_membo_34-membo_2_.wav", "mixed_membo_34"),
    ]:
        assert M.positive_group(name) == want


def test_negative_slices_group_by_source_clip():
    esc = M.load_esc50(ROOT / "data" / "esc50_meta.csv")
    g1, c1 = M.negative_info("1-172649-A-403.wav", esc)
    g2, c2 = M.negative_info("1-172649-A-401.wav", esc)
    assert g1 == g2
    assert c1 == c2 == "helicopter"


def test_categories_come_from_metadata_not_filename_arithmetic(df):
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
    assert checked > 9000
    assert mismatched == 0


def test_hard_confusers_are_present(df):
    for c in ("helicopter", "chainsaw", "vacuum_cleaner",
              "washing_machine", "engine"):
        assert (df.bg_category == c).sum() >= 100, c


def test_splits_are_deterministic():
    a = [M.assign_tier1(f"group_{i}") for i in range(500)]
    b = [M.assign_tier1(f"group_{i}") for i in range(500)]
    assert a == b
    assert set(a) == {"train", "val", "test"}


def test_short_clips_are_excluded_not_dropped(df):
    short = df[df.duration < M.MIN_DURATION]
    assert len(short) > 0
    assert (short.exclude_reason != "").all()
    assert (short.split_tier1 == "excluded").all()


def test_no_duplicate_rows(df):
    assert not df.duplicated(["path", "start_s"]).any()


def test_outdoor_corpus_is_test_only(df):
    svan = df[df.source == "svanstrom"]
    if svan.empty:
        pytest.skip("Svanstrom audio not downloaded")
    assert (svan.split_tier3 == "test").all()
    assert (svan.split_tier1 == "excluded").all()
    assert (svan.split_tier2 == "excluded").all()
    assert (df[df.source != "svanstrom"].split_tier3 != "test").all()


def test_outdoor_windows_group_by_recording(df):
    svan = df[df.source == "svanstrom"]
    if svan.empty:
        pytest.skip("Svanstrom audio not downloaded")
    assert svan.groupby("path").group_id.nunique().max() == 1
    assert svan.groupby("group_id").size().min() == 10
    assert set(svan.label) == {"drone", "background"}
    assert (svan[svan.label == "background"].bg_category.notna()).all()
