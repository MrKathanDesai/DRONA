from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dronedet.data.manifest import split_svanstrom_confusers


def make_rows(recordings_per_category: int = 30) -> pd.DataFrame:
    rows = []
    for category in ("drone", "helicopter", "outdoor_background"):
        label = "drone" if category == "drone" else "background"
        for recording in range(recordings_per_category):
            group = f"svan_{category}_{recording:02d}"
            rows.extend(
                {
                    "source": "svanstrom",
                    "group_id": group,
                    "label": label,
                    "category": category,
                }
                for _ in range(10)
            )
    return pd.DataFrame(rows)


def test_svanstrom_confuser_split_is_grouped_stratified_and_deterministic():
    rows = make_rows()
    first = split_svanstrom_confusers(rows)
    second = split_svanstrom_confusers(rows)
    assert first == second

    assignments = rows[["group_id", "category"]].drop_duplicates()
    assignments["split"] = assignments.group_id.map(first)
    counts = assignments.groupby(["category", "split"]).size().unstack(fill_value=0)
    assert counts.loc["helicopter", ["train", "val", "test"]].tolist() == [18, 6, 6]
    assert counts.loc["outdoor_background", ["train", "val", "test"]].tolist() == [18, 6, 6]
    assert set(assignments.loc[assignments.category == "drone", "split"]) == {"test"}


def test_svanstrom_confuser_split_rejects_missing_negative_category():
    rows = make_rows().query("category != 'helicopter'")
    with pytest.raises(ValueError, match="two outdoor negative categories"):
        split_svanstrom_confusers(rows)
