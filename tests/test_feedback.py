import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronedet import feedback as FB


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "FEED", tmp_path / "feedback")
    monkeypatch.setattr(FB, "CLIPS", tmp_path / "feedback" / "clips")
    monkeypatch.setattr(FB, "LABELS", tmp_path / "feedback" / "labels.csv")
    monkeypatch.setattr(FB, "VERSIONS", tmp_path / "versions")
    monkeypatch.setattr(FB, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(FB, "BASE_POINTER", tmp_path / "base.json")
    return tmp_path


def test_save_clip_records_label_and_audio(store):
    FB.save_clip("INC-0001", "drone", np.zeros(3 * FB.SR, np.float32), "Microphone")
    FB.save_clip("INC-0002", "not_drone", np.zeros(2 * FB.SR, np.float32))
    labels = FB.load_labels()
    assert list(labels.label) == ["drone", "not_drone"]
    assert (FB.CLIPS / "INC-0001.wav").exists()
    st = FB.status()
    assert st["labelled"] == {"drone": 1, "not_drone": 1} and st["new"] == 2
    assert st["version"].startswith("v0")


def test_relabelling_replaces_and_bad_input_is_refused(store):
    FB.save_clip("INC-0001", "drone", np.zeros(FB.SR, np.float32))
    FB.save_clip("INC-0001", "not_drone", np.zeros(FB.SR, np.float32))
    assert list(FB.load_labels().label) == ["not_drone"]
    with pytest.raises(ValueError):
        FB.save_clip("x", "maybe", np.zeros(FB.SR, np.float32))
    with pytest.raises(ValueError):
        FB.save_clip("x", "drone", np.zeros(100, np.float32))


def test_unsafe_ids_cannot_escape_the_clip_folder(store):
    row = FB.save_clip("../../etc/passwd", "drone", np.zeros(FB.SR, np.float32))
    assert "/" not in row["id"] and (FB.CLIPS / f"{row['id']}.wav").exists()


def test_rollback_walks_back_through_promoted_versions(store):
    FB.REGISTRY.write_text(json.dumps({"current": "v2", "history": [
        {"version": "v1", "promoted": True, "clips": []},
        {"version": None, "promoted": False, "clips": []},
        {"version": "v2", "promoted": True, "clips": []}]}))
    assert FB.rollback() == "v1"
    assert FB.current_model_path() == FB.VERSIONS / "v1.joblib"
    assert FB.rollback() is None
    assert FB.current_model_path() == FB.ORIGINAL


def test_retrain_refuses_without_new_labels(store):
    assert FB.retrain(enc=None, champion=None)["promoted"] is False
