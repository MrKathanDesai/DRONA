"""Readiness checks, four-state semantics, fusion headline and theme helpers
(src/dronedet/status.py), plus a headless render of the dashboard."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet import status as S
from dronedet.data import rf as R


# --- four states -----------------------------------------------------------
def test_four_states_have_honest_definitions():
    assert set(S.STATE_LABEL) == set(S.STATES) == set(S.STATE_MEANING)
    assert [S.STATE_LABEL[s] for s in S.STATES] == [
        "No Drone", "Known Drone", "Unknown Drone", "Low Confidence"]
    assert "not proof" in S.STATE_MEANING["no drone"]
    assert "not an identification" in S.STATE_MEANING["drone"]
    assert "unlike every drone" in S.STATE_MEANING["unknown drone"]
    assert "review" in S.STATE_MEANING["low confidence"]


def test_unrecognised_state_goes_to_review():
    assert S.state_label("drone") == "Known Drone"
    assert S.state_label("something new") == "Low Confidence"


@pytest.mark.parametrize("states, min_run, expected", [
    (["no drone"] * 5, 2, "no drone"),
    (["no drone", "drone", "drone", "no drone"], 2, "drone"),
    (["drone", "no drone", "drone"], 2, "low confidence"),          # isolated hits
    (["unknown drone", "unknown drone", "drone"], 2, "unknown drone"),
    (["drone", "drone", "unknown drone"], 2, "drone"),
    (["low confidence", "low confidence", "no drone"], 2, "low confidence"),
    (["low confidence", "no drone", "low confidence"], 2, "no drone"),
    (["drone"], 1, "drone"),
    ([], 2, "low confidence"),
])
def test_recording_verdict(states, min_run, expected):
    v = S.recording_verdict(states, min_run)
    assert v.state == expected and v.label == S.STATE_LABEL[expected]
    assert v.reason


def test_low_confidence_windows_do_not_count_towards_an_alarm():
    v = S.recording_verdict(["low confidence", "drone", "low confidence"], 2)
    assert v.longest_drone_run == 1 and v.state == "low confidence"


# --- artifact readiness ----------------------------------------------------
def _cache(directory: Path, n: int, n_index: int | None = None, fingerprint=None):
    directory.mkdir(parents=True)
    X = np.zeros((n, 2, R.DEFAULT.n_bins), np.float32)
    np.save(directory / "X.npy", X)
    pd.DataFrame({"i": range(n if n_index is None else n_index)}).to_parquet(
        directory / "index.parquet")
    (directory / "config.json").write_text(json.dumps({
        "fingerprint": fingerprint or R.DEFAULT.fingerprint(), "shape": list(X.shape)}))


def test_empty_data_dir_reports_missing_and_blocked(tmp_path):
    rows = {a.name: a for a in S.artifact_status(tmp_path)}
    assert rows["Audio detector (tier 4 + uncertainty flags)"].status == S.MISSING
    assert rows["DroneRF PSD feature cache"].status == S.MISSING
    assert rows["RF v2 detector (deployed RF model) + dataset"].status == S.BLOCKED
    assert rows["Fusion rules (sum + learned)"].status == S.BLOCKED
    assert not S.rf_replay_ready(list(rows.values()))
    assert all(a.next_step for a in rows.values() if not a.ready)


def test_legacy_fusion_artifact_still_counts(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fusion.joblib").write_bytes(b"x")
    rows = {a.name: a for a in S.artifact_status(tmp_path)}
    fusion = rows["Fusion rules (sum + learned)"]
    assert fusion.ready and "legacy" in fusion.detail


def test_feature_cache_checks(tmp_path):
    fp, bins = R.DEFAULT.fingerprint(), R.DEFAULT.n_bins
    _cache(tmp_path / "ok", 20)
    assert S.check_rf_feature_cache(tmp_path / "ok", fp, bins)[0] == S.READY
    _cache(tmp_path / "misaligned", 20, n_index=19)
    status, detail = S.check_rf_feature_cache(tmp_path / "misaligned", fp, bins)
    assert status == S.INVALID and "misaligned" in detail
    _cache(tmp_path / "stale", 20, fingerprint="000000000000")
    assert S.check_rf_feature_cache(tmp_path / "stale", fp, bins)[0] == S.INVALID
    assert S.check_rf_feature_cache(tmp_path / "absent", fp, bins)[0] == S.MISSING


# --- fusion headline -------------------------------------------------------
FUSION = ROOT / "reports" / "fusion.json"


@pytest.mark.skipif(not FUSION.exists(), reason="reports/fusion.json not generated")
def test_fusion_rows_keep_denominators_and_learned_is_not_deployed():
    report = json.loads(FUSION.read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in S.fusion_rows(report, "RF-silent drone (autonomous)")}
    deployed = report["deploy"]
    assert deployed != "learned"
    assert rows[deployed]["Role"] == "deployed rule"
    assert rows["learned"]["Role"] == "experimental"
    assert rows["learned"]["Recall"] < rows[deployed]["Recall"]
    assert re.fullmatch(r"[\d,]+/[\d,]+", rows["learned"]["Drones caught"])
    assert re.fullmatch(r"[\d,]+/[\d,]+", rows["learned"]["False alarms"])


# --- theme -----------------------------------------------------------------
def test_normalize_theme_defaults_to_dark():
    assert S.DEFAULT_THEME == "Dark"
    assert S.normalize_theme(None) == "Dark"
    assert S.normalize_theme("purple") == "Dark"
    assert S.normalize_theme("system") == "System"
    assert S.normalize_theme("light") == "Light"
    assert S.normalize_theme(["DARK"]) == "Dark"


def test_theme_script_is_idempotent_and_targets_streamlit_setting():
    js = S.theme_sync_script("dark")
    assert '"Dark"' in js and S.THEME_STORAGE_TEMPLATE in js
    assert "if (current === want) return;" in js          # no reload loop
    assert 'searchParams.set("theme"' in js


def test_installed_streamlit_uses_the_expected_theme_storage():
    """Guards the one Streamlit-internal detail the theme control relies on."""
    st = pytest.importorskip("streamlit")
    js_dir = Path(st.__file__).parent / "static" / "static" / "js"
    bundle = "".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in js_dir.glob("*.js")
                     if "stActiveTheme" in p.read_text(encoding="utf-8", errors="ignore"))
    assert "stActiveTheme-${window.location.pathname}" in bundle
    assert re.search(r"=2,\w+=`stActiveTheme-", bundle), "theme cache version changed"
    for value in ("`System`", "`Light`", "`Dark`"):
        assert value in bundle


def test_streamlit_config_defines_light_and_dark():
    import tomllib
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["theme"]["light"]["base"] == "light"
    assert cfg["theme"]["dark"]["base"] == "dark"


# --- dashboard, rendered headless -------------------------------------------
@pytest.fixture(scope="module")
def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def test_dashboard_renders_without_exceptions(app):
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs][:3] == ["Replay detector", "Project briefing", "Fusion"]


def test_dashboard_shows_states_caveats_and_readiness(app):
    text = _text(app)
    for label in ("No Drone", "Known Drone", "Unknown Drone", "Low Confidence"):
        assert label in text
    assert "not proof that no drone is physically present" in text
    assert "not calibrated probabilities" in text
    assert "No live RF receiver" in text
    assert "not the same flight" in text
    assert "Unpaired data" in text
    assert "not** the recommended rule" in text
    assert "Readiness of models and data" in text
    assert "Learned fusion layer · final multimodal result" not in text


def test_theme_control_present(app):
    radios = [r for r in app.sidebar.radio if r.label == "Theme"]
    assert radios and list(radios[0].options) == ["System", "Light", "Dark"]
    assert radios[0].value == "Dark"                       # dark by default
    assert app.query_params.get("theme") in ("dark", ["dark"])


def test_every_input_source_is_offered(app):
    source = next(r for r in app.main.selectbox if r.label == "Source")  # inside ACOUSTIC-01
    assert list(source.options) == [
        "Svanström outdoor", "DroneAudioDataset", "ITU ARIS Lab",
        "Upload audio or video", "Paste a link (YouTube, …)", "Record with microphone"]


def test_no_dashboard_derived_metrics_left():
    src = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "precision_from_rates" not in src


@pytest.mark.skipif(not (ROOT / "data" / "models" / "detector.joblib").exists()
                    or not (ROOT / "data" / "manifest.parquet").exists(),
                    reason="audio detector or manifest not available")
def test_replay_puts_the_result_first_and_labels_audio_only():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=600)
    at.run()
    clip = next(s for s in at.main.selectbox if s.label == "Recording / clip")
    clip.select_index(1).run()                    # first real Svanström drone recording
    assert not at.exception, [e.value for e in at.exception]
    html = "\n".join(str(m.value) for m in at.markdown)
    assert "Recording-level result · offline replay · not field-validated" in html
    shown = re.search(r'<div class="state">\S+ ([A-Za-z ]+)</div>', html)
    assert shown and shown.group(1) in S.STATE_LABEL.values()
    assert "audio only" in html and "Not a multimodal decision" in html
    assert "Final · audio only" in html and "Not used" in html     # RF tile is off
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "not calibrated probabilities" in captions
    # The console layout: sensor tiles, coverage map, incidents.
    assert "Acoustic-01" in html and "Radio-01" in html and "Sensor coverage" in html
    assert "INC-0001" in html
    # The console-style top bar's badge shows that recording's final state.
    badge = re.search(r'class="threat"[^>]*>([A-Za-z ]+)</span>', html)
    assert badge and badge.group(1) == shown.group(1)


def test_top_bar_is_on_standby_without_a_clip(app):
    html = "\n".join(str(m.value) for m in app.markdown)
    assert 'class="mark">DRONA</span>' in html
    assert re.search(r'class="threat"[^>]*>Standby</span>', html)


@pytest.mark.skipif(not (ROOT / "data" / "models" / "rf_detector.joblib").exists()
                    or not ((ROOT / "data" / "models" / "fusion_models.joblib").exists()
                            or (ROOT / "data" / "models" / "fusion.joblib").exists())
                    or not (ROOT / "data" / "manifest.parquet").exists(),
                    reason="RF v2 / fusion artifacts not built on this machine")
def test_replay_with_rf_shows_fused_final_decision():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=600)
    at.run()
    rf = next(s for s in at.main.selectbox
              if s.label == "Replay recorded RF test windows from")
    drone_rf = next(o for o in rf.options if o.startswith("DroneRF drone"))
    rf.set_value(drone_rf).run()
    method = next(r for r in at.main.radio if r.label == "Fusion rule")
    assert method.value == "sum"                         # deployed rule is the default
    clip = next(s for s in at.main.selectbox if s.label == "Recording / clip")
    clip.select_index(1).run()
    assert not at.exception, [e.value for e in at.exception]
    html = "\n".join(str(m.value) for m in at.markdown)
    assert "Final · audio + RF fused" in html and "RF sensor (recorded replay)" in html
    assert "Not a synchronized multimodal decision" in html
    final = re.search(r'<div class="tile final".*?<div class="v">\S+ ([A-Za-z ]+)</div>',
                      html, re.S)
    assert final and final.group(1) in S.STATE_LABEL.values()


def test_theme_choice_survives_reruns(app):
    for choice in ("Light", "System"):
        radio = next(r for r in app.sidebar.radio if r.label == "Theme")
        radio.set_value(choice).run()
        assert next(r for r in app.sidebar.radio if r.label == "Theme").value == choice
        assert app.query_params.get("theme") in (choice.lower(), [choice.lower()])
        app.run()                                             # an ordinary rerun
        assert next(r for r in app.sidebar.radio if r.label == "Theme").value == choice
    assert not app.exception
