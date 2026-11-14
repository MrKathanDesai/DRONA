"""What the dashboard is allowed to claim: artifact readiness, the meaning of
the four output states, a recording-level verdict, and the fusion headline.

Everything here is pure Python (no Streamlit), so the wording and the checks
the professor-facing pages rely on are unit-tested (tests/test_status.py).

Readiness is read from files on disk, never from a Markdown status line: a
report in reports/ can be older than, or exist without, the model it describes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# The four states
# ---------------------------------------------------------------------------
STATES = ("no drone", "drone", "unknown drone", "low confidence")
STATE_LABEL = {
    "no drone": "No Drone",
    "drone": "Known Drone",
    "unknown drone": "Unknown Drone",
    "low confidence": "Low Confidence",
}
# Plain-language definitions shown next to every result. Kept here so the
# dashboard, the docs and the tests use the same sentences.
STATE_MEANING = {
    "no drone": (
        "The available sensor evidence is below the decision threshold. "
        "This is not proof that no drone is physically present."
    ),
    "drone": (
        "Drone-like evidence is above the threshold and does not trigger the "
        "unknown-signature flag. It means 'resembles the drones in training', "
        "not an identification of the exact drone model."
    ),
    "unknown drone": (
        "Drone-like evidence is above the threshold, but the signature is "
        "unlike every drone seen in training."
    ),
    "low confidence": (
        "The evidence is ambiguous or near the review boundary, or the input "
        "is unlike the training data. A person should review it."
    ),
}
DRONE_STATES = ("drone", "unknown drone")


def state_label(state: str) -> str:
    """Internal state -> display label. Unknown values are shown as Low
    Confidence: an unrecognised state must go to a person, not be hidden."""
    return STATE_LABEL.get(str(state), "Low Confidence")


def longest_run(flags) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


@dataclass(frozen=True)
class Verdict:
    state: str            # one of STATES
    reason: str
    windows: int
    drone_windows: int    # windows whose own state is Known or Unknown Drone
    unknown_windows: int
    low_windows: int
    longest_drone_run: int
    min_run: int

    @property
    def label(self) -> str:
        return state_label(self.state)


def recording_verdict(states, min_run: int = 2) -> Verdict:
    """Summarise per-window states into one recording-level result.

    Same rule as the operations console (console/server.py, Session): only
    windows whose state is Known/Unknown Drone count as hits, so a window
    routed to review does not count towards an alarm.

    - Known / Unknown Drone: at least `min_run` consecutive drone windows.
      Unknown when most of the drone windows carry the unknown-signature flag.
    - Low Confidence: some drone windows but no qualifying run, or `min_run`
      consecutive low-confidence windows.
    - No Drone: otherwise.
    """
    s = [str(x) for x in states]
    n = len(s)
    if min_run < 1:
        raise ValueError("min_run must be at least 1")
    drone = [x in DRONE_STATES for x in s]
    low = [x == "low confidence" for x in s]
    n_drone, n_unknown, n_low = sum(drone), s.count("unknown drone"), sum(low)
    run = longest_run(drone)
    if n == 0:
        return Verdict("low confidence", "no windows were scored", 0, 0, 0, 0, 0, min_run)
    if run >= min_run:
        if n_unknown * 2 > n_drone:
            state = "unknown drone"
            reason = (f"{run} consecutive drone windows (need {min_run}); "
                      f"{n_unknown} of {n_drone} drone windows have an unfamiliar signature")
        else:
            state = "drone"
            reason = f"{run} consecutive drone windows (need {min_run})"
    elif n_drone:
        state = "low confidence"
        reason = (f"{n_drone} drone window(s), but never {min_run} in a row; "
                  "isolated hits need review")
    elif longest_run(low) >= min_run:
        state = "low confidence"
        reason = f"{longest_run(low)} consecutive low-confidence windows"
    else:
        state = "no drone"
        reason = "no window reached a drone state"
    return Verdict(state, reason, n, n_drone, n_unknown, n_low, run, min_run)


def alarm_episodes(states, audio_states=None, rf_states=None, min_run: int = 2,
                   hold: int = 3) -> list[dict]:
    """Incidents in a sequence of window states, with the operations console's
    rule (console/server.py, Session): an incident opens once `min_run` drone
    windows (Known or Unknown) come in a row, and stays open until `hold`
    windows pass without one.

    Each incident: first and last window index, whether any window in it was an
    unknown drone, and which sensors ('acoustic', 'radio') flagged a drone in it,
    read from that sensor's own per-window states."""
    if min_run < 1:
        raise ValueError("min_run must be at least 1")
    s = [str(x) for x in states]
    episodes, run, cur, quiet = [], 0, None, 0
    for i, state in enumerate(s):
        drone = state in DRONE_STATES
        run = run + 1 if drone else 0
        if cur is None:
            if run >= min_run:
                cur, quiet = {"start": i - run + 1, "end": i}, 0
        elif drone:
            cur["end"], quiet = i, 0
        else:
            quiet += 1
            if quiet >= hold:
                episodes.append(cur)
                cur = None
    if cur is not None:
        episodes.append(cur)
    for ep in episodes:
        span = range(ep["start"], ep["end"] + 1)
        ep["unknown"] = any(s[i] == "unknown drone" for i in span)
        sources = []
        if audio_states is not None and any(str(audio_states[i]) in DRONE_STATES for i in span):
            sources.append("acoustic")
        if rf_states is not None and any(str(rf_states[i]) in DRONE_STATES for i in span):
            sources.append("radio")
        if not sources:      # only the fused evidence crossed: every sensor in use contributed
            sources = ["acoustic"] + (["radio"] if rf_states is not None else [])
        ep["sources"] = sources
    return episodes


# ---------------------------------------------------------------------------
# Artifact readiness
# ---------------------------------------------------------------------------
READY, MISSING, BLOCKED, INVALID = "ready", "missing", "blocked", "invalid"


@dataclass(frozen=True)
class Artifact:
    name: str
    status: str           # READY / MISSING / BLOCKED / INVALID
    detail: str
    paths: tuple[str, ...]
    next_step: str = ""

    @property
    def ready(self) -> bool:
        return self.status == READY


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root.parent)).replace("\\", "/")
    except ValueError:
        return str(p)


def check_rf_feature_cache(directory: Path, expected_fingerprint: str | None = None,
                           expected_bins: int | None = None) -> tuple[str, str]:
    """(status, detail) for a prepared RF feature cache: files present, X rows
    equal index rows, config shape and fingerprint match. Uses mmap, so it
    does not read the array into memory."""
    x, idx, cfg = directory / "X.npy", directory / "index.parquet", directory / "config.json"
    missing = [p.name for p in (x, idx, cfg) if not p.is_file()]
    if missing:
        return MISSING, f"missing {', '.join(missing)}"
    try:
        import pyarrow.parquet as pq
        X = np.load(x, mmap_mode="r")
        n_index = pq.ParquetFile(idx).metadata.num_rows
        config = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as e:                           # corrupt or unreadable
        return INVALID, f"could not read the cache: {type(e).__name__}: {e}"
    if len(X) != n_index:
        return INVALID, f"misaligned: {len(X)} feature rows vs {n_index} index rows"
    if list(config.get("shape", [])) != list(X.shape):
        return INVALID, f"config.json shape {config.get('shape')} != X.npy {list(X.shape)}"
    if expected_fingerprint and config.get("fingerprint") != expected_fingerprint:
        return INVALID, (f"feature fingerprint {config.get('fingerprint')} != "
                         f"current config {expected_fingerprint}; re-run prepare_rf.py")
    if expected_bins and X.shape[-1] != expected_bins:
        return INVALID, f"{X.shape[-1]} PSD bins, expected {expected_bins}"
    return READY, f"{X.shape[0]:,} windows x {X.shape[1]} bands x {X.shape[2]} bins"


def artifact_status(data_dir: Path) -> list[Artifact]:
    """Readiness of every model/data artifact the dashboard depends on."""
    from dronedet.data import rf as R

    d = Path(data_dir)
    models = d / "models"
    rows: list[Artifact] = []

    def present(name, paths, ok_detail, next_step, blocked_by=None):
        paths = tuple(paths)
        if all(p.exists() for p in paths):
            rows.append(Artifact(name, READY, ok_detail, tuple(_rel(p, d) for p in paths)))
        else:
            gone = [p.name for p in paths if not p.exists()]
            status = BLOCKED if blocked_by else MISSING
            detail = f"missing {', '.join(gone)}" + (f"; {blocked_by}" if blocked_by else "")
            rows.append(Artifact(name, status, detail, tuple(_rel(p, d) for p in paths),
                                 next_step))

    present("Audio detector (tier 4 + uncertainty flags)",
            [models / "detector.joblib", d / "pretrained" / "MobileNetV1_mAP=0.389.pth"],
            "trained; used by the replay page (encoder weights present)",
            "python scripts/eval_uncertainty.py")

    raw = d / "raw_dronerf"
    archives = sorted(raw.glob("*.rar")) if raw.is_dir() else []
    if len(archives) == 23:
        rows.append(Artifact("DroneRF raw archives", READY,
                             "23 of 23 archives present (checksums: prepare_rf.py --verify)",
                             (_rel(raw, d),)))
    else:
        rows.append(Artifact("DroneRF raw archives", MISSING,
                             f"{len(archives)} of 23 archives present", (_rel(raw, d),),
                             "python scripts/prepare_rf.py --download-only"))

    status, detail = check_rf_feature_cache(d / "rf_features", R.DEFAULT.fingerprint(),
                                            R.DEFAULT.n_bins)
    rows.append(Artifact("DroneRF PSD feature cache", status, detail,
                         (_rel(d / "rf_features", d),),
                         "" if status == READY else "python scripts/prepare_rf.py --process-only"))

    present("RF v1 forest (DroneRF only; calls noise 'drone')",
            [models / "rf_forest.joblib"], "trained; diagnostic only, not a usable detector",
            "python scripts/train_rf.py")
    present("RF calibration / Phantom probe (rf_trust)",
            [models / "rf_trust.joblib"], "trained; used by the RF research replay panel",
            "python scripts/eval_rf_trust.py")

    tampere = d / "raw_tampere"
    n_tampere = len(list(tampere.glob("*_2G*.bin"))) if tampere.is_dir() else 0
    if n_tampere:
        rows.append(Artifact("Tampere RF recordings (second receiver)", READY,
                             f"{n_tampere} 2.4 GHz files", (_rel(tampere, d),)))
    else:
        rows.append(Artifact("Tampere RF recordings (second receiver)", MISSING,
                             "no *_2G*.bin files; 5.3 GB download from Zenodo 4264467",
                             (_rel(tampere, d),),
                             "python scripts/download_data.py --only-extra"))
    tampere_note = None if n_tampere else "needs the Tampere recordings first"
    present("Tampere PSD cache", [d / "rf_features_tampere" / "X.npy"],
            "built", "python scripts/eval_rf_tampere.py", tampere_note)
    present("RF v2 detector (deployed RF model) + dataset",
            [models / "rf_detector.joblib", d / "rf_features_v2" / "X.npy"],
            "trained; scores the RF replay feed", "python scripts/train_rf_v2.py", tampere_note)

    rf_v2_note = None if (models / "rf_detector.joblib").exists() else "needs RF v2 first"
    fm, legacy = models / "fusion_models.joblib", models / "fusion.joblib"
    if fm.exists():
        rows.append(Artifact("Fusion rules (sum + learned)", READY,
                             "both rules saved; replay defaults to the deployed rule",
                             (_rel(fm, d),)))
    elif legacy.exists():
        rows.append(Artifact("Fusion rules (sum + learned)", READY,
                             "legacy fusion.joblib only: evidence sum; learned rule not saved "
                             "(re-run eval_fusion.py to add it)", (_rel(legacy, d),)))
    else:
        rows.append(Artifact("Fusion rules (sum + learned)",
                             BLOCKED if rf_v2_note else MISSING,
                             "missing fusion_models.joblib and fusion.joblib"
                             + (f"; {rf_v2_note}" if rf_v2_note else ""),
                             (_rel(fm, d), _rel(legacy, d)), "python scripts/eval_fusion.py"))
    return rows


def rf_replay_ready(rows: list[Artifact]) -> bool:
    by = {r.name: r for r in rows}
    return (by["RF v2 detector (deployed RF model) + dataset"].ready
            and by["Fusion rules (sum + learned)"].ready)


# ---------------------------------------------------------------------------
# Fusion headline (from reports/fusion.json, never hard-coded)
# ---------------------------------------------------------------------------
FUSION_METHOD_LABEL = {
    "audio only": "Audio only",
    "RF only": "RF only",
    "sum": "Evidence sum",
    "learned": "Learned fusion layer",
    "OR": "OR rule",
    "AND": "AND rule",
}
HEADLINE_SCENARIOS = ("both sensors", "RF-silent drone (autonomous)")


def fusion_rows(report: dict, scenario: str,
                methods=("audio only", "RF only", "sum", "learned", "OR", "AND")) -> list[dict]:
    """One row per method with denominators, for a scenario of fusion.json."""
    res = report["scenarios"][scenario]
    n_pos = report.get("pairs_per_class")
    deployed = report.get("deploy")
    out = []
    for m in methods:
        if m not in res:
            continue
        r = res[m]
        conf = r.get("confusion")
        tp = conf[1][1] if conf else None
        positives = (conf[1][0] + conf[1][1]) if conf else n_pos
        out.append({
            "method": m,
            "Method": FUSION_METHOD_LABEL.get(m, m),
            "Role": ("deployed rule" if m == deployed else
                     "experimental" if m == "learned" else "baseline"),
            "Recall": r.get("recall"),
            "Drones caught": f"{tp:,}/{positives:,}" if tp is not None else "n/a",
            "False-positive rate": r.get("fpr"),
            "False alarms": (f"{r['false_alarms']:,}/{r['negatives']:,}"
                             if "false_alarms" in r and "negatives" in r else "n/a"),
            "PR-AUC": r.get("pr_auc"),
        })
    return out


# ---------------------------------------------------------------------------
# Theme preference (Streamlit's own System / Light / Dark setting)
# ---------------------------------------------------------------------------
THEMES = ("System", "Light", "Dark")
DEFAULT_THEME = "Dark"          # the dashboard opens dark unless ?theme= says otherwise
# Streamlit 1.67 keeps the Settings-menu choice in localStorage under this key
# (frontend `stActiveTheme-${pathname}-v2`, values "System" / "Light" / "Dark").
# tests/test_status.py checks the installed frontend still uses it.
THEME_STORAGE_TEMPLATE = "stActiveTheme-${window.location.pathname}-v2"


def normalize_theme(value, default: str = DEFAULT_THEME) -> str:
    """Query-parameter / stored value -> one of THEMES; missing or unknown
    values give `default` (Dark)."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    v = str(value or "").strip().lower()
    return {"system": "System", "light": "Light", "dark": "Dark"}.get(v, default)


def theme_sync_script(choice: str) -> str:
    """JavaScript that makes Streamlit's own theme setting equal `choice`.

    Streamlit applies its theme to every native widget, table, chart and the
    sidebar, and "System" follows the browser's prefers-color-scheme live.
    It reads the stored choice when the page loads, so the script stores it,
    records it in the URL (?theme=...) so the app's control agrees after the
    reload, and reloads once. If the stored choice already matches, it does
    nothing, so it cannot loop."""
    choice = normalize_theme(choice)
    key = THEME_STORAGE_TEMPLATE
    return f"""
<script>
(() => {{
  const want = {json.dumps(choice)};
  const key = `{key}`;
  let current = "System";
  try {{
    const raw = window.localStorage.getItem(key);
    if (raw) current = JSON.parse(raw);
  }} catch (e) {{ current = "System"; }}
  if (current === want) return;
  try {{ window.localStorage.setItem(key, JSON.stringify(want)); }} catch (e) {{ return; }}
  const url = new URL(window.location.href);
  url.searchParams.set("theme", want.toLowerCase());
  window.location.replace(url.toString());
}})();
</script>
"""
