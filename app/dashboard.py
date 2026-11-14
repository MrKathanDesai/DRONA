"""Counter-drone audio/RF project dashboard and audio monitor.

    streamlit run app/dashboard.py

Summarizes model results and pipeline status, then replays an audio recording
with per-window predictions and Grad-CAM. An RF feed (held-out RF windows scored
by the RF v2 detector) can be replayed beside it and fused with the audio score
(data/models/fusion.joblib). Audio and RF were never recorded together, so the
fused view is a what-if; see reports/fusion.md.
"""
from __future__ import annotations

import io
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from dronedet import live as L
from dronedet import media as M
from dronedet import status as S
from dronedet.eval import calibration as RF_CALIBRATION
from dronedet.eval import open_set as RF_OPEN_SET
from dronedet.models import explain as X
from dronedet.models import rf as RF_MODEL
from dronedet.paths import DATA_DIR

REPORTS = ROOT / "reports"
STATE_COLOR = {"drone": "#d6452a", "unknown drone": "#7a4fd6",
               "low confidence": "#d99a0b", "no drone": "#8a8984"}
STATE_ICON = {"drone": "🔴", "unknown drone": "🟣", "low confidence": "🟡", "no drone": "⚪"}
FINAL_STATE_LABEL = S.STATE_LABEL
# Mid-tone so a reference line / marker reads on light and dark charts alike.
CHART_REFERENCE = "#8a8f98"

st.set_page_config(page_title="Skyguard · Counter-drone Lab", page_icon="🛰️",
                   layout="wide")

# Colour-neutral CSS. Streamlit owns the palette (.streamlit/config.toml has a
# light and a dark theme; the sidebar Theme control picks System / Light /
# Dark). Every colour below is either inherited or mixed with `currentColor`
# (the active theme's text colour), so the same rules read on both
# backgrounds: an accent mixed with dark text gets darker on white, mixed with
# light text gets lighter on navy. Selectors use Streamlit's stable
# data-testid attributes only (checked against streamlit 1.67).
APP_CSS = """
<style>
:root {
    --dd-accent: #0e7490;
    --dd-amber: #b45309;
    --dd-line: color-mix(in srgb, currentColor 16%, transparent);
    --dd-soft: color-mix(in srgb, currentColor 4%, transparent);
    --dd-muted: color-mix(in srgb, currentColor 72%, transparent);
}
.block-container { max-width: 1440px; padding-top: 1.8rem; }
h1, h2, h3 { letter-spacing: -.02em; }
[data-testid="stMetric"] {
    background: var(--dd-soft); border: 1px solid var(--dd-line);
    border-radius: 12px; padding: 12px 14px;
}
.hero {
    padding: clamp(18px, 3vw, 32px); margin: 0 0 1rem;
    border: 1px solid var(--dd-line); border-radius: 18px;
    background: color-mix(in srgb, var(--dd-accent) 9%, transparent);
}
.eyebrow, .panel .tag {
    color: color-mix(in srgb, var(--dd-accent) 62%, currentColor);
    text-transform: uppercase; letter-spacing: .14em;
    font: 700 .72rem/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
}
.hero h1 { margin: .3rem 0 .4rem; font-size: clamp(1.8rem, 4.5vw, 2.8rem); }
.hero p { color: var(--dd-muted); max-width: 820px; margin: 0; line-height: 1.6; }
.pill {
    display: inline-block; border: 1px solid var(--dd-line); border-radius: 999px;
    padding: 4px 10px; margin: 12px 6px 0 0; font-size: .78rem; background: var(--dd-soft);
}
.panel {
    height: 100%; padding: 16px; border-radius: 14px;
    background: var(--dd-soft); border: 1px solid var(--dd-line);
}
.panel h3 { margin: 6px 0 4px; font-size: 1.08rem; }
.panel p { color: var(--dd-muted); font-size: .92rem; line-height: 1.5; margin: 0; }
.panel.pending { border-color: color-mix(in srgb, var(--dd-amber) 55%, transparent); }
.panel.pending .tag { color: color-mix(in srgb, var(--dd-amber) 70%, currentColor); }
.callout {
    padding: 13px 16px; margin: 8px 0 16px; line-height: 1.55; border-radius: 10px;
    background: color-mix(in srgb, var(--dd-amber) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--dd-amber) 40%, transparent);
    border-left: 5px solid var(--dd-amber);
}
.stage-flow {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px; margin: 10px 0 18px;
}
.stage { padding: 14px; border: 1px solid var(--dd-line); border-radius: 12px;
         background: var(--dd-soft); }
.stage small { color: color-mix(in srgb, var(--dd-accent) 62%, currentColor);
               letter-spacing: .08em; text-transform: uppercase; font-weight: 700; }
.stage strong { display: block; margin: 6px 0; font-size: 1.02rem; }
.stage span { color: var(--dd-muted); font-size: .9rem; line-height: 1.5; }
.result {
    padding: 16px 20px; margin: 4px 0 14px; border-radius: 14px;
    border: 1px solid color-mix(in srgb, var(--state) 50%, transparent);
    border-left: 10px solid var(--state);
    background: color-mix(in srgb, var(--state) 11%, transparent);
}
.result .kicker { font-size: .78rem; letter-spacing: .1em; text-transform: uppercase;
                  color: var(--dd-muted); font-weight: 700; }
.result .state { font-size: clamp(1.7rem, 4vw, 2.4rem); font-weight: 800;
                 line-height: 1.15; margin: .15rem 0 .35rem; }
.result p { margin: .2rem 0; line-height: 1.5; }
.result .meaning { color: var(--dd-muted); }
.sensor { padding: 12px 14px; border-radius: 12px; border: 1px solid var(--dd-line);
          background: var(--dd-soft); height: 100%; }
.sensor h4 { margin: 0 0 4px; font-size: 1rem; }
.sensor p { margin: .15rem 0; font-size: .92rem; line-height: 1.45; }
.sensor .muted { color: var(--dd-muted); }
.badge { display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .06em;
         text-transform: uppercase; border-radius: 6px; padding: 2px 7px;
         border: 1px solid var(--dd-line); margin-right: 4px; }
.badge.ok { border-color: color-mix(in srgb, #15803d 60%, transparent);
            color: color-mix(in srgb, #15803d 70%, currentColor); }
.badge.warn { border-color: color-mix(in srgb, var(--dd-amber) 60%, transparent);
              color: color-mix(in srgb, var(--dd-amber) 70%, currentColor); }
.badge.off { color: var(--dd-muted); }
.flow { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; margin: 6px 0 14px; }
.flow .op { align-self: center; font-size: 1.8rem; font-weight: 800; color: var(--dd-muted);
            padding: 0 2px; }
.tile { flex: 1 1 210px; padding: 14px 16px; border-radius: 14px;
        border: 2px solid color-mix(in srgb, var(--state) 60%, transparent);
        background: color-mix(in srgb, var(--state) 12%, transparent); }
.tile.final { flex: 1.3 1 240px; border-width: 3px;
              background: color-mix(in srgb, var(--state) 20%, transparent); }
.tile.off { border-style: dashed; background: var(--dd-soft); }
.tile .k { font-size: .74rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
           color: var(--dd-muted); }
.tile .v { font-size: 1.45rem; font-weight: 800; margin: 2px 0 8px; line-height: 1.2; }
.tile.final .v { font-size: 1.8rem; }
.tile p { margin: .2rem 0 0; font-size: .86rem; line-height: 1.4; color: var(--dd-muted); }
.gauge { position: relative; height: 12px; border-radius: 6px; margin: 4px 0 6px;
         background: color-mix(in srgb, currentColor 12%, transparent); }
.gauge span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }
.gauge i { position: absolute; top: -4px; bottom: -4px; width: 3px; border-radius: 2px;
           background: currentColor; transform: translateX(-1px); }
/* Visible keyboard focus on everything a keyboard can reach. */
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible,
[role="radio"]:focus-visible, input:focus-visible, summary:focus-visible {
    outline: 3px solid color-mix(in srgb, var(--dd-accent) 70%, currentColor) !important;
    outline-offset: 2px;
}
[data-testid="stTabs"] button { font-weight: 650; }
@media (max-width: 900px) { .stage-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) {
    .block-container { padding: 1rem .75rem 2rem; }
    .stage-flow { grid-template-columns: 1fr; }
    [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; flex-wrap: nowrap; }
    [data-testid="stTabs"] button { white-space: nowrap; font-size: .85rem; }
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


def theme_control():
    """Sidebar System / Light / Dark control.

    It drives Streamlit's own theme setting (the same one as ⋮ -> Settings),
    so native widgets, tables, charts and the sidebar change together and
    "System" follows the browser's colour scheme. The choice lives in the URL
    (?theme=light), so it survives reruns, reloads and can be bookmarked."""
    if "theme_pref" not in st.session_state:
        st.session_state.theme_pref = S.normalize_theme(st.query_params.get("theme"))
    st.sidebar.radio(
        "Theme", S.THEMES, key="theme_pref", horizontal=True,
        help="System follows your operating system / browser setting. "
             "Changing it reloads the page once. Also available under ⋮ → Settings.",
    )
    choice = st.session_state.theme_pref
    if st.session_state.get("theme_synced") != choice:
        st.session_state.theme_synced = choice
        # Always in the URL, so a reload keeps System instead of the Dark default.
        st.query_params["theme"] = choice.lower()
        st.html(S.theme_sync_script(choice), unsafe_allow_javascript=True)


@st.cache_resource(show_spinner="Loading encoder and detector ...")
def detector() -> L.LiveDetector:
    return L.LiveDetector("detector")


@st.cache_resource(show_spinner="Loading prepared RF feature cache ...")
def rf_replay_resources():
    features_dir = DATA_DIR / "rf_features"
    X_rf = np.load(features_dir / "X.npy", mmap_mode="r")
    index = pd.read_parquet(features_dir / "index.parquet").reset_index(drop=True)
    config = json.loads((features_dir / "config.json").read_text(encoding="utf-8"))
    if len(X_rf) != len(index):
        raise ValueError(
            f"RF replay cache is misaligned: {len(X_rf)} feature rows, {len(index)} index rows"
        )
    artifact_path = DATA_DIR / "models" / "rf_trust.joblib"
    artifact = joblib.load(artifact_path) if artifact_path.is_file() else None
    if artifact is not None and artifact.get("feature_fingerprint") != config.get("fingerprint"):
        raise ValueError("RF replay model and feature cache fingerprints do not match")
    return X_rf, index, artifact


@st.cache_data(show_spinner="Scoring windows ...", max_entries=32)
def analyse(key: str, audio: bytes, hop: float):
    sig = L.load(io.BytesIO(audio))
    df, clips = detector().analyse(sig, hop)
    return df, np.stack(clips), sig


@st.cache_data(show_spinner="Computing Grad-CAM ...", max_entries=256)
def cam_for(key: str, i: int, clip: np.ndarray, layer: int):
    return detector().explain(clip, layer)


def to_wav(y: np.ndarray, sr: int = L.SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def timeline(df: pd.DataFrame, thr: float, upto: int | None = None,
             selected: int | None = None):
    d = df if upto is None else df.iloc[:upto + 1]
    d = d[["t", "p", "state", "reason", "tags"]].assign(
        state_label=d.state.map(S.state_label))
    base = alt.Chart(d).encode(x=alt.X("t:Q", title="time (s)",
                                       scale=alt.Scale(domain=[0, float(df.t.max()) + 1])))
    bars = base.mark_bar(size=8, cornerRadiusTopLeft=2,
                         cornerRadiusTopRight=2).encode(
        y=alt.Y("p:Q", title="audio model score (not a probability)",
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("state_label:N", title="audio state",
                        scale=alt.Scale(domain=[FINAL_STATE_LABEL[k] for k in STATE_COLOR],
                                        range=list(STATE_COLOR.values()))),
        tooltip=[alt.Tooltip("t:Q", title="time (s)"),
                 alt.Tooltip("p:Q", format=".3f", title="score"),
                 alt.Tooltip("state_label:N", title="state"), "reason", "tags"])
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(
        strokeDash=[4, 4], color=CHART_REFERENCE).encode(y="thr:Q")
    layers = [bars, rule]
    if selected is not None and selected < len(d):
        sel = alt.Chart(d.iloc[[selected]]).mark_point(
            shape="triangle-down", size=120, filled=True, color=CHART_REFERENCE).encode(
            x="t:Q", y=alt.value(0))
        layers.append(sel)
    return alt.layer(*layers).properties(height=220)


def cam_figure(M: np.ndarray, cam: np.ndarray):
    hz = X.mel_axis_hz(M.shape[1])
    top = int(np.searchsorted(hz, 8000)) + 1          # audio is band-limited to 8 kHz
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    ext = [0, 1, 0, top]
    axes[0].imshow(M[:, :top].T, origin="lower", aspect="auto", cmap="magma", extent=ext)
    axes[0].set_title("log-mel (encoder input)", fontsize=10)
    axes[1].imshow(M[:, :top].T, origin="lower", aspect="auto", cmap="gray", extent=ext)
    axes[1].imshow(cam[:, :top].T, origin="lower", aspect="auto", cmap="inferno",
                   alpha=0.55, extent=ext, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: evidence for 'drone'", fontsize=10)
    ticks = [b for b in (0, 8, 16, 24, 32, 40) if b < top]
    axes[0].set_yticks(ticks, [f"{hz[b]:.0f}" for b in ticks])
    axes[0].set_ylabel("Hz")
    for ax in axes:
        ax.set_xlabel("time in window (s)")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def read_report(name: str) -> dict | None:
    path = REPORTS / name
    if not path.exists():
        return None                    # main() lists every missing report once
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_md_report(name: str, title: str, expanded: bool = False):
    """Show a generated Markdown report (reports/<name>) in an expander."""
    path = REPORTS / name
    with st.expander(title, expanded=expanded):
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.caption(f"`reports/{name}` not generated yet.")


NO_RF = "None (audio only)"


@st.cache_resource(show_spinner="Loading RF detector and fusion ...")
def rf_feed():
    try:
        feed_factory = getattr(L, "RFFeed", None)
        if feed_factory is None:
            importlib.invalidate_caches()
            importlib.reload(L)
            feed_factory = getattr(L, "RFFeed", None)
        if feed_factory is None:
            return (
                "The loaded live module has no RF replay support. Stop the old "
                "Streamlit process and restart it from this project directory."
            )
        return feed_factory()
    except FileNotFoundError as e:
        return str(e)
    except (ValueError, KeyError, EOFError, ModuleNotFoundError) as e:
        return f"RF/fusion artifacts could not be loaded ({type(e).__name__}: {e})."
    except AttributeError as e:
        return (
            "The live module is stale or incompatible. Restart Streamlit from the "
            f"project directory. Details: {e}"
        )


def rf_choice(ui=None):
    ui = ui or st.sidebar
    ui.header("RF feed")
    feed = rf_feed()
    if isinstance(feed, str):
        ui.info("RF replay unavailable: the RF detector or fusion files it needs are "
                        "missing or unreadable on this computer, so results here are audio "
                        "only. See *Project briefing → Readiness* for what to build.")
        with ui.expander("Technical detail"):
            st.caption(feed)
        return None, NO_RF, None
    src = ui.selectbox("Replay recorded RF test windows from", [NO_RF, *feed.sources],
                               help="Recorded, held-out RF windows of this class (not a live "
                                    "receiver), one 25 ms snapshot per audio window. Audio and "
                                    "RF were never recorded together, so this is a what-if.")
    for warning in getattr(feed, "warnings", []):
        ui.warning(warning)
    deployed = getattr(feed, "deployed", "sum")
    # The deployed rule first, so it is the default; the learned layer is
    # offered for the ablation but labelled for what reports/fusion.md shows.
    methods = sorted(feed.fusions, key=lambda m: m != deployed)
    labels = {
        "sum": "Evidence sum",
        "learned": "Learned layer (experimental: misses RF-silent drones)",
    }
    if len(methods) > 1:
        method = ui.radio(
            "Fusion rule", methods,
            format_func=lambda m: labels.get(m, m) + (" · deployed" if m == deployed else ""),
            help="Evidence sum is the deployed rule. The learned layer nearly ignores "
                 "audio when RF is quiet; see the RF-silent scenario in the Fusion tab "
                 "(reports/fusion.md).",
        )
    else:
        method = methods[0]
        ui.caption(f"Fusion rule: {labels.get(method, method)} "
                           "(only rule saved; re-run eval_fusion.py for the learned layer)")
    return feed, src, method


def fused_timeline(df: pd.DataFrame, thr: float):
    base = alt.Chart(df).encode(x=alt.X("t:Q", title="time (s)"))
    bars = base.mark_bar(size=8).encode(
        y=alt.Y("fused:Q", title="fused evidence (log-likelihood ratio)"),
        color=alt.Color("fused_state_label:N", title="fused state (replay)",
                        scale=alt.Scale(domain=[FINAL_STATE_LABEL[k] for k in STATE_COLOR],
                                        range=list(STATE_COLOR.values()))),
        tooltip=["t", alt.Tooltip("llr_audio:Q", format=".2f"),
                 alt.Tooltip("llr_rf:Q", format=".2f"),
                 alt.Tooltip("fused:Q", format=".2f"),
                 "fused_state_label", "fused_reason"])
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(
        strokeDash=[4, 4], color=CHART_REFERENCE).encode(y="thr:Q")
    return alt.layer(bars, rule).properties(height=200)


def rf_figure(psd: np.ndarray):
    """One RF window: power spectrum of both DroneRF bands (2.400-2.480 GHz)."""
    f = np.linspace(0, 20, psd.shape[1])
    fig, ax = plt.subplots(figsize=(6, 2.4))
    for b, (label, color) in enumerate((("2.400-2.440 GHz", "#2a78d6"),
                                        ("2.440-2.480 GHz", "#eb6834"))):
        ax.plot(f, psd[b] - np.median(psd[b]), lw=0.8, color=color, label=label)
    ax.set_xlabel("offset from band centre (MHz)")
    ax.set_ylabel("dB above median")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def hero():
    st.markdown(
        """
        <div class="hero" role="banner">
          <div class="eyebrow">Research prototype · audio + RF drone detection</div>
          <h1>Skyguard counter-drone lab</h1>
          <p>Detects drones from their sound (1-second microphone windows) and, as a
          research experiment, from their 2.4 GHz radio link, then reports one of four
          states: No Drone, Known Drone, Unknown Drone or Low Confidence. Everything
          here runs on recorded datasets; nothing is connected to a live microphone
          array or RF receiver, and nothing is field-validated.</p>
          <span class="pill">Offline replay of recordings</span>
          <span class="pill">No live RF receiver</span>
          <span class="pill">Human review required</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(tag: str, title: str, description: str, state: str = "ready"):
    st.markdown(
        f'<div class="panel {state}"><div class="tag">{tag}</div>'
        f"<h3>{title}</h3><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def percent(value: float) -> str:
    return f"{value:.1%}"


def render_report_downloads(names: list[str], section: str):
    with st.expander("Download detailed evaluation data"):
        st.caption("These are the complete machine-readable reports used by this dashboard.")
        columns = st.columns(3)
        names = [n for n in names if (REPORTS / n).exists()]
        for i, name in enumerate(names):
            path = REPORTS / name
            with columns[i % len(columns)]:
                st.download_button(
                    name.replace(".json", ""),
                    data=path.read_bytes(),
                    file_name=name,
                    mime=(
                        "text/csv" if path.suffix.lower() == ".csv"
                        else "text/markdown" if path.suffix.lower() == ".md"
                        else "application/json"
                    ),
                    key=f"download-{section}-{name}",
                    width="stretch",
                )


def render_dataset_inventory(manifest: pd.DataFrame | None):
    st.subheader("Audio data inventory")
    st.caption("The project manifest currently contains audio clips only; RF data is tracked separately.")
    if manifest is None:
        st.warning("Audio manifest unavailable. Run scripts/build_manifest.py.")
        return

    required = {"source", "label", "group_id", "exclude_reason"}
    missing = required - set(manifest.columns)
    if missing:
        st.error(f"Audio manifest is missing required columns: {', '.join(sorted(missing))}")
        return

    frame = manifest.copy()
    frame["exclude_reason"] = frame["exclude_reason"].fillna("")
    rows = []
    for source, group in frame.groupby("source", dropna=False):
        usable = group[group["exclude_reason"] == ""]
        rows.append({
            "Audio dataset": str(source),
            "Manifest rows": len(group),
            "Usable clips": len(usable),
            "Excluded clips": len(group) - len(usable),
            "Drone clips": int((usable["label"] == "drone").sum()),
            "Background clips": int((usable["label"] == "background").sum()),
            "Parent recordings": int(group["group_id"].nunique()),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button(
        "Download complete audio manifest (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="audio_manifest.csv",
        mime="text/csv",
        key="download-audio-manifest",
    )
    with st.expander("Preview manifest rows"):
        st.dataframe(
            frame.head(500),
            hide_index=True,
            width="stretch",
            height=360,
        )


def render_proposal_data_status():
    st.subheader("Proposal dataset coverage")
    sources = pd.DataFrame([
        ("DroneAudioDataset + ESC-50 background clips", "Available",
         "Audio manifest; indoor training and environmental negatives"),
        ("Svanström", "Available · separate confuser experiment",
         "Primary tier-3/4 reports keep it test-only. A distinct experiment trains on "
         "recording-held-out helicopter and outdoor-background negatives; all drone "
         "recordings remain test-only. The integrated audio manifest has no explicitly "
         "aircraft-labeled outdoor recordings."),
        ("ITU ARIS Lab", "Available",
         "Audio manifest; 4,491 drone and 1,000 background clips"),
        ("AudioSet", "Partial",
         "PANNs encoder was pretrained on AudioSet; raw AudioSet clips are not in this project"),
        ("UrbanSound8K / FSD50K", "Not included",
         "Not used for training or evaluation"),
        ("DroneRF", "Available · limited",
         "Only communication-RF corpus; 227 segments from one lab session"),
        ("DroneDetect V2", "Partial · exploratory prefix evaluation",
         "20 matched hovering/flying recordings were evaluated with recording-grouped CV "
         "(MA1 vs MAV); the 1.000 scores are not drone-vs-background detection. Five "
         "switched-on recordings are excluded; no background class is present. "
         "IEEE DataPort documents interleaved-float I/Q from BladeRF at 2.4375 GHz"),
        ("CardRF", "Not downloaded / not integrated",
         "The project RF reference identifies it as a possible outdoor cross-dataset corpus; "
         "no local CardRF files were found"),
        ("Other local .dat files", "Unverified / not integrated",
         "IEEE_DATA names and binary shape resemble DroneDetect, but download origin is "
         "unverified; do not count it as another independent corpus"),
        ("RadioML 2018.01A", "Not included",
         "Not used; it is a modulation dataset, not paired drone/audio capture"),
    ], columns=["Proposal dataset", "Availability", "Use / limitation"])
    st.dataframe(sources, hide_index=True, width="stretch")
    st.caption(
        "Raw audio clips can be selected and replayed in the Live audio tab. RF data is not "
        "sample-replayed by the dashboard; its complete evaluation report and matrices are shown "
        "in RF research and the raw JSON download."
    )


def render_rf_asset_inventory():
    root = DATA_DIR / "RF_datasets"
    if not root.is_dir():
        st.warning(f"Raw RF dataset folder is not available: {root}")
        return
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        st.warning(f"No raw RF files found under {root}.")
        return
    rows = [{
        "Local folder": path.relative_to(root).parts[0],
        "File": path.name,
        "Relative path": str(path.relative_to(DATA_DIR)),
        "Format": path.suffix.lower() or "(no extension)",
        "Size (MB)": path.stat().st_size / 1_000_000,
    } for path in files]
    with st.expander(f"Raw RF file inventory · {len(files)} local files"):
        st.caption(
            "The files remain on this computer under data/RF_datasets and are not uploaded or "
            "copied into the dashboard. The matching DroneRF archive and extracted CSV files "
            "are the same corpus, not independent validation. The 25 DroneDetect V2 files have "
            "a source identified but are only a partial copy and are not used in evaluation. "
            "The 46 IEEE_DATA files resemble DroneDetect by name and format, but their source "
            "is unverified."
        )
        inventory = pd.DataFrame(rows)
        st.metric("Total raw RF data", f"{inventory['Size (MB)'].sum() / 1000:.2f} GB")
        audit_specs = [
            (("RF_Dataset1", "RF_Dataset1"), "DroneRF archive files",
             "One archive was checked against its extracted CSV folder: all 20 member names "
             "and byte sizes match. This is a duplicate copy of DroneRF, not a second corpus."),
            (("RF_Dataset2", "RF_datasets", "FR Data_00000_H2"),
             "Extracted DroneRF files",
             "20 CSV files matching the corresponding archive members by name and byte size; "
             "only this archive group is extracted here."),
            (("RF_Dataset2", "RF_datasets", "IEEE_DATA"),
             "Possible DroneDetect captures",
             "46 .dat files with AIR/DIS/INS/MIN prefixes. Filename layout, byte sizes, and "
             "sampled values resemble DroneDetect V2, but there is no download-origin metadata "
             "to confirm their source; do not treat them as independent data."),
            (("datasets_2",), "DroneDetect V2 (partial)",
             "25 .dat files with IEEE DataPort origin metadata identifying DroneDetect V2 "
             "(DOI 10.21227/5jjj-1m32). The official page specifies interleaved-float I/Q, "
             "1.2e8 complex samples per 2-second recording, and a 2.4375 GHz center frequency. "
             "All local files are Bluetooth-only (code 01), with MA1/MAV prefixes; this is a "
             "partial subset with no explicit background class."),
        ]
        audit_rows = []
        for prefix, collection, assessment in audit_specs:
            selected = [
                path for path in files
                if path.relative_to(root).parts[:len(prefix)] == prefix
            ]
            if selected:
                audit_rows.append({
                    "Collection": collection,
                    "Local files": len(selected),
                    "Size (GB)": sum(path.stat().st_size for path in selected) / 1_000_000_000,
                    "Provenance assessment": assessment,
                })
        st.subheader("Local RF source audit")
        st.dataframe(pd.DataFrame(audit_rows).style.format({"Size (GB)": "{:.2f}"}),
                     hide_index=True, width="stretch")
        st.markdown(
            "[DroneDetect dataset source and file-format documentation]"
            "(https://ieee-dataport.org/open-access/"
            "dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine)"
        )
        st.dataframe(inventory.style.format({"Size (MB)": "{:.2f}"}),
                     hide_index=True, width="stretch", height=420)
        st.download_button(
            "Download RF file inventory (CSV)",
            data=inventory.to_csv(index=False).encode("utf-8"),
            file_name="rf_file_inventory.csv",
            mime="text/csv",
            key="download-rf-file-inventory",
        )


def audio_tier_rows(baseline: dict, pretrained: dict, outdoor: dict) -> pd.DataFrame:
    rows = []
    baseline_names = {"logreg": "Audio baseline · Logistic regression",
                      "hgb": "Audio baseline · HistGradientBoosting"}
    pretrained_names = {"logreg": "PANNs encoder · Logistic head",
                        "mlp": "PANNs encoder · MLP head"}
    tiers = (
        ("tier1_test", "Tier 1 · indoor holdout"),
        ("tier2_test", "Tier 2 · unseen drone"),
        ("tier3_test", "Tier 3 · outdoor test"),
    )
    for tier, label in tiers:
        for model, display_name in baseline_names.items():
            rows.append({"Tier": label, "Model": display_name,
                         **baseline["models"][model][tier]})
        for model, display_name in pretrained_names.items():
            rows.append({"Tier": label, "Model": display_name,
                         **pretrained["models"][model][tier]})
    for model, result in outdoor["models"].items():
        rows.append({"Tier": "Tier 4 · outdoor + ARIS", "Model": model, **result})
    return pd.DataFrame(rows)


def render_audio_results(baseline: dict, pretrained: dict, outdoor: dict):
    results = audio_tier_rows(baseline, pretrained, outdoor)
    st.subheader("Audio model · cross-tier evaluation")
    st.caption("PR-AUC is emphasized because the test-set class balance differs from deployment. "
               "Thresholds are selected on validation negatives, not on the test set.")
    st.caption("Values as stored in the reports; training settings and epochs for every model "
               "are in the project report (Section: Model training details).")
    chart = alt.Chart(results).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Tier:N", sort=["Tier 1 · indoor holdout", "Tier 2 · unseen drone",
                                "Tier 3 · outdoor test", "Tier 4 · outdoor + ARIS"],
                title=None, axis=alt.Axis(labelAngle=-18)),
        y=alt.Y("recall:Q", title="Recall at frozen threshold",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", title="Model"),
        tooltip=["Tier", "Model", alt.Tooltip("pr_auc:Q", format=".3f", title="PR-AUC"),
                 alt.Tooltip("recall:Q", format=".1%", title="Recall"),
                 alt.Tooltip("fpr:Q", format=".2%", title="FPR")],
    ).properties(height=290)
    st.altair_chart(chart, width="stretch")
    display = results[["Tier", "Model", "threshold", "prevalence", "pr_auc",
                       "pr_auc_ci95", "roc_auc", "recall", "fpr",
                       "false_alarms", "negatives"]].copy()
    display["PR-AUC (95% CI)"] = display["pr_auc_ci95"].map(
        lambda ci: f"{ci[0]:.3f}–{ci[1]:.3f}")
    display["PR-AUC"] = display["pr_auc"]
    display = display.drop(columns=["pr_auc_ci95", "pr_auc"])
    display = display.rename(columns={
        "Tier": "Evaluation tier", "threshold": "Frozen threshold",
        "prevalence": "Drone prevalence", "roc_auc": "ROC-AUC",
        "recall": "Recall", "fpr": "FPR", "false_alarms": "False alarms",
        "negatives": "Negative windows",
    })
    st.dataframe(display.style.format({
        "Frozen threshold": "{:.3f}", "Drone prevalence": "{:.1%}",
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Recall": "{:.1%}", "FPR": "{:.2%}",
    }), hide_index=True, width="stretch")


def render_svanstrom_confusers(report: dict | None):
    st.subheader("Outdoor helicopter-negative training experiment")
    if report is None:
        st.info("Run `python scripts/train_svanstrom_confusers.py` to generate this experiment.")
        return

    st.warning(report["aircraft_status"])
    st.caption(
        f"{report['split']} This is an experimental recording-held-out split "
        "within Svanström, not an independent-corpus validation."
    )
    rows = []
    category_rows = []
    for name, metrics in report["models"].items():
        rows.append({
            "Model": name.replace("_", " "),
            "Svanström negatives added": "Yes" if "with_svanstrom" in name else "No",
            "PR-AUC": metrics["pr_auc"],
            "PR-AUC 95% CI": (
                f"{metrics['pr_auc_ci95'][0]:.3f}–{metrics['pr_auc_ci95'][1]:.3f}"
            ),
            "ROC-AUC": metrics["roc_auc"],
            "Recall": metrics["recall"],
            "Window FPR": metrics["fpr"],
            "Threshold": metrics["threshold"],
        })
        false_alarms = {
            row["category"]: row for row in metrics["false_alarms_by_category"]
        }
        for recording in metrics["recording_level"]:
            fpr = false_alarms.get(recording["category"])
            category_rows.append({
                "Model": name.replace("_", " "),
                "Category": recording["category"],
                "Window FPR": fpr["fa_rate"] if fpr else None,
                "False alarms / windows": (
                    f"{fpr['false_alarms']}/{fpr['clips']}" if fpr else "—"
                ),
                "Recording alarm rate": recording["alarm_rate"],
                "Alarms / recordings": (
                    f"{recording['alarms']}/{recording['recordings']}"
                ),
            })
    st.dataframe(pd.DataFrame(rows).style.format({
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Recall": "{:.1%}",
        "Window FPR": "{:.2%}", "Threshold": "{:.3f}",
    }), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(category_rows).style.format({
        "Window FPR": "{:.2%}", "Recording alarm rate": "{:.1%}",
    }), hide_index=True, width="stretch")
    st.caption(
        "The Svanström recordings were used in earlier project evaluation, so this is not "
        "a blind external test. This experiment does not replace the tier-4 detector."
    )


def render_audio_architecture_comparison(report: dict | None):
    st.subheader("Audio architecture comparison")
    if report is None:
        st.info(
            "Run `python scripts/compare_audio_architectures.py` to train and "
            "evaluate the compact architecture suite."
        )
        return

    st.caption(
        "All architectures use the same cached log-mel features pooled to 64 bands "
        "and 21 frames, and recording-grouped splits. Thresholds are set on validation "
        "negatives. These are comparison experiments; none replaces the deployed detector."
    )
    rows = []
    for architecture, model in report["models"].items():
        for evaluation, metrics in model["evaluations"].items():
            lo, hi = metrics["pr_auc_ci95"]
            rows.append({
                "Evaluation": evaluation,
                "Architecture": architecture.replace("_", " "),
                "PR-AUC": metrics["pr_auc"],
                "PR-AUC 95% CI": f"{lo:.3f}–{hi:.3f}",
                "ROC-AUC": metrics["roc_auc"],
                "Recall": metrics["recall"],
                "FPR": metrics["fpr"],
                "False alarms / negatives": (
                    f"{metrics['false_alarms']}/{metrics['negatives']}"
                ),
                "Best epoch": model["training"]["best_epoch"],
                "Parameters": model["training"]["parameters"],
            })
    st.dataframe(pd.DataFrame(rows).style.format({
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}",
        "Recall": "{:.1%}", "FPR": "{:.2%}",
    }), hide_index=True, width="stretch")
    confusion_rows = []
    for architecture, model in report["models"].items():
        for evaluation, metrics in model["evaluations"].items():
            tn, fp = metrics["confusion_matrix"][0]
            fn, tp = metrics["confusion_matrix"][1]
            confusion_rows.append({
                "Evaluation": evaluation,
                "Architecture": architecture.replace("_", " "),
                "True negatives": tn,
                "False positives": fp,
                "False negatives": fn,
                "True positives": tp,
            })
    st.dataframe(pd.DataFrame(confusion_rows), hide_index=True, width="stretch")
    st.warning(
        "Tier 3 uses Svanström, which appeared in earlier project evaluation; "
        "it is not a blind independent-corpus test. See the report for the full "
        "protocol, confusion matrices, and architecture caveats."
    )


def render_tier1_cnn_report(baseline_run: dict, cnn_run: dict):
    st.subheader("Additional Tier 1 audio baseline and CNN run")
    st.caption("Separate manifest-based run; do not combine its Tier 1 scores with the cross-tier "
               "tables above. Metrics below use thresholds selected on validation data.")
    threshold_name = st.radio(
        "Reported test threshold",
        ["validation_tuned_threshold", "threshold_0_5"],
        format_func=lambda value: "Validation-tuned threshold" if value == "validation_tuned_threshold"
        else "Fixed 0.5 threshold",
        horizontal=True,
        key="tier1_audio_threshold",
    )
    rows = []
    chosen = {}
    for label, report in (("Random Forest baseline", baseline_run), ("Audio CNN", cnn_run)):
        result = report["test"][threshold_name]
        chosen[label] = result
        summary = result["classification_report"]
        rows.append({
            "Model": label,
            "Threshold": result["threshold"],
            "Accuracy": summary["accuracy"],
            "Precision (drone)": summary["drone"]["precision"],
            "Recall (drone)": summary["drone"]["recall"],
            "F1 (drone)": summary["drone"]["f1-score"],
            "PR-AUC": result["pr_auc"],
            "ROC-AUC": result["roc_auc"],
        })
    st.dataframe(pd.DataFrame(rows).style.format({
        "Threshold": "{:.3f}", "Accuracy": "{:.1%}", "Precision (drone)": "{:.1%}",
        "Recall (drone)": "{:.1%}", "F1 (drone)": "{:.1%}",
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}",
    }), hide_index=True, width="stretch")
    class_rows = []
    for label, result in chosen.items():
        for class_name, metrics in result["classification_report"].items():
            if isinstance(metrics, dict) and "precision" in metrics:
                class_rows.append({
                    "Model": label, "Class": class_name,
                    "Precision": metrics["precision"], "Recall": metrics["recall"],
                    "F1": metrics["f1-score"], "Support": int(metrics["support"]),
                })
    st.markdown("#### Complete class-wise report")
    st.dataframe(pd.DataFrame(class_rows).style.format({
        "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}",
    }), hide_index=True, width="stretch")

    matrices = st.columns(2)
    for column, (label, result) in zip(matrices, chosen.items()):
        with column:
            st.markdown(f"#### {label} · confusion matrix")
            st.dataframe(pd.DataFrame(
                result["confusion_matrix"],
                index=["Actual background", "Actual drone"],
                columns=["Predicted background", "Predicted drone"],
            ), width="stretch")

    history = cnn_run["training"]["history"]
    history_frame = pd.DataFrame(history)
    st.markdown("#### CNN training history")
    st.caption(
        f"{cnn_run['training']['epochs_run']} epochs run; best epoch "
        f"{cnn_run['training']['best_epoch']} at validation PR-AUC "
        f"{cnn_run['training']['best_validation_pr_auc']:.3f}."
    )
    history_chart = alt.Chart(history_frame).transform_fold(
        ["train_loss", "validation_pr_auc"],
        as_=["Metric", "Value"],
    ).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="Epoch"),
        y=alt.Y("Value:Q", title="Metric value"),
        color=alt.Color("Metric:N"),
        tooltip=["epoch:Q", "Metric:N", alt.Tooltip("Value:Q", format=".3f")],
    ).properties(height=240)
    st.altair_chart(history_chart, width="stretch")


def render_synthetic_noise_report(report: dict):
    st.subheader("Additional synthetic white/pink-noise experiment")
    protocol = report["protocol"]
    st.caption(
        f"{protocol['test_clips']:,} Tier 1 test clips; {protocol['test_source_groups']:,} "
        "held-out source recordings. White and pink noise are controlled synthetic conditions, "
        "not real-world recordings. Thresholds remain fixed at clean-validation values."
    )
    rows = [
        {
            "Condition": result["condition"],
            "SNR": f"{result['snr_db']} dB" if result["snr_db"] is not None else "Clean",
            "Model": result["model"].replace("_", " ").title(),
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
            "PR-AUC": result["pr_auc"],
            "ROC-AUC": result["roc_auc"],
            "False positives": result["false_positives"],
            "False negatives": result["false_negatives"],
            "F1 change": result["f1_delta_from_clean"],
            "PR-AUC change": result["pr_auc_delta_from_clean"],
            "Threshold": result["threshold"],
        }
        for result in report["results"]
    ]
    st.dataframe(pd.DataFrame(rows).style.format({
        "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}",
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}",
        "F1 change": "{:+.1%}", "PR-AUC change": "{:+.3f}",
        "Threshold": "{:.3f}",
    }), hide_index=True, width="stretch")


def render_archived_training_curves(report: dict):
    with st.expander("Additional archived training and learning curves"):
        st.caption(
            "These diagnostics come from a separate earlier run in training_curves.json; "
            "they are not the 11-epoch CNN run summarized above."
        )
        cnn = pd.DataFrame(report["cnn"])
        cnn["epoch"] = np.arange(1, len(cnn) + 1)
        loss_frame = cnn.melt(
            id_vars="epoch",
            value_vars=["train_loss", "val_loss"],
            var_name="Series",
            value_name="Loss",
        )
        st.markdown("#### Archived CNN loss")
        st.altair_chart(
            alt.Chart(loss_frame).mark_line().encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("Loss:Q"),
                color="Series:N",
                tooltip=["epoch:Q", "Series:N", "Loss:Q"],
            ).properties(height=220),
            width="stretch",
        )
        pr_frame = cnn.melt(
            id_vars="epoch",
            value_vars=["train_pr_auc", "val_pr_auc"],
            var_name="Series",
            value_name="PR-AUC",
        )
        st.markdown("#### Archived CNN PR-AUC")
        st.altair_chart(
            alt.Chart(pr_frame).mark_line().encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("PR-AUC:Q", scale=alt.Scale(domain=[0, 1])),
                color="Series:N",
                tooltip=["epoch:Q", "Series:N", "PR-AUC:Q"],
            ).properties(height=220),
            width="stretch",
        )
        learning_rows = []
        for key, model in (("audio_head_learning_curve", "Audio head"),
                           ("rf_forest_learning_curve", "RF forest")):
            curve = report[key]
            for i, clips in enumerate(curve["train_clips"]):
                learning_rows.extend([
                    {"Model": model, "Training clips": clips, "Split": "Train",
                     "PR-AUC": curve["train_pr_auc"][i]},
                    {"Model": model, "Training clips": clips, "Split": "Validation",
                     "PR-AUC": curve["val_pr_auc"][i]},
                ])
        learning = pd.DataFrame(learning_rows)
        st.markdown("#### Archived training-size curves")
        st.altair_chart(
            alt.Chart(learning).mark_line(point=True).encode(
                x=alt.X("Training clips:Q"),
                y=alt.Y("PR-AUC:Q", scale=alt.Scale(domain=[0, 1])),
                color="Model:N",
                strokeDash="Split:N",
                tooltip=["Model:N", "Training clips:Q", "Split:N",
                         alt.Tooltip("PR-AUC:Q", format=".3f")],
            ).properties(height=240),
            width="stretch",
        )


FUSION_FORMAT = {"Recall": "{:.1%}", "False-positive rate": "{:.2%}", "PR-AUC": "{:.3f}"}


def fusion_ablation_rows(report: dict, scenario: str) -> pd.DataFrame:
    rows = S.fusion_rows(report, scenario)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop(columns=["method"])
    # OR / AND are hard decisions with no ranking score: show n/a, not "None".
    numeric = ["Recall", "False-positive rate", "PR-AUC"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def _highlight_deployed(row: pd.Series) -> list[str]:
    if row.get("Role") == "deployed rule":
        return ["font-weight: 700" for _ in row]
    return ["" for _ in row]


def show_fusion_table(report: dict, scenario: str):
    table = fusion_ablation_rows(report, scenario)
    if table.empty:
        st.caption(f"No methods reported for {scenario}.")
        return
    st.dataframe(table.style.apply(_highlight_deployed, axis=1)
                 .format(FUSION_FORMAT, na_rep="n/a"),
                 hide_index=True, width="stretch")


FUSION_CAVEAT = (
    "<strong>Unpaired data.</strong> No dataset records audio and RF of the same flight. "
    "Each fusion test pair is an audio window and an RF window of the same class drawn "
    "independently, so these numbers are an upper bound, not synchronized multimodal "
    "validation. The {n:,} pairs per class are resampled from a smaller pool of windows "
    "(confidence intervals resample audio recordings)."
)


def render_fusion_highlights(report: dict | None):
    st.markdown("#### Fusion ablation · headline (synthetic pairs)")
    if report is None:
        st.info("Fusion report unavailable. Run `python scripts/eval_fusion.py` "
                "(needs the RF v2 artifacts first).")
        return
    st.markdown(f'<div class="callout">{FUSION_CAVEAT.format(n=report["pairs_per_class"])}'
                "</div>", unsafe_allow_html=True)
    deployed = report.get("deploy", "sum")
    chart_rows = []
    for scenario, label in (("both sensors", "Both sensors see the drone"),
                            ("RF-silent drone (autonomous)", "Drone with no radio link")):
        for row in S.fusion_rows(report, scenario, methods=("audio only", "RF only", "sum", "learned")):
            chart_rows.append({"Scenario": label, "Method": row["Method"]
                               + (" (deployed)" if row["method"] == deployed else ""),
                               "Recall": row["Recall"], "Drones caught": row["Drones caught"]})
    if chart_rows:
        method_colors = {"Audio only": "#38bdf8", "RF only": "#a78bfa",
                         "Evidence sum (deployed)": "#f59e0b",
                         "Learned fusion layer": "#8a8f98"}
        bars = alt.Chart(pd.DataFrame(chart_rows)).mark_bar().encode(
            x=alt.X("Recall:Q", title="share of drone pairs detected (recall)",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Method:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("Method:N", legend=None,
                            scale=alt.Scale(domain=list(method_colors),
                                            range=list(method_colors.values()))),
            row=alt.Row("Scenario:N", title=None, sort=["Both sensors see the drone",
                                                        "Drone with no radio link"],
                        header=alt.Header(labelAngle=0, labelAlign="left",
                                          labelFontWeight="bold")),
            tooltip=["Scenario", "Method", alt.Tooltip("Recall:Q", format=".1%"),
                     "Drones caught"],
        ).properties(height=120)
        st.altair_chart(bars, width="stretch")
    left, right = st.columns(2)
    for column, scenario, note in (
        (left, "both sensors",
         "Drone pairs: drone audio + drone RF. RF alone is already near-perfect here, so "
         "fusion adds little: this is RF's best case (one lab's negatives)."),
        (right, "RF-silent drone (autonomous)",
         "Drone pairs: drone audio + background RF (a drone with no radio link). "
         "Only audio can detect these."),
    ):
        if scenario not in report["scenarios"]:
            continue
        with column:
            st.markdown(f"**Scenario: {scenario}**")
            st.caption(note)
            show_fusion_table(report, scenario)
    silent = report["scenarios"].get("RF-silent drone (autonomous)", {})
    if "learned" in silent and deployed in silent:
        st.warning(
            f"The learned fusion layer is **not** the recommended rule. On RF-silent drones it "
            f"catches {silent['learned']['recall']:.1%} of drone pairs, versus "
            f"{silent[deployed]['recall']:.1%} for the deployed "
            f"{S.FUSION_METHOD_LABEL[deployed].lower()}"
            + (f" and {silent['audio only']['recall']:.1%} for audio alone"
               if "audio only" in silent else "")
            + ". It was fitted on pairs that never contain a drone without a radio link, so it "
              "learned to distrust audio when RF is quiet. (Switching to the sum was decided "
              "after seeing these test results; reports/fusion.md says so.)"
        )


def _status_badge(status: str) -> str:
    return {S.READY: "✅ Ready", S.MISSING: "⛔ Missing", S.BLOCKED: "🔒 Blocked",
            S.INVALID: "⚠️ Invalid"}.get(status, status)


def render_readiness(artifacts: list[S.Artifact]):
    frame = pd.DataFrame([{
        "Component": a.name,
        "Status": _status_badge(a.status),
        "What is on disk": a.detail,
        "Next step": a.next_step or "—",
    } for a in artifacts])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Checked from files under data/ each time the page loads (not from a status "
               "document). A report in reports/ can exist without the model it describes.")


def _metric_line(label: str, value: str, detail: str):
    st.markdown(f"**{label}** · {value}  \n<span style='opacity:.8'>{detail}</span>",
                unsafe_allow_html=True)


def render_headline_results(reports: dict):
    outdoor = (reports.get("outdoor") or {}).get("models", {}).get("pretrained balanced")
    rf_v2 = reports.get("rf_v2")
    fusion = reports.get("fusion")
    a, r, f = st.columns(3)
    with a:
        st.markdown("##### Audio · outdoor, unseen recordings")
        if outdoor:
            tn, fp = outdoor["confusion"][0]
            fn, tp = outdoor["confusion"][1]
            _metric_line("Drone windows caught", f"{outdoor['recall']:.0%} ({tp}/{tp + fn})",
                         "1-s windows, Svanström outdoor test set")
            _metric_line("False alarms", f"{outdoor['fpr']:.1%} ({fp}/{fp + tn})",
                         "helicopter + outdoor-background windows; threshold frozen at "
                         "1% FPR on validation")
            rec = {x["category"]: x for x in reports["outdoor"].get("recording_level", [])}
            if "drone" in rec:
                d = rec["drone"]
                heli = rec.get("helicopter")
                _metric_line("Drone recordings alarmed", f"{d['alarms']}/{d['recordings']}",
                             "recording-level; helicopter recordings alarmed: "
                             + (f"{heli['alarms']}/{heli['recordings']}" if heli else "n/a"))
            st.caption("Source: reports/outdoor_audio.json (tier 4, pretrained balanced head).")
        else:
            st.caption("reports/outdoor_audio.json unavailable.")
    with r:
        st.markdown("##### RF v2 · held-out test windows")
        if rf_v2:
            t = rf_v2["variants"][rf_v2["deployed"]]["test_overall"]
            (tn, fp), (fn, tp) = t["confusion"]
            _metric_line("Drone windows caught", f"{t['recall']:.1%} ({tp}/{tp + fn})",
                         "25 ms windows: DroneRF held-out time + held-out Tampere drones")
            _metric_line("False alarms", f"{t['fpr']:.1%} ({fp}/{fp + tn})",
                         "DroneRF lab background + synthetic receiver noise only")
            st.caption("Source: reports/rf_v2.json. No real-world Wi-Fi/Bluetooth negatives; "
                       "see the readiness table for whether the model file exists here.")
        else:
            st.caption("reports/rf_v2.json unavailable.")
    with f:
        st.markdown("##### Fusion · synthetic audio+RF pairs")
        if fusion:
            sc = fusion["scenarios"]
            dep = fusion.get("deploy", "sum")
            both = sc.get("both sensors", {})
            silent = sc.get("RF-silent drone (autonomous)", {})
            if dep in both:
                _metric_line("Both sensors, evidence sum", f"{both[dep]['recall']:.1%} recall",
                             f"false alarms {both[dep]['false_alarms']}/{both[dep]['negatives']}")
            if dep in silent and "learned" in silent:
                _metric_line("RF-silent drone", f"sum {silent[dep]['recall']:.1%} · "
                             f"learned {silent['learned']['recall']:.1%}",
                             f"audio only {silent['audio only']['recall']:.1%}; "
                             f"{fusion['pairs_per_class']:,} drone pairs per method")
            st.caption("Source: reports/fusion.json. Unpaired, independently sampled pairs.")
        else:
            st.caption("reports/fusion.json unavailable.")


def render_state_definitions():
    cols = st.columns(4)
    for column, state in zip(cols, S.STATES):
        column.markdown(
            f'<div class="result" style="--state:{STATE_COLOR[state]};padding:12px 14px">'
            f'<div class="state" style="font-size:1.15rem">{STATE_ICON[state]} '
            f'{S.STATE_LABEL[state]}</div><p class="meaning">{S.STATE_MEANING[state]}</p></div>',
            unsafe_allow_html=True,
        )


def render_overview(reports: dict, manifest: pd.DataFrame | None):
    artifacts = S.artifact_status(DATA_DIR)
    by = {a.name: a for a in artifacts}
    rf_ready = S.rf_replay_ready(artifacts)
    audio_ready = by["Audio detector (tier 4 + uncertainty flags)"].ready

    st.markdown("### Current readiness")
    a, r, f = st.columns(3)
    with a:
        panel("Audio · trained and evaluated" if audio_ready else "Audio · model missing",
              "Acoustic detection",
              "1-second windows scored by a frozen pretrained audio encoder (PANNs) plus a "
              "trained head, with uncertainty flags and Grad-CAM. Usable in the replay tab on "
              "recorded or uploaded audio.", "ready" if audio_ready else "pending")
    with r:
        panel("RF · reports exist, " + ("model ready" if rf_ready else "model not built here"),
              "Radio-link detection (2.4 GHz)",
              "RF v2 is evaluated in reports/rf_v2.md. Its model file "
              + ("is present, so recorded RF test windows can be replayed."
                 if rf_ready else
                 "is not on this computer: it needs the Tampere recordings (5.3 GB) first.")
              + " There is no live RF receiver.", "ready" if rf_ready else "pending")
    with f:
        panel("Fusion · research experiment",
              "Audio + RF evidence fusion",
              "Evidence-sum fusion (deployed) and a learned layer (experimental), evaluated "
              "on unpaired synthetic pairs. "
              + ("Replay available." if rf_ready else "Replay unavailable until RF v2 and "
                 "fusion artifacts are built."), "pending")
    st.caption(
        "Replay mode available now: "
        + ("**audio + recorded RF windows (unpaired what-if)**" if rf_ready and audio_ready
           else "**audio only**" if audio_ready else "**none: audio model missing**")
        + ". Live microphone use is in the separate operations console (`python -m console`)."
    )

    st.markdown("### How a prediction is made")
    st.markdown(
        """
        <div class="stage-flow">
          <div class="stage"><small>1 · Audio</small><strong>Score each 1-s window</strong>
            <span>Log-mel → frozen PANNs encoder → logistic head → a model score from 0 to 1,
            plus flags: near the threshold, unlike training audio, unlike known drones.</span></div>
          <div class="stage"><small>2 · RF (replay only)</small><strong>Score a 25 ms spectrum</strong>
            <span>Recorded 2.4 GHz power spectrum → RF v2 forest → score + flags. Windows come
            from held-out test recordings, not a receiver.</span></div>
          <div class="stage"><small>3 · Fusion</small><strong>Add calibrated evidence</strong>
            <span>Each score → capped log-likelihood ratio → sum. Threshold set for 1% false
            alarms on validation pairs, separately for each set of available sensors.</span></div>
          <div class="stage"><small>4 · Decision</small><strong>Four states + alarm rule</strong>
            <span>Each window gets one of the four states below. A recording alarms only after
            N consecutive drone windows. A person reviews every alarm.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How to read a result")
    render_state_definitions()
    st.caption(
        "Scores are model scores, not calibrated probabilities: 0.95 does not mean a 95% chance "
        "of a drone. Fused values are log-likelihood ratios (positive favours drone). "
        "'Known Drone' never names the drone model; the per-model identification results are "
        "a separate, same-session experiment (reports/identify.md)."
    )

    st.markdown("### Headline results")
    render_headline_results(reports)

    st.markdown("### What the results do and do not show")
    shows, not_shows = st.columns(2)
    with shows:
        st.markdown(
            "**They show**\n"
            "- The audio detector works on held-out indoor recordings, and catches only "
            "about a fifth of outdoor drone windows at a 1% false-alarm setting.\n"
            "- RF v1 learned a shortcut (pure receiver noise was called 'drone'); RF v2 "
            "fixes that on the tested noise families and on a second receiver.\n"
            "- With RF-silent drones, only audio helps; the evidence sum keeps most of audio's "
            "recall and the learned layer does not.\n"
            "- Uncertainty flags route part of the errors to review."
        )
    with not_shows:
        st.markdown(
            "**They do not show**\n"
            "- Field performance: nothing was recorded at a real deployment site.\n"
            "- Synchronized audio+RF performance: fusion pairs are not the same flight.\n"
            "- RF false alarms on real Wi-Fi/Bluetooth traffic: RF negatives come from one "
            "lab plus synthetic noise.\n"
            "- Calibrated probabilities for audio, or exact drone-model identification "
            "in the four states."
        )

    st.markdown("### Readiness of models and data on this computer")
    render_readiness(artifacts)

    st.markdown("### Blocked: needs data or hardware this project does not have")
    st.markdown(
        ("" if rf_ready else
         "- **Tampere RF recordings** (Zenodo 4264467, 5.3 GB download): needed to rebuild "
         "RF v2 and the fusion models locally.\n")
        + "- **Independent real-world RF negatives** (Wi-Fi, Bluetooth, other 2.4 GHz traffic "
        "recorded outside the DroneRF lab): needed to measure RF false alarms.\n"
        "- **Synchronized audio + RF recordings of the same flights**: needed to validate "
        "fusion instead of the synthetic-pair upper bound.\n"
        "- **An RF receiver (SDR) integration**: the dashboard only replays recorded RF windows.\n"
        "- **Field recordings at the intended site**: needed before any operational claim."
    )

    st.markdown("### Where to look next")
    st.dataframe(pd.DataFrame([
        ("Replay detector", "Run the detector on a recording and see the result, per-sensor "
         "evidence and Grad-CAM"),
        ("Fusion", "Full audio-only / RF-only / sum / learned / OR / AND ablation by scenario"),
        ("Audio intelligence", "Cross-tier audio results, noise robustness, uncertainty"),
        ("RF research", "RF v1 shortcut, v2, v3 experiment, calibration and file inventory"),
        ("Proposal checklist", "Each proposal item and how it was implemented"),
    ], columns=["Tab", "What it contains"]), hide_index=True, width="stretch")

    with st.expander("Data inventory, proposal coverage and downloads"):
        if manifest is not None:
            total = len(manifest)
            usable = int((manifest.exclude_reason == "").sum())
            drone = int((manifest.label == "drone").sum())
            background = int((manifest.label == "background").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Audio manifest rows", f"{total:,}")
            c2.metric("Usable rows", f"{usable:,}")
            c3.metric("Drone-labelled rows", f"{drone:,}")
            c4.metric("Background rows", f"{background:,}")
        render_dataset_inventory(manifest)
        render_proposal_data_status()
        st.subheader("Proposal coverage at a glance")
        coverage = pd.DataFrame([
            ("Acoustic baseline + CNN / CNN+RNN / transformer comparison", "Implemented",
             "Audio intelligence tab; outdoor transfer remains weak for every model"),
            ("Audio confidence, uncertainty and open-set states", "Implemented",
             "Four states per window; unseen-drone flag AUROC 0.81"),
            ("Noise robustness by condition and severity", "Implemented",
             "Voice, traffic, wind, music, mixed, real speech/music at 20/10/0 dB"),
            ("Grad-CAM audio explanation", "Implemented",
             "Per-window spectrogram evidence in the replay tab"),
            ("RF baseline vs deep model", "Implemented",
             "Forest / logistic regression / 1D-CNN; v2 fixes v1's noise shortcut"),
            ("Per-drone-model classification and RF open set", "Implemented with limits",
             "Which-drone confusion matrices; RF unknown flag works for 2 of 3 held-out drones"),
            ("Cross-dataset evaluation", "Implemented (reports)",
             "Audio: outdoor Svanström; RF: Tampere (raw data not on this computer)"),
            ("Probability calibration / ECE", "Implemented (RF only)",
             "Temperature scaling on DroneRF, rf_trust.md; audio scores are not calibrated"),
            ("Replay dashboard", "Implemented (audio); RF replay needs RF v2 artifacts",
             "Recorded/uploaded audio, optional recorded RF windows; no live RF receiver"),
            ("Learned audio/RF fusion", "Implemented, evaluated, not deployed",
             "Synthetic pairs only; misses RF-silent drones, so the evidence sum is deployed"),
        ], columns=["Proposal item", "Current status", "Where / limitation"])
        st.dataframe(coverage, hide_index=True, width="stretch")
        render_report_downloads([
            "baseline_audio.json", "pretrained_audio.json", "outdoor_audio.json",
            "noise_robustness.json", "uncertainty_audio.json", "audio_baseline.json",
            "audio_cnn.json", "audio_noise_robustness.json", "audio_noise_robustness.csv",
            "rf_detection.json", "rf_v2.json", "fusion.json", "training_curves.json",
        ], "overview")


def render_noise(noise: dict):
    st.subheader("Noise robustness · tier 1 held-out recordings")
    st.caption(
        "The clean-validation threshold is held fixed. This is a stress test, not a live noise "
        "classifier. Voice/traffic/environment clips are drawn from mapped sound classes; music "
        "and mixed conditions are synthetic."
    )
    conditions = ["voice", "traffic", "wind", "environment", "music", "mixed"]
    condition = st.selectbox("Noise condition", conditions, key="noise_condition")
    sources = noise["noise_sources"].get(condition, [])
    if sources:
        st.caption("Noise classes used: " + ", ".join(sources))
    else:
        st.caption("This condition uses generated synthetic noise rather than a named ESC-50 class.")
    snr_names = ("low", "medium", "high")
    snr_values = noise["snr_db"]
    rows = []
    for model, values in noise["results"].items():
        for snr in snr_names:
            result = values[f"{condition}/{snr}"]
            rows.append({
                "Model": model, "SNR (dB)": snr_values[snr],
                "PR-AUC": result["pr_auc"], "ROC-AUC": result["roc_auc"],
                "PR-AUC 95% CI": result["pr_auc_ci95"],
                "Threshold": result["threshold"],
                "Drone prevalence": result["prevalence"],
                "Recall": result["recall"], "FPR": result["fpr"],
                "False alarms": result["false_alarms"],
                "Negative windows": result["negatives"],
            })
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("SNR (dB):Q", sort="descending", title="Signal-to-noise ratio (dB)"),
        y=alt.Y("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", title="Model"),
        tooltip=["Model", "SNR (dB)", alt.Tooltip("Recall:Q", format=".1%"),
                 alt.Tooltip("FPR:Q", format=".2%")],
    ).properties(height=280)
    st.altair_chart(chart, width="stretch")
    frame["PR-AUC 95% CI"] = frame["PR-AUC 95% CI"].map(
        lambda ci: f"{ci[0]:.3f}–{ci[1]:.3f}")
    st.dataframe(frame.sort_values(["SNR (dB)", "Model"], ascending=[False, True])
                 .style.format({"PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}",
                                "Threshold": "{:.3f}", "Drone prevalence": "{:.1%}",
                                "Recall": "{:.1%}", "FPR": "{:.2%}"}),
                 hide_index=True, width="stretch")


def render_uncertainty(uncertainty: dict):
    st.subheader("Uncertainty and open-set evaluation")
    rows = []
    for tier, result in uncertainty["tiers"].items():
        rows.append({
            "Evaluation": tier,
            "Error rate": result["error_rate"],
            "Flagged for review": result["flagged_rate"],
            "Error rate when unflagged": result["error_rate_unflagged"],
            "Errors caught by flag": result["errors_caught"],
            "Unfamiliar-input rate": result["unfamiliar_rate"],
        })
    st.dataframe(pd.DataFrame(rows).style.format({
        "Error rate": "{:.1%}", "Flagged for review": "{:.1%}",
        "Error rate when unflagged": "{:.1%}", "Errors caught by flag": "{:.1%}",
        "Unfamiliar-input rate": "{:.1%}",
    }), hide_index=True, width="stretch")
    state_rows = []
    for tier, result in uncertainty["tiers"].items():
        for actual_class, predictions in result["states"].items():
            state_rows.append({"Evaluation": tier, "Actual class": actual_class, **predictions})
    st.markdown("#### Window states by actual class")
    st.dataframe(pd.DataFrame(state_rows), hide_index=True, width="stretch")
    risk_rows = []
    for tier, result in uncertainty["tiers"].items():
        for method, risk in result["risk_coverage"].items():
            risk_rows.append({"Evaluation": tier, "Review score": method,
                              "AURC": risk["aurc"],
                              **risk["error_at_coverage"]})
    st.markdown("#### Risk versus retained coverage")
    st.dataframe(pd.DataFrame(risk_rows).style.format({
        column: "{:.1%}" for column in ("100%", "90%", "80%", "70%", "50%")
    } | {"AURC": "{:.3f}"}), hide_index=True, width="stretch")
    open_set = uncertainty["open_set"]
    x, y, z = st.columns(3)
    x.metric("Unseen Mambo vs held-out Bebop AUROC", f"{open_set['auroc_mambo_vs_bebop']:.3f}")
    y.metric("Mambo flagged unfamiliar", percent(open_set["flag_rate_mambo"]))
    z.metric("Held-out Bebop flagged unfamiliar", percent(open_set["flag_rate_bebop"]))
    st.caption("The unfamiliar-input flag has limited sensitivity to the outdoor domain shift; "
               "see the tier 3/4 rates above.")
    open_state_rows = []
    for label, key in (("Held-out Bebop", "states_bebop"), ("Unseen Mambo", "states_mambo")):
        open_state_rows.append({"Drone set": label, **open_set[key]})
    st.dataframe(pd.DataFrame(open_state_rows).style.format({
        state: "{:.1%}" for state in
        ("no drone", "drone", "unknown drone", "low confidence")
    }), hide_index=True, width="stretch")


def render_audio_tab(reports: dict):
    render_audio_results(reports["baseline"], reports["pretrained"], reports["outdoor"])
    st.markdown("#### Full reports")
    render_md_report("deep_audio.md", "CNN vs CNN+GRU vs transformer (from scratch, tiers 1-4)")
    render_md_report("noise_robustness.md", "Noise robustness: full condition × severity grid")
    render_md_report("uncertainty_audio.md", "Uncertainty flags and unknown-drone test")
    if not st.toggle("Show detailed audio tables", key="audio_details",
                     help="Every experiment table. The project report summarises them, "
                          "with each model's training settings."):
        st.caption("Detailed tables are hidden: architecture sweep, Tier 1 CNN run, noise grid, "
                   "uncertainty, Svanström confusers, training curves.")
        return
    render_svanstrom_confusers(reports.get("svanstrom_confusers"))
    render_audio_architecture_comparison(reports.get("audio_architectures"))
    render_tier1_cnn_report(reports["audio_baseline"], reports["audio_cnn"])
    left, right = st.columns([1.2, 1])
    with left:
        render_noise(reports["noise"])
    with right:
        render_uncertainty(reports["uncertainty"])
    render_synthetic_noise_report(reports["audio_noise"])
    render_archived_training_curves(reports["training_curves"])
    st.markdown("#### More audio reports")
    render_md_report("identify.md", "Which-drone confusion matrices (audio and RF)")
    render_report_downloads([
        "baseline_audio.json", "pretrained_audio.json", "outdoor_audio.json",
        "svanstrom_confusers.json", "svanstrom_confusers.md",
        "audio_architecture_comparison.json", "audio_architecture_comparison.md",
        "noise_robustness.json", "uncertainty_audio.json", "audio_baseline.json",
        "audio_cnn.json", "audio_noise_robustness.json", "audio_noise_robustness.csv",
        "training_curves.json",
    ], "audio")


def render_dronedetect_subset(report: dict):
    st.markdown("#### DroneDetect V2 · exploratory two-prefix classification")
    if "by_mode" not in report["metrics"]:
        st.warning(
            "The saved experiment predates the matched-mode evaluation. Re-run "
            "`python scripts/eval_dronedetect_subset.py`; the older result is not shown."
        )
        return
    st.warning(
        "This is not drone-vs-background detection: the local subset contains only "
        "Bluetooth-interference drone recordings and no background class. The scores "
        "are a small, same-source classification experiment, not deployment validation."
    )
    metrics = report["metrics"]
    cols = st.columns(4)
    cols[0].metric("Recording accuracy", f"{metrics['recording_accuracy']:.1%}")
    cols[1].metric("Balanced accuracy", f"{metrics['recording_balanced_accuracy']:.1%}")
    cols[2].metric("Macro F1", f"{metrics['recording_macro_f1']:.3f}")
    cols[3].metric("ROC-AUC", f"{metrics['recording_roc_auc']:.3f}")
    st.caption(
        f"{report['recording_count']} recordings, {report['window_count']:,} 20 ms windows; "
        f"{report['split']}. Results are aggregated per recording."
    )
    classes = report["classes"]
    confusion = pd.DataFrame(
        metrics["confusion_matrix"], index=classes, columns=classes
    )
    confusion.index.name = "Actual ↓ / predicted →"
    st.dataframe(confusion, width="stretch")
    class_report = metrics["classification_report"]
    st.dataframe(pd.DataFrame([
        {
            "Filename code": code,
            "Precision": class_report[code]["precision"],
            "Recall": class_report[code]["recall"],
            "F1": class_report[code]["f1-score"],
            "Recordings": int(class_report[code]["support"]),
        }
        for code in classes
    ]).style.format({
        "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}",
    }), hide_index=True, width="stretch")
    st.markdown("#### Matched flight-mode results")
    st.dataframe(pd.DataFrame([
        {
            "Mode": mode,
            "Recordings": values["recording_count"],
            "Accuracy": values["accuracy"],
            "Balanced accuracy": values["balanced_accuracy"],
            "Macro F1": values["macro_f1"],
        }
        for mode, values in metrics["by_mode"].items()
    ]).style.format({
        "Accuracy": "{:.1%}",
        "Balanced accuracy": "{:.1%}",
        "Macro F1": "{:.3f}",
    }), hide_index=True, width="stretch")
    st.markdown(f"[Source and file-format documentation]({report['source_url']})")
    with st.expander("Recording-level out-of-fold predictions"):
        st.dataframe(pd.DataFrame(report["recordings"]), hide_index=True, width="stretch")


def render_rf_tab(
    report: dict,
    dronedetect_report: dict | None = None,
    trust_report: dict | None = None,
):
    st.subheader("RF (2.4 GHz radio link) · research")
    st.markdown(
        "- **v1** (trained on DroneRF only) scored near-perfectly, but on a second receiver it "
        "called pure receiver noise *drone*: it had learned the recording session.\n"
        "- **v2** keeps the features and changes the training data; noise and silent windows are "
        "now *no drone*, and drones from a receiver it never trained on are caught.\n"
        "- Every RF result here uses **recorded datasets**; there is no live RF receiver, and "
        "false alarms on real Wi-Fi/Bluetooth traffic are unmeasured."
    )
    st.markdown("#### RF detector versions")
    st.caption("v2 is the deployed RF detector used by the replay and fusion (when its model "
               "file has been built on this computer; see Project briefing). v3 is an experiment "
               "with simulated Wi-Fi / Bluetooth / microwave negatives.")
    render_md_report("rf_tampere.md", "v1 on a new receiver (Tampere): the noise check it fails")
    render_md_report("rf_v2.md", "v2: deployed RF detector", expanded=True)
    render_md_report("rf_v3.md", "v3 experiment: Wi-Fi / Bluetooth / microwave, 1D-CNN, smoothing, timing features")
    render_md_report("rf_uncertainty.md", "RF uncertainty and unknown-drone flags")
    render_md_report("rf_trust.md", "RF probability calibration and held-out Phantom")
    downloads = ["rf_detection.json", "rf_v2.json", "rf_v3.json", "rf_tampere.json"]
    if dronedetect_report is not None:
        downloads.append("dronedetect_subset.json")
    if trust_report is not None:
        downloads.append("rf_trust.json")
    render_report_downloads(downloads, "rf")
    if st.toggle("Show detailed RF tables", key="rf_details",
                 help="v1 diagnostics, calibration, recorded-window replay, drone-type "
                      "identification and the local file inventory."):
        render_rf_details(report, dronedetect_report, trust_report)


def render_rf_details(
    report: dict,
    dronedetect_report: dict | None = None,
    trust_report: dict | None = None,
):
    st.markdown("#### DroneRF data coverage")
    st.caption("RF here means communication/Wi-Fi transmission features from DroneRF—not mmWave radar.")
    rf_inventory = pd.DataFrame([
        ("Source corpus", "DroneRF"),
        ("Capture sessions", "One lab session (not independent-session validation)"),
        ("Segments", "227 × 0.25 s: 84 Bebop, 81 AR, 21 Phantom, 41 background"),
        ("Derived windows", "2,270 × 25 ms"),
        ("Frequency bands", "Two receiver bands covering lower/upper 2.4 GHz"),
        ("Other local RF folders", "Present in the file inventory; provenance and model use not verified"),
        ("Independent-session RF validation", "Not available in the current DroneRF report"),
    ], columns=["RF data coverage", "Current data"])
    st.dataframe(rf_inventory, hide_index=True, width="stretch")
    st.markdown("#### Probability calibration and unknown-signature test")
    if trust_report is None:
        st.caption(
            "No RF trust metrics are available yet. The evaluation is not run until "
            "the prepared DroneRF feature cache exists."
        )
    else:
        calibration_rows = []
        for split in ("validation", "test"):
            for kind in ("raw", "temperature_scaled"):
                metrics = trust_report["binary_calibration"][split][kind]
                calibration_rows.append({
                    "Split": split,
                    "Probabilities": kind.replace("_", " "),
                    "ECE": metrics["ece"],
                    "Brier": metrics["brier_score"],
                    "NLL": metrics["nll"],
                    "Windows": metrics["sample_count"],
                })
        st.dataframe(pd.DataFrame(calibration_rows).style.format({
            "ECE": "{:.4f}", "Brier": "{:.4f}", "NLL": "{:.4f}",
        }), hide_index=True, width="stretch")
        open_set_metrics = trust_report["open_set"]["open_set_test"]
        st.dataframe(pd.DataFrame([{
            "Unknown model": trust_report["open_set"]["unknown_drone_model"],
            "Unknown ROC-AUC": open_set_metrics["unknown_roc_auc"],
            "Known acceptance": open_set_metrics["known_accept_rate"],
            "Unknown rejection": open_set_metrics["unknown_reject_rate"],
            "Known drone windows": open_set_metrics["known_count"],
            "Unknown windows": open_set_metrics["unknown_count"],
        }]).style.format({
            "Unknown ROC-AUC": "{:.3f}",
            "Known acceptance": "{:.1%}",
            "Unknown rejection": "{:.1%}",
        }), hide_index=True, width="stretch")
        st.caption(
            "Temperature scaling and the known-drone acceptance threshold are fit on "
            "validation recordings. Phantom is the only held-out unknown signature."
        )
    st.markdown("#### Recorded RF replay · DroneRF PSD windows (not a live receiver)")
    feature_dir = DATA_DIR / "rf_features"
    required_cache = (
        feature_dir / "X.npy",
        feature_dir / "index.parquet",
        feature_dir / "config.json",
    )
    rf_features = None
    if not all(path.is_file() for path in required_cache):
        st.info(
            "Replay scores are unavailable because the prepared DroneRF feature cache "
            "is missing. After placing and verifying the 23 source archives, run "
            "`python scripts/prepare_rf.py`; then run `python scripts/eval_rf_trust.py` "
            "to produce calibrated research scores. This panel never connects to an SDR."
        )
    else:
        try:
            rf_features, rf_index, rf_artifact = rf_replay_resources()
        except (ValueError, OSError, KeyError) as e:
            rf_features = None
            st.error(f"The prepared RF cache or rf_trust model could not be used: {e}. "
                     "Re-run `python scripts/prepare_rf.py --process-only` and then "
                     "`python scripts/eval_rf_trust.py`.")
    if rf_features is not None:
        segment_index = rf_index[["bui", "segment"]].drop_duplicates()
        segment_options = [
            (str(row.bui), int(row.segment))
            for row in segment_index.itertuples(index=False)
        ]
        selected_segment = st.selectbox(
            "Recorded DroneRF segment",
            segment_options,
            format_func=lambda key: f"BUI {key[0]} · segment {key[1]}",
        )
        segment_rows = rf_index.loc[
            rf_index.bui.astype(str).eq(selected_segment[0])
            & rf_index.segment.eq(selected_segment[1])
        ].sort_values("start_s")
        positions = segment_rows.index.to_numpy(dtype=int)
        if len(positions) != 10:
            raise ValueError(
                f"expected 10 aligned 25 ms windows in this segment; found {len(positions)}"
            )
        band_label = st.radio(
            "PSD receiver band", ("Lower band", "Upper band"), horizontal=True
        )
        band_index = 0 if band_label == "Lower band" else 1
        selected_window = st.slider("Window within 250 ms segment", 1, 10, 1)
        selected_position = positions[selected_window - 1]
        psd = pd.DataFrame({
            "PSD bin": np.arange(rf_features.shape[2]),
            "Power (dB)": np.asarray(rf_features[selected_position, band_index]),
        })
        st.line_chart(psd, x="PSD bin", y="Power (dB)")
        st.caption(
            "Each selection replays one recorded 25 ms PSD window. The two bands are "
            "the dataset's lower/upper receiver bands; this is not live RF capture."
        )
        if rf_artifact is None:
            st.warning(
                "The PSD is available, but calibrated RF predictions are not: "
                "run `python scripts/eval_rf_trust.py` after preparing the cache."
            )
        else:
            segment_X = np.asarray(rf_features[positions])
            feature_values = RF_MODEL.band_features(
                RF_MODEL.shape_features(segment_X)
            )
            drone_probability = RF_CALIBRATION.temperature_scale(
                rf_artifact["binary_model"].predict_proba(feature_values),
                rf_artifact["binary_temperature"],
            )[:, 1]
            class_probabilities = RF_CALIBRATION.temperature_scale(
                rf_artifact["open_set_model"].predict_proba(feature_values),
                rf_artifact["open_set_temperature"],
            )
            known_indices = [
                rf_artifact["known_class_indices"][name]
                for name in ("bebop", "ar")
            ]
            unknownness = RF_OPEN_SET.unknownness_scores(
                class_probabilities, known_indices
            )
            recognized_labels = rf_artifact["open_set_model"].classes_[
                class_probabilities.argmax(axis=1)
            ]
            decisions = []
            for score, novelty in zip(drone_probability, unknownness):
                if score < rf_artifact["binary_threshold"]:
                    decisions.append("Background / below threshold")
                elif novelty > rf_artifact["unknownness_threshold"]:
                    decisions.append("Unknown-signature candidate")
                else:
                    decisions.append("Known-signature drone candidate")
            replay = pd.DataFrame({
                "Window": np.arange(1, len(positions) + 1),
                "Drone probability": drone_probability,
                "Unknownness": unknownness,
                "Model class": recognized_labels,
                "Research decision": decisions,
            })
            st.line_chart(
                replay,
                x="Window",
                y=["Drone probability", "Unknownness"],
            )
            st.dataframe(replay, hide_index=True, width="stretch")
            st.caption(
                "Thresholds and temperatures were fit on validation recordings only. "
                "Phantom is the sole held-out unknown model; all scores remain "
                "single-session research diagnostics."
            )
    st.markdown(
        '<div class="callout"><strong>Research caution — do not present these as real-world '
        'detection accuracy.</strong> On quiet windows with no visible transmission, the RF forest '
        f'still separates classes at AUC {report["quiet_probe"]["auc"]:.3f}. That is consistent '
        'with recording-session/receiver confounding. This dataset has one lab session and no '
        'external RF test corpus.</div>',
        unsafe_allow_html=True,
    )
    p = report["quiet_probe"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Quiet-window AUC · confounded probe", f"{p['auc']:.3f}")
    c2.metric("Quiet test windows", f"{p['test']:,}")
    c3.metric("External RF validation tiers", "None")
    st.dataframe(pd.DataFrame([
        ("Quiet-probe train windows", p["train"]),
        ("Quiet-probe test windows", p["test"]),
        ("Quiet drone windows", p["quiet_drone"]),
        ("Quiet background windows", p["quiet_background"]),
    ], columns=["Quiet-window diagnostic", "Count"]), hide_index=True, width="stretch")
    st.markdown("#### 0.25-second segment alarms · forest")
    st.dataframe(pd.DataFrame([
        {"Actual class": label, "Alarms": values[0], "Segments": values[1]}
        for label, values in report["segment_level"].items()
    ]), hide_index=True, width="stretch")

    rows = []
    names = {"tier1_test": "Held-out time", "tier2_bebop": "Unseen Bebop",
             "tier2_ar": "Unseen AR", "tier2_phantom": "Unseen Phantom"}
    for model, tiers in report["models"].items():
        if model == "forest+level":
            continue
        for tier, label in names.items():
            result = tiers[tier]
            rows.append({"Model": model, "Evaluation": label,
                         "PR-AUC": result["pr_auc"],
                         "PR-AUC 95% CI": result.get("pr_auc_ci95"),
                         "ROC-AUC": result["roc_auc"],
                         "Recall": result["recall"],
                         "FPR": result["fpr"], "Threshold": result.get("threshold"),
                         "False alarms": result.get("false_alarms"),
                         "Negative windows": result.get("negatives")})
    table = pd.DataFrame(rows)
    table["PR-AUC 95% CI"] = table["PR-AUC 95% CI"].map(
        lambda ci: f"{ci[0]:.3f}–{ci[1]:.3f}" if isinstance(ci, list) else "not reported")
    st.markdown("#### Model comparisons")
    st.dataframe(table.style.format({
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Recall": "{:.1%}",
        "FPR": "{:.2%}", "Threshold": "{:.3f}",
    }), hide_index=True, width="stretch")
    diagnostic_rows = []
    for tier, label in names.items():
        result = report["models"]["forest+level"][tier]
        diagnostic_rows.append({
            "Evaluation": label, "PR-AUC": result["pr_auc"], "ROC-AUC": result["roc_auc"],
            "Recall": result["recall"], "FPR": result["fpr"],
            "Threshold": result.get("threshold"),
        })
    with st.expander("Shortcut diagnostic · forest with absolute signal level (not a candidate model)"):
        st.dataframe(pd.DataFrame(diagnostic_rows).style.format({
            "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}",
            "Recall": "{:.1%}", "FPR": "{:.2%}", "Threshold": "{:.3f}",
        }), hide_index=True, width="stretch")
        st.caption(f"Level-only AUC: low band {report['level_probe_auc']['level_db_low']:.3f}; "
                   f"high band {report['level_probe_auc']['level_db_high']:.3f}.")
    st.markdown("#### Drone-type identification")
    classes = report["identify"]["classes"]
    confusion = pd.DataFrame(report["identify"]["confusion"], index=classes, columns=classes)
    confusion.index.name = "Actual ↓ / predicted →"
    st.dataframe(confusion, width="stretch")
    normalized = confusion.div(confusion.sum(axis=1).replace(0, np.nan), axis=0)
    st.markdown("#### Drone-type confusion matrix · row-normalized")
    st.dataframe(normalized.style.format("{:.1%}"), width="stretch")
    total_identified = int(confusion.to_numpy().sum())
    correct_identified = int(np.trace(confusion.to_numpy()))
    if total_identified:
        st.metric("Drone-type identification accuracy (same-session test)",
                  f"{correct_identified / total_identified:.1%}",
                  help="Descriptive only; it is not an independent-session deployment result.")
    if dronedetect_report is not None:
        render_dronedetect_subset(dronedetect_report)
    render_rf_asset_inventory()


def render_fusion_tab(fusion_report: dict | None):
    st.subheader("Audio + RF fusion · ablation")
    if fusion_report is None:
        st.info("Run `python scripts/eval_fusion.py` to generate the fusion ablation.")
        return
    render_fusion_highlights(fusion_report)
    scenario = st.selectbox("Compare methods in scenario", list(fusion_report["scenarios"]),
                            index=0, key="fusion_ablation_scenario")
    st.markdown(f"#### Full ablation · {scenario}")
    show_fusion_table(fusion_report, scenario)
    st.caption(
        "Bold row = deployed rule. Thresholds are selected on validation pairs (1% FPR). "
        "'Drones caught' and 'False alarms' give counts over the synthetic pairs. "
        "The four outputs are No Drone, Known Drone, Unknown Drone and Low Confidence."
    )
    render_md_report("fusion.md", "Full fusion report")


def render_methods_tab():
    st.subheader("Proposal-to-implementation checklist")
    rows = [
        ("Compare audio architectures", "Implemented", "CNN vs CNN+GRU vs CNN+LSTM vs lightweight transformer "
         "(audio_architecture_comparison.md, deep_audio.md), plus frozen pretrained encoder and classical baseline."),
        ("Use 2–4 second audio windows", "Different implementation", "1-second windows with a configurable hop; "
         "alarms need consecutive hits."),
        ("Compare RF baseline and deep model", "Implemented", "Forest / logistic regression vs 1D-CNN "
         "(rf_detection.md, rf_v3.md)."),
        ("Learned audio + RF fusion ablation", "Implemented (upper bound)", "Logistic-regression fusion and "
         "sum rule vs audio-only and RF-only (fusion.md). Pairs are synthetic: no co-recorded corpus exists."),
        ("Unknown / open-set recognition", "Implemented", "Audio (unseen Mambo) and RF (held-out drones); "
         "RF works for Bebop/Phantom, not AR."),
        ("Confidence and uncertainty", "Implemented", "Four states per window (drone / no drone / unknown / "
         "low confidence); RF probability calibration (rf_trust.md)."),
        ("Noise robustness", "Implemented", "Voice, traffic, wind, music, mixed, real speech and real music "
         "at 20/10/0 dB; synthetic white/pink noise test."),
        ("Simulated live-feed dashboard", "Implemented", "Audio replay with optional RF feed and fused state."),
        ("Grad-CAM", "Implemented", "Per-window spectrogram evidence overlay."),
        ("Cross-dataset evaluation", "Implemented", "Audio: outdoor Svanström; RF: Tampere recordings from a "
         "second receiver."),
        ("Confusion matrix, F1/accuracy", "Implemented", "Per-model and which-drone confusion matrices "
         "(identify.md)."),
        ("Temporal smoothing (stretch)", "Implemented", "Consecutive-hit alarm rule in the live feed; "
         "3-of-5 RF smoothing in rf_v3.md."),
        ("Real Wi-Fi / Bluetooth RF negatives", "Simulated only", "rf_v3.md uses synthetic emitters; "
         "real recordings (CardRF / DroneDetect) need a login."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Proposal requirement", "Status", "Implementation note"]),
                 hide_index=True, width="stretch")
    st.info("The proposal mentions accuracy/F1. Because the evaluation sets are imbalanced and differ "
            "from deployment prevalence, accuracy/F1 are shown alongside PR-AUC, ROC-AUC, recall and "
            "FPR but must not be used as standalone deployment claims.")


@st.cache_data(show_spinner="Decoding audio ...", max_entries=16)
def decode_clip(name: str, data: bytes) -> M.Clip:
    return M.decode_bytes(data, name)


@st.cache_data(show_spinner="Fetching audio from the link (this can take a minute) ...",
               max_entries=16, ttl=3600)
def fetch_clip(url: str) -> M.Clip:
    return M.fetch_url(url)


def _external_key(kind: str, clip: M.Clip) -> str:
    # Shape "<kind>:<title>:<n>"; render_live_monitor shows the title part.
    title = clip.title.replace(":", " ")
    return f"upload:{title}:{kind}-{len(clip.wav)}"


def _report_clip(clip: M.Clip, ui):
    ui.caption(f"Loaded **{clip.title}** · {clip.seconds:.0f} s analysed")
    if clip.truncated:
        ui.warning(f"Only the first {M.MAX_SECONDS // 60} minutes are analysed.")


SOURCE_DATASETS = ("Svanström outdoor", "DroneAudioDataset", "ITU ARIS Lab")
SOURCE_UPLOAD = "Upload audio or video"
SOURCE_LINK = "Paste a link (YouTube, …)"
SOURCE_MIC = "Record with microphone"


def recording_choice(manifest: pd.DataFrame | None, ui=None):
    ui = ui or st.sidebar
    ui.header("Audio input")
    ui.caption("The result appears in the **Replay detector** tab.")
    src = ui.radio(
        "Source", [*SOURCE_DATASETS, SOURCE_UPLOAD, SOURCE_LINK, SOURCE_MIC],
        horizontal=True,
        help="Dataset recordings, any audio or video file, a YouTube (or other site) link, "
             "or a recording from this computer's microphone. Video files are used for "
             "their sound only.")

    if src == SOURCE_UPLOAD:
        up = ui.file_uploader(
            "Audio or video file", type=M.UPLOAD_TYPES,
            help="WAV, MP3, M4A, FLAC, OGG, AAC, MP4, MOV, WEBM, MKV, AVI and more. "
                 "For example a video saved from YouTube or a phone recording.")
        if up is None:
            return None, None
        try:
            clip = decode_clip(up.name, up.getvalue())
        except M.MediaError as e:
            ui.error(str(e))
            return None, None
        _report_clip(clip, ui)
        return _external_key("file", clip), clip.wav

    if src == SOURCE_LINK:
        url = ui.text_input(
            "Link", placeholder="https://www.youtube.com/watch?v=…",
            help="YouTube and most video sites (via yt-dlp), or a direct link to an audio or "
                 "video file. Use only clips you are allowed to use.")
        if not url.strip():
            ui.caption("Paste a link and press Enter.")
            return None, None
        try:
            clip = fetch_clip(url.strip())
        except M.MediaError as e:
            ui.error(str(e))
            return None, None
        _report_clip(clip, ui)
        return _external_key("link", clip), clip.wav

    if src == SOURCE_MIC:
        rec = ui.audio_input(
            "Record", help="Allow microphone access in the browser, then record a few seconds. "
                           "Hold a phone playing a drone video near the microphone to try it.")
        if rec is None:
            return None, None
        try:
            clip = decode_clip("microphone recording.wav", rec.getvalue())
        except M.MediaError as e:
            ui.error(str(e))
            return None, None
        _report_clip(clip, ui)
        return _external_key("mic", clip), clip.wav

    if manifest is None:
        ui.error("Audio manifest unavailable. Run scripts/build_manifest.py.")
        return None, None

    source_name = {
        "Svanström outdoor": "svanstrom",
        "DroneAudioDataset": "droneaudio",
        "ITU ARIS Lab": "itu_aris",
    }[src]
    rows = manifest[manifest["source"] == source_name].copy()
    if rows.empty:
        ui.warning(f"{src} is not present in the audio manifest.")
        return None, None

    if source_name == "svanstrom":
        kind = ui.selectbox("Category", ["DRONE", "HELICOPTER", "BACKGROUND"])
        if kind == "DRONE":
            rows = rows[rows["label"] == "drone"]
        elif kind == "HELICOPTER":
            rows = rows[rows["bg_category"] == "helicopter"]
        else:
            rows = rows[rows["bg_category"] == "outdoor_background"]
    else:
        label = ui.selectbox("Label", ["drone", "background"])
        rows = rows[rows["label"] == label]
        if source_name == "droneaudio":
            category_column = "drone_model" if label == "drone" else "bg_category"
            rows = rows.copy()
            rows[category_column] = rows[category_column].fillna("").astype(str)
            categories = sorted(rows[category_column].unique())
            category_labels = {
                category: category or "uncategorized / noise"
                for category in categories
            }
            category = ui.selectbox(
                "Drone model" if label == "drone" else "Background category",
                categories,
                format_func=lambda value: category_labels[value],
            )
            rows = rows[rows[category_column] == category]
        elif label == "background":
            categories = sorted(rows["bg_category"].fillna("").astype(str).unique())
            category = ui.selectbox("Background category", categories)
            rows = rows[rows["bg_category"].fillna("").astype(str) == category]

    recordings = rows.drop_duplicates("path").sort_values("path")
    ui.caption(f"{len(recordings):,} source recordings/clips in this selection.")
    relative_paths = recordings["path"].astype(str).tolist()
    selected = ui.selectbox(
        "Recording / clip",
        [None, *relative_paths],
        format_func=lambda value: (
            "Choose a recording..." if value is None else Path(value).stem
        ),
    )
    if selected is None:
        return None, None
    path = DATA_DIR / selected
    if not path.is_file():
        ui.error(f"Audio file listed in the manifest is missing: {path}")
        return None, None
    return str(path), path.read_bytes()


def sensor_card(title: str, badge: str, badge_class: str, lines: list[str]):
    body = "".join(f"<p>{line}</p>" for line in lines)
    st.markdown(f'<div class="sensor"><h4>{title} <span class="badge {badge_class}">{badge}</span>'
                f"</h4>{body}</div>", unsafe_allow_html=True)


def result_panel(verdict: S.Verdict, basis: str):
    st.markdown(
        f'<div class="result" role="status" aria-live="polite" '
        f'style="--state:{STATE_COLOR[verdict.state]}">'
        f'<div class="kicker">Recording-level result · offline replay · not field-validated</div>'
        f'<div class="state">{STATE_ICON[verdict.state]} {verdict.label}</div>'
        f"<p><strong>Why:</strong> {verdict.reason}.</p>"
        f'<p class="meaning"><strong>Meaning:</strong> {S.STATE_MEANING[verdict.state]}</p>'
        f"<p><strong>Based on:</strong> {basis}</p></div>",
        unsafe_allow_html=True,
    )


def sidebar_inputs(manifest: pd.DataFrame | None) -> dict:
    """Audio input, RF feed and replay settings, all in the sidebar."""
    key, audio = recording_choice(manifest)
    feed, rf_src, fusion_method = rf_choice()
    st.sidebar.header("Replay settings")
    return dict(
        key=key, audio=audio, feed=feed, rf_src=rf_src, fusion_method=fusion_method,
        hop=st.sidebar.select_slider("Window hop (s)", [0.25, 0.5, 1.0], value=0.5),
        min_hits=int(st.sidebar.number_input("Consecutive drone windows to alarm", 1, 5, 2)),
        layer_name=st.sidebar.radio("Grad-CAM resolution", list(X.LAYERS)),
        speed=st.sidebar.select_slider("Replay speed", [1.0, 2.0, 4.0, 8.0], value=2.0,
                                       format_func=lambda s: f"{s:g}x"),
    )


def gauge(value: float, lo: float, hi: float, threshold: float, color: str) -> str:
    """A bar for `value` on [lo, hi] with a tick at `threshold`."""
    def pos(v):
        return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0))
    return (f'<div class="gauge" role="img" aria-label="value {value:.2f}, threshold '
            f'{threshold:.2f}"><span style="width:{pos(value):.1f}%;background:{color}"></span>'
            f'<i style="left:{pos(threshold):.1f}%"></i></div>')


def decision_tile(kind: str, verdict: S.Verdict | None, bar: str, lines: list[str],
                  final: bool = False, off_reason: str = "") -> str:
    if verdict is None:
        return (f'<div class="tile off" style="--state:{STATE_COLOR["no drone"]}">'
                f'<div class="k">{kind}</div><div class="v">Not used</div>'
                f"<p>{off_reason}</p></div>")
    body = "".join(f"<p>{line}</p>" for line in lines)
    return (f'<div class="tile{" final" if final else ""}" '
            f'style="--state:{STATE_COLOR[verdict.state]}"><div class="k">{kind}</div>'
            f'<div class="v">{STATE_ICON[verdict.state]} {verdict.label}</div>{bar}{body}</div>')


def state_strip(df: pd.DataFrame, hop: float, use_rf: bool):
    """One coloured row per sensor: each window's state over time."""
    rows = [("Audio", df.state)]
    if use_rf:
        rows += [("RF", df.rf_state), ("Fused = final", df.fused_state)]
    order = [name for name, _ in rows]
    long = pd.concat([pd.DataFrame({"t": df.t, "t2": df.t + hop, "Sensor": name,
                                    "State": states.map(S.state_label)})
                      for name, states in rows], ignore_index=True)
    return alt.Chart(long).mark_rect(stroke=None).encode(
        x=alt.X("t:Q", title="time (s)"), x2="t2:Q",
        y=alt.Y("Sensor:N", sort=order, title=None),
        color=alt.Color("State:N", title="state",
                        scale=alt.Scale(domain=[FINAL_STATE_LABEL[k] for k in STATE_COLOR],
                                        range=list(STATE_COLOR.values()))),
        tooltip=[alt.Tooltip("t:Q", title="time (s)"), "Sensor:N", "State:N"],
    ).properties(height=34 * len(rows) + 30)


def render_live_monitor(manifest: pd.DataFrame | None, inputs: dict | None = None):
    st.subheader("Replay detector")
    st.caption("Choose the audio in the sidebar. The trained detector runs **automatically** "
               "as soon as a clip is loaded, and the result appears here. Nothing is retrained: "
               "the clip is only evaluated. This is an offline replay, not a live sensor feed.")
    inputs = inputs if inputs is not None else sidebar_inputs(manifest)
    key, audio = inputs["key"], inputs["audio"]
    feed, rf_src, fusion_method = inputs["feed"], inputs["rf_src"], inputs["fusion_method"]
    hop, min_hits = inputs["hop"], inputs["min_hits"]
    layer_name, speed = inputs["layer_name"], inputs["speed"]
    if audio is None:
        st.info("Choose the audio in the sidebar → **Audio input**: a dataset recording, any "
                "audio or video file (MP3, MP4, M4A, WAV, …), a YouTube or other link, or a "
                "microphone recording. Only the sound is analysed.")
        return

    try:
        det = detector()
        df, clips, sig = analyse(key, audio, hop)
    except FileNotFoundError as e:
        st.error(f"The audio detector cannot run: {e}. Run `python scripts/eval_uncertainty.py` "
                 "(and `python scripts/download_data.py` for the encoder weights).")
        return
    except Exception as e:                       # unreadable upload, decoder errors
        st.error(f"Could not analyse this audio ({type(e).__name__}: {e}). Try a WAV file.")
        return
    df = df.copy()
    use_rf = feed is not None and rf_src != NO_RF
    psd = None
    if use_rf:
        psd, a_rf = feed.windows(rf_src, len(df))
        selected_fusion = feed.fusions[fusion_method]
        fz, fthr = L.fuse(df, selected_fusion, a_rf["p"],
                          {k: a_rf[k] for k in L.AUDIO_FLAGS})
        df = pd.concat([df, fz], axis=1)
        df["rf_p"] = a_rf["p"]
        df["rf_state"] = a_rf["state"]
        df["shown_state"], df["shown_reason"] = df.fused_state, df.fused_reason
        df["fused_state_label"] = df.fused_state.map(S.state_label)
    else:
        df["shown_state"], df["shown_reason"] = df.state, df.reason
    df["shown_label"] = df.shown_state.map(S.state_label)
    verdict = S.recording_verdict(df.shown_state, min_hits)
    title = Path(key).stem if not key.startswith("upload") else key.split(":")[1]

    # 1. The result, first.
    if use_rf:
        rule = ("evidence sum (deployed rule)" if fusion_method == "sum"
                else "learned fusion layer (experimental; misses RF-silent drones)")
        basis = (f"audio from <em>{title}</em> + recorded RF test windows "
                 f"(<em>{rf_src}</em>), combined by the {rule}. Unpaired what-if.")
    else:
        basis = f"audio only, from <em>{title}</em>. Not a multimodal decision."
    result_panel(verdict, basis)
    if use_rf:
        st.markdown(
            '<div class="callout"><strong>Not a synchronized multimodal decision.</strong> '
            "The RF windows are recorded test data of the chosen class, replayed one per audio "
            "window. They were never recorded with this audio, and no RF receiver is "
            "connected.</div>", unsafe_allow_html=True)

    # 2. Evidence per sensor, as a visual flow: Audio + RF -> Final.
    audio_v = S.recording_verdict(df.state, min_hits)
    audio_tile = decision_tile(
        "Audio sensor", audio_v,
        gauge(float(df.p.max()), 0.0, 1.0, det.threshold, STATE_COLOR[audio_v.state]),
        [f"Highest window score <b>{df.p.max():.2f}</b> · threshold {det.threshold:.2f}",
         f"{audio_v.drone_windows} of {len(df)} windows drone-like"])
    if use_rf:
        rf_v = S.recording_verdict(df.rf_state, min_hits)
        rf_tile = decision_tile(
            "RF sensor (recorded replay)", rf_v,
            gauge(float(df.rf_p.max()), 0.0, 1.0, feed.det.threshold, STATE_COLOR[rf_v.state]),
            [f"Highest window score <b>{df.rf_p.max():.2f}</b> · threshold "
             f"{feed.det.threshold:.2f}",
             f"{rf_v.drone_windows} of {len(df)} windows drone-like"])
        lo, hi = -10.0, 10.0
        final_tile = decision_tile(
            "Final · audio + RF fused", verdict,
            gauge(float(df.fused.max()), lo, hi, fthr, STATE_COLOR[verdict.state]),
            [f"Highest fused evidence <b>{df.fused.max():+.1f}</b> · threshold {fthr:+.1f}",
             f"Rule: {'evidence sum (deployed)' if fusion_method == 'sum' else 'learned layer (experimental)'}"],
            final=True)
    else:
        reason = rf_feed()
        rf_tile = decision_tile(
            "RF sensor", None, "", [], off_reason=(
                "No RF source selected (sidebar → RF feed)." if not isinstance(reason, str)
                else "RF model files not built on this computer."))
        final_tile = decision_tile(
            "Final · audio only", verdict,
            gauge(float(df.p.max()), 0.0, 1.0, det.threshold, STATE_COLOR[verdict.state]),
            ["Same as the audio sensor: no RF evidence was used."], final=True)
    st.markdown(f'<div class="flow">{audio_tile}<div class="op">+</div>{rf_tile}'
                f'<div class="op">=</div>{final_tile}</div>', unsafe_allow_html=True)
    st.caption("Bars: the highest window score, with the threshold as a tick. Scores are model "
               "scores, not calibrated probabilities. A recording alarms after "
               f"{min_hits} consecutive drone-like windows; isolated hits give Low Confidence.")
    st.markdown("##### Window by window")
    st.altair_chart(state_strip(df, hop, use_rf), width="stretch")

    m = st.columns(5)
    m[0].metric("Windows scored", len(df))
    m[1].metric("Drone-state windows", verdict.drone_windows)
    m[2].metric("Longest drone run", f"{verdict.longest_drone_run} (need {min_hits})")
    m[3].metric("Low-confidence windows", verdict.low_windows)
    m[4].metric("Unknown-signature windows", verdict.unknown_windows)
    state_counts = df.shown_label.value_counts()
    st.caption("Windows per state" + (" (fused)" if use_rf else " (audio)") + ": "
               + " · ".join(f"{S.STATE_LABEL[s]} {int(state_counts.get(S.STATE_LABEL[s], 0))}"
                            for s in S.STATES))

    live_slot = st.empty()
    if st.button("▶ Replay window by window", type="primary",
                 help="Steps through the windows at the chosen speed."):
        for i in range(len(df)):
            r = df.iloc[i]
            with live_slot.container():
                c1, c2 = st.columns([1, 3])
                c1.metric("time", f"{r.t:.1f} s")
                c1.markdown(f"### {STATE_ICON.get(r.shown_state, '🟡')} {r.shown_label}")
                c1.caption(r.shown_reason or r.tags)
                c2.altair_chart(timeline(df, det.threshold, upto=i), width="stretch")
            time.sleep(hop / speed)
        live_slot.empty()
    st.audio(to_wav(sig), format="audio/wav")

    # 3. Single-window drill-down.
    st.markdown("#### Single-window prediction")
    st.caption("One 1-second window. A single window is never an alarm on its own; the "
               "recording-level result above applies the consecutive-window rule.")
    i = st.slider("Inspect window", 0, len(df) - 1, int(df.p.idxmax()),
                  format="%d", help="Defaults to the highest-scoring window")
    st.altair_chart(timeline(df, det.threshold, selected=i), width="stretch")
    if use_rf:
        st.caption(f"Audio score above; fused evidence below (RF: {rf_src}). "
                   "Dashed line = threshold, 1% false alarms on validation data.")
        st.altair_chart(fused_timeline(df, fthr), width="stretch")
    r = df.iloc[i]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"##### {STATE_ICON.get(r.shown_state, '🟡')} {r.shown_label} · "
                    f"{r.t:.1f}–{r.t + 1:.1f} s")
        if r.shown_reason:
            st.caption(f"Flagged: {r.shown_reason}")
        if use_rf:
            st.dataframe(pd.DataFrame({
                "evidence (log-likelihood ratio, capped)": ["audio", "RF", "fused"],
                "value": [f"{r.llr_audio:+.2f}", f"{r.llr_rf:+.2f}", f"{r.fused:+.2f}"]}),
                hide_index=True, width="stretch")
            st.caption(f"Audio-only state: {S.state_label(r.state)}. "
                       f"RF score {r.rf_p:.3f} ({S.state_label(r.rf_state)} on RF alone).")
        st.metric("Audio model score", f"{r.p:.3f}", f"{r.p - det.threshold:+.3f} vs threshold",
                  delta_color="off")
        extra = {k: r[k] for k in ("margin", "distance", "spread") if k in r}
        if extra:
            st.dataframe(pd.DataFrame({
                "signal": ["logit margin to threshold", "distance from training data",
                           "ensemble spread (logit)"][:len(extra)],
                "value": [f"{v:.2f}" for v in extra.values()]}),
                hide_index=True, width="stretch")
        st.markdown("**Encoder hears:** " + r.tags)
        st.audio(to_wav(clips[i]), format="audio/wav")
    with right:
        mel, cam, _ = cam_for(key, i, clips[i], X.LAYERS[layer_name])
        figure = cam_figure(mel, cam)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption("Bright regions pushed the score towards 'drone'. A window scored "
                   "'no drone' can still show weak evidence.")
        if use_rf:
            rf_fig = rf_figure(psd[i])
            st.pyplot(rf_fig, width="stretch")
            plt.close(rf_fig)
            st.caption("Recorded RF snapshot replayed with this window (25 ms, both halves "
                       "of the 2.4 GHz band, level removed).")
    with st.expander("All windows"):
        st.dataframe(df.drop(columns=["hit", "shown_state", "shown_reason",
                                      *[c for c in L.AUDIO_FLAGS if c in df]]).round(3),
                     hide_index=True, width="stretch")


def main():
    theme_control()
    dronedetect_report_path = REPORTS / "dronedetect_subset.json"
    rf_trust_report_path = REPORTS / "rf_trust.json"
    reports = {
        "baseline": read_report("baseline_audio.json"),
        "pretrained": read_report("pretrained_audio.json"),
        "outdoor": read_report("outdoor_audio.json"),
        "svanstrom_confusers": read_report("svanstrom_confusers.json"),
        "audio_architectures": (
            json.loads((REPORTS / "audio_architecture_comparison.json").read_text(
                encoding="utf-8"
            ))
            if (REPORTS / "audio_architecture_comparison.json").exists() else None
        ),
        "noise": read_report("noise_robustness.json"),
        "uncertainty": read_report("uncertainty_audio.json"),
        "rf": read_report("rf_detection.json"),
        "rf_v2": (
            json.loads((REPORTS / "rf_v2.json").read_text(encoding="utf-8"))
            if (REPORTS / "rf_v2.json").exists() else None
        ),
        "fusion": (
            json.loads((REPORTS / "fusion.json").read_text(encoding="utf-8"))
            if (REPORTS / "fusion.json").exists() else None
        ),
        "audio_baseline": read_report("audio_baseline.json"),
        "audio_cnn": read_report("audio_cnn.json"),
        "audio_noise": read_report("audio_noise_robustness.json"),
        "training_curves": read_report("training_curves.json"),
        "dronedetect_subset": (
            json.loads(dronedetect_report_path.read_text(encoding="utf-8"))
            if dronedetect_report_path.exists() else None
        ),
        "rf_trust": (
            json.loads(rf_trust_report_path.read_text(encoding="utf-8"))
            if rf_trust_report_path.exists() else None
        ),
    }
    report_files = {
        "baseline": "baseline_audio.json", "pretrained": "pretrained_audio.json",
        "outdoor": "outdoor_audio.json", "noise": "noise_robustness.json",
        "uncertainty": "uncertainty_audio.json", "rf": "rf_detection.json",
        "audio_baseline": "audio_baseline.json", "audio_cnn": "audio_cnn.json",
        "audio_noise": "audio_noise_robustness.json",
        "training_curves": "training_curves.json",
    }
    missing = [f for k, f in report_files.items() if reports[k] is None]
    audio_keys = ("baseline", "pretrained", "outdoor", "noise", "uncertainty",
                  "audio_baseline", "audio_cnn", "audio_noise", "training_curves")
    manifest_path = DATA_DIR / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None

    inputs = sidebar_inputs(manifest)
    hero()
    if inputs["audio"] is not None:
        title = (Path(inputs["key"]).stem if not inputs["key"].startswith("upload")
                 else inputs["key"].split(":")[1])
        st.success(f"**{title}** is loaded. Open the **Replay detector** tab to see the "
                   "result (it is computed automatically).")
    if missing:
        st.warning("Some evaluation reports are missing from reports/, so the tabs that use "
                   "them are limited: " + ", ".join(f"`{m}`" for m in missing)
                   + ". Re-run the evaluation scripts listed in README.md.")
    tabs = st.tabs(["Project briefing", "Replay detector", "Fusion",
                    "Audio intelligence", "RF research", "Proposal checklist"])
    with tabs[0]:
        guarded("Project briefing", render_overview, reports, manifest)
    with tabs[1]:
        guarded("Replay detector", render_live_monitor, manifest, inputs)
    with tabs[2]:
        guarded("Fusion", render_fusion_tab, reports["fusion"])
    with tabs[3]:
        if any(reports[k] is None for k in audio_keys):
            st.info("Audio results need the audio reports listed above.")
        else:
            guarded("Audio intelligence", render_audio_tab, reports)
    with tabs[4]:
        if reports["rf"] is None:
            st.info("RF results need `reports/rf_detection.json` "
                    "(`python scripts/train_rf.py`).")
        else:
            guarded("RF research", render_rf_tab, reports["rf"],
                    reports["dronedetect_subset"], reports["rf_trust"])
    with tabs[5]:
        guarded("Proposal checklist", render_methods_tab)


def guarded(section: str, render, *args):
    """Render a tab; a failure shows an actionable message, not a stack trace,
    and the other tabs keep working."""
    try:
        render(*args)
    except Exception as e:                                   # noqa: BLE001
        st.error(f"{section} could not be displayed: {type(e).__name__}: {e}. "
                 "This usually means a report or artifact is missing or out of date; "
                 "see the readiness table in Project briefing.")

main()
