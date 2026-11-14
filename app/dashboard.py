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
from dronedet.eval import calibration as RF_CALIBRATION
from dronedet.eval import open_set as RF_OPEN_SET
from dronedet.models import explain as X
from dronedet.models import rf as RF_MODEL
from dronedet.paths import DATA_DIR

REPORTS = ROOT / "reports"
STATE_COLOR = {"drone": "#d6452a", "unknown drone": "#7a4fd6",
               "low confidence": "#d99a0b", "no drone": "#8a8984"}
STATE_ICON = {"drone": "🔴", "unknown drone": "🟣", "low confidence": "🟡", "no drone": "⚪"}
FINAL_STATE_LABEL = {
    "no drone": "No Drone",
    "drone": "Known Drone",
    "unknown drone": "Unknown Drone",
    "low confidence": "Low Confidence",
}

st.set_page_config(page_title="Skyguard · Counter-drone Lab", page_icon="🛰️",
                   layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #e7f2ff;
        --muted: #91a5bd;
        --panel: rgba(15, 29, 49, .88);
        --line: rgba(97, 190, 255, .20);
        --cyan: #54dcff;
        --violet: #9c83ff;
        --green: #63efc0;
        --amber: #ffcb75;
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 12% 0%, rgba(19, 122, 176, .21), transparent 38%),
            radial-gradient(ellipse at 92% 10%, rgba(104, 67, 176, .18), transparent 34%),
            #07111f;
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .75); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2d, #091322);
        border-right: 1px solid var(--line);
    }
    .block-container { max-width: 1440px; padding-top: 1.8rem; }
    h1, h2, h3 { letter-spacing: -.035em; }
    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 14px; padding: 14px 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,.16);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted); font-size: .78rem; text-transform: uppercase;
        letter-spacing: .09em;
    }
    .hero {
        position: relative; overflow: hidden; padding: clamp(22px, 4vw, 42px);
        border: 1px solid rgba(84,220,255,.32); border-radius: 22px;
        background: linear-gradient(115deg, rgba(14,39,62,.97), rgba(22,25,55,.94));
        box-shadow: 0 18px 70px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.05);
        margin: 0 0 1.2rem;
    }
    .hero:after {
        content: ""; position: absolute; width: 290px; height: 290px;
        right: -80px; top: -150px; border-radius: 50%;
        border: 1px solid rgba(84,220,255,.24);
        box-shadow: 0 0 0 24px rgba(84,220,255,.035), 0 0 0 52px rgba(156,131,255,.035);
    }
    .eyebrow {
        color: var(--cyan); text-transform: uppercase; letter-spacing: .18em;
        font: 700 .72rem/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .hero h1 { margin: .35rem 0 .45rem; color: #f3f8ff; font-size: clamp(2rem, 5vw, 3.25rem); }
    .hero p { color: #a9bdd3; max-width: 760px; margin: 0; line-height: 1.65; }
    .pill {
        display: inline-block; border: 1px solid var(--line); border-radius: 999px;
        padding: 5px 10px; margin: 14px 7px 0 0; color: #c6d9ec;
        background: rgba(7,17,31,.45); font-size: .76rem;
    }
    .panel {
        height: 100%; padding: 18px; border-radius: 16px;
        background: var(--panel); border: 1px solid var(--line);
        box-shadow: 0 10px 35px rgba(0,0,0,.14);
    }
    .panel .tag {
        font: 700 .68rem/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
        color: var(--cyan); letter-spacing: .12em; text-transform: uppercase;
    }
    .panel h3 { margin: 8px 0 6px; color: #edf6ff; font-size: 1.12rem; }
    .panel p { color: #9db1c8; font-size: .91rem; line-height: 1.55; margin: 0; }
    .panel.pending { border-color: rgba(255,203,117,.43); }
    .panel.pending .tag { color: var(--amber); }
    .panel.ready .tag { color: var(--green); }
    .callout {
        padding: 15px 18px; margin: 10px 0 18px;
        background: rgba(255,176,77,.09);
        border: 1px solid rgba(255,203,117,.35); border-left: 4px solid var(--amber);
        border-radius: 12px; color: #f5dfb9; line-height: 1.55;
    }
    .stage-flow {
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px; margin: 12px 0 22px;
    }
    .stage {
        padding: 16px; border: 1px solid var(--line); border-radius: 14px;
        background: linear-gradient(145deg, rgba(18,39,62,.95), rgba(14,25,43,.9));
    }
    .stage small { color: var(--cyan); letter-spacing: .1em; text-transform: uppercase; }
    .stage strong { display: block; color: #eff8ff; margin: 7px 0; font-size: 1.05rem; }
    .stage span { color: #9db1c8; font-size: .88rem; line-height: 1.5; }
    [data-testid="stTabs"] button { font-weight: 650; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--cyan); border-bottom-color: var(--cyan);
    }
    [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] {
        border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    }
    @media (max-width: 760px) {
        .block-container { padding: 1rem .75rem 2rem; }
        .stage-flow { grid-template-columns: 1fr; }
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .55rem; }
        div[data-testid="column"] { flex: 1 1 100%; min-width: 100%; }
        [data-testid="stMetric"] { padding: 11px 12px; }
        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto; flex-wrap: nowrap; gap: .25rem;
        }
        [data-testid="stTabs"] button { white-space: nowrap; font-size: .84rem; }
    }
    @media (min-width: 761px) and (max-width: 1050px) {
        div[data-testid="column"] { min-width: 30%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    base = alt.Chart(d).encode(x=alt.X("t:Q", title="time (s)",
                                       scale=alt.Scale(domain=[0, float(df.t.max()) + 1])))
    bars = base.mark_bar(size=8, cornerRadiusTopLeft=2,
                         cornerRadiusTopRight=2).encode(
        y=alt.Y("p:Q", title="drone score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("state:N", title="state",
                        scale=alt.Scale(domain=list(STATE_COLOR),
                                        range=list(STATE_COLOR.values()))),
        tooltip=["t", alt.Tooltip("p:Q", format=".3f"), "state", "reason", "tags"])
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(
        strokeDash=[4, 4], color="#52514e").encode(y="thr:Q")
    layers = [bars, rule]
    if selected is not None and selected < len(d):
        sel = alt.Chart(d.iloc[[selected]]).mark_point(
            shape="triangle-down", size=120, filled=True, color="#0b0b0b").encode(
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
        st.warning(f"Report unavailable: `{path}`. Run the corresponding evaluation script.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    except AttributeError as e:
        return (
            "The live module is stale or incompatible. Restart Streamlit from the "
            f"project directory. Details: {e}"
        )


def rf_choice():
    st.sidebar.header("RF feed")
    feed = rf_feed()
    if isinstance(feed, str):
        st.sidebar.caption(f"RF unavailable: {feed}")
        return None, NO_RF, None
    src = st.sidebar.selectbox("Replay RF windows from", [NO_RF, *feed.sources],
                               help="Held-out windows of this class, one 25 ms "
                                    "snapshot per audio window. A what-if: audio and "
                                    "RF were never recorded together.")
    methods = list(feed.fusions)
    if "learned" in methods and "sum" in methods:
        method = st.sidebar.radio(
            "Fusion rule",
            ["learned", "sum"],
            format_func=lambda value: (
                "Learned fusion layer" if value == "learned"
                else "Evidence sum (current research default)"
            ),
            help=(
                "Both rules are research replays on unpaired audio/RF recordings. "
                "The demo opens in learned mode. The project's selected research "
                "rule remains evidence sum because learned fusion nearly misses "
                "RF-silent drones in the reported experiment."
            ),
        )
    else:
        method = methods[0]
        st.sidebar.caption(f"Fusion rule available: {method}")
    return feed, src, method


def fused_timeline(df: pd.DataFrame, thr: float):
    base = alt.Chart(df).encode(x=alt.X("t:Q", title="time (s)"))
    bars = base.mark_bar(size=8).encode(
        y=alt.Y("fused:Q", title="fused evidence (LLR)"),
        color=alt.Color("fused_state_label:N", title="final multimodal state"),
        tooltip=["t", alt.Tooltip("llr_audio:Q", format=".2f"),
                 alt.Tooltip("llr_rf:Q", format=".2f"),
                 alt.Tooltip("fused:Q", format=".2f"),
                 "fused_state_label", "fused_reason"])
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(
        strokeDash=[4, 4], color="#52514e").encode(y="thr:Q")
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
        <div class="hero">
          <div class="eyebrow">MULTI-SENSOR RESEARCH CONSOLE · AUDIO / RF</div>
          <h1>SKYGUARD <span style="color:#54dcff">/</span> COUNTER-DRONE LAB</h1>
          <p>One evidence-led view of acoustic detection, radio-frequency experiments,
          robustness, uncertainty and the next step toward multimodal fusion.</p>
          <span class="pill">SIMULATED FEED</span>
          <span class="pill">HUMAN-IN-THE-LOOP</span>
          <span class="pill">RESEARCH PROTOTYPE</span>
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


def precision_from_rates(prevalence: float, recall: float, fpr: float) -> float:
    true_positive_rate = prevalence * recall
    false_positive_rate = (1 - prevalence) * fpr
    denominator = true_positive_rate + false_positive_rate
    return true_positive_rate / denominator if denominator else 0.0


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
    st.caption(
        "Precision, F1 and accuracy in the table are derived from the report's prevalence, "
        "recall and FPR; per-example confusion matrices are only available for the supplemental "
        "Tier 1 baseline/CNN run."
    )
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
    display["Precision"] = display.apply(
        lambda row: precision_from_rates(row["prevalence"], row["recall"], row["fpr"]),
        axis=1,
    )
    display["F1"] = display.apply(
        lambda row: (
            2 * row["Precision"] * row["recall"] / (row["Precision"] + row["recall"])
            if row["Precision"] + row["recall"] else 0.0
        ),
        axis=1,
    )
    display["Accuracy"] = (
        display["prevalence"] * display["recall"]
        + (1 - display["prevalence"]) * (1 - display["fpr"])
    )
    display["PR-AUC (95% CI)"] = display["pr_auc_ci95"].map(
        lambda ci: f"{ci[0]:.3f}–{ci[1]:.3f}")
    display = display.drop(columns=["pr_auc_ci95", "pr_auc"])
    display = display.rename(columns={
        "Tier": "Evaluation tier", "threshold": "Frozen threshold",
        "prevalence": "Drone prevalence", "roc_auc": "ROC-AUC",
        "recall": "Recall", "fpr": "FPR", "false_alarms": "False alarms",
        "negatives": "Negative windows",
    })
    st.dataframe(display.style.format({
        "Frozen threshold": "{:.3f}", "Drone prevalence": "{:.1%}",
        "ROC-AUC": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}",
        "F1": "{:.1%}", "Accuracy": "{:.1%}", "FPR": "{:.2%}",
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
            "Accuracy": (
                protocol["test_clips"]
                - result["false_positives"]
                - result["false_negatives"]
            ) / protocol["test_clips"],
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
        "Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}",
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


def fusion_ablation_rows(report: dict, scenario: str) -> pd.DataFrame:
    methods = report["scenarios"][scenario]
    labels = {
        "audio only": "Audio only",
        "RF only": "RF only",
        "sum": "Evidence sum",
        "learned": "Learned fusion layer",
        "OR": "OR rule",
        "AND": "AND rule",
    }
    preferred_order = ("audio only", "RF only", "sum", "learned", "OR", "AND")
    rows = []
    for method in preferred_order:
        if method not in methods:
            continue
        metrics = methods[method]
        rows.append({
            "Method": labels[method],
            "PR-AUC": metrics.get("pr_auc"),
            "Recall": metrics.get("recall"),
            "False-positive rate": metrics.get("fpr"),
            "Accuracy": metrics.get("accuracy"),
            "F1": metrics.get("f1"),
        })
    return pd.DataFrame(rows)


def _highlight_learned_fusion(row: pd.Series) -> list[str]:
    color = "background-color: rgba(84, 220, 255, 0.16); font-weight: 700"
    return [color if row["Method"] == "Learned fusion layer" else "" for _ in row]


def render_fusion_highlights(report: dict | None):
    st.markdown("### Learned fusion layer · final multimodal result")
    if report is None:
        st.info("Fusion evaluation report is unavailable; run `python scripts/eval_fusion.py`.")
        return

    scenario = "both sensors" if "both sensors" in report["scenarios"] else next(
        iter(report["scenarios"])
    )
    methods = report["scenarios"][scenario]
    learned = methods.get("learned")
    audio = methods.get("audio only")
    rf = methods.get("RF only")
    st.markdown(
        '<div class="callout"><strong>Research result, not a validated field decision.</strong> '
        'The learned logistic-regression fusion layer combines calibrated audio and RF '
        'evidence. The test pairs are independently sampled by class because the datasets '
        'do not contain synchronized audio/RF recordings. The current replay default remains '
        'the evidence-sum rule; select the learned layer in the Live feed to inspect its '
        'four-state output.</div>',
        unsafe_allow_html=True,
    )
    if learned is not None:
        cards = st.columns(3)
        for column, title, metrics in (
            (cards[0], "Audio only", audio),
            (cards[1], "RF only", rf),
            (cards[2], "Learned fusion · both sensors", learned),
        ):
            if metrics is None:
                continue
            column.markdown(f"**{title}**")
            column.metric("Recall", f"{metrics['recall']:.1%}")
            column.caption(
                f"FPR {metrics['fpr']:.2%} · PR-AUC "
                f"{metrics['pr_auc']:.3f}" if metrics.get("pr_auc") is not None
                else f"FPR {metrics['fpr']:.2%}"
            )
    state_columns = st.columns(4)
    state_descriptions = (
        ("⚪ No Drone", "Fused evidence is below the calibrated decision threshold."),
        ("🔴 Known Drone", "Positive evidence; no unknown-signature flag is raised."),
        ("🟣 Unknown Drone", "Positive evidence with an unfamiliar drone signature."),
        ("🟡 Low Confidence", "Near the review boundary or unfamiliar input; request review."),
    )
    for column, (label, description) in zip(state_columns, state_descriptions):
        column.markdown(f"**{label}**")
        column.caption(description)
    st.caption(
        f"Shown scenario: {scenario}. These are synthetic-pair test results, not "
        "performance on synchronized flight captures."
    )
    table = fusion_ablation_rows(report, scenario)
    st.dataframe(
        table.style.apply(_highlight_learned_fusion, axis=1).format({
            "PR-AUC": "{:.3f}",
            "Recall": "{:.1%}",
            "False-positive rate": "{:.2%}",
            "Accuracy": "{:.1%}",
            "F1": "{:.3f}",
        }, na_rep="n/a"),
        hide_index=True,
        width="stretch",
    )


def render_overview(reports: dict, manifest: pd.DataFrame | None):
    st.markdown(
        '<div class="callout"><strong>Honest system status:</strong> Audio is trained and '
        'evaluated. RF v2, the learned fusion layer, and an evidence-sum rule are available as '
        'research experiments. Fusion pairs are not co-recorded, so no field-validated combined '
        'decision is claimed.</div>',
        unsafe_allow_html=True,
    )
    a, r, f = st.columns(3)
    with a:
        panel("01 · OPERATIONAL", "Audio intelligence",
              "Trained tier 1–4 audio models, noise evaluation, uncertainty flags, open-set checks "
              "and Grad-CAM explanations.")
    with r:
        panel("02 · EXPERIMENTAL", "RF research",
              "RF v2, second-receiver checks, and simulated-emitter experiments; real field "
              "Wi-Fi/Bluetooth false alarms remain unknown.", "pending")
    with f:
        panel("03 · RESEARCH PROTOTYPE", "Learned multimodal fusion",
              "Learned logistic fusion and four-state output are implemented; synthetic pairs "
              "are an upper-bound experiment, not paired-flight validation.", "pending")

    st.subheader("End-to-end project pipeline")
    st.markdown(
        """
        <div class="stage-flow">
          <div class="stage"><small>Stage 01 · audio</small><strong>Acoustic evidence</strong>
            <span>1-second windows → log-mel / frozen PANNs → score, uncertainty state and Grad-CAM.
            Noise robustness is evaluated separately.</span></div>
          <div class="stage"><small>Stage 02 · RF</small><strong>Radio evidence</strong>
            <span>RF v2 uses transmission-aware training and a second receiver. Real Wi-Fi /
            Bluetooth field negatives are still missing.</span></div>
          <div class="stage"><small>Stage 03 · fusion</small><strong>Final multimodal state · research</strong>
            <span>Audio + RF evidence → learned fusion layer → No Drone / Known Drone /
            Unknown Drone / Low Confidence. Current evaluation uses unpaired synthetic
            sensor pairs.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if manifest is not None:
        total = len(manifest)
        usable = int((manifest.exclude_reason == "").sum())
        drone = int((manifest.label == "drone").sum())
        background = int((manifest.label == "background").sum())
    else:
        total = usable = drone = background = 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audio manifest rows", f"{total:,}" if total else "Unavailable")
    c2.metric("Usable rows", f"{usable:,}" if total else "Unavailable")
    c3.metric("Drone-labelled rows", f"{drone:,}" if total else "Unavailable")
    c4.metric("Background rows", f"{background:,}" if total else "Unavailable")

    render_fusion_highlights(reports.get("fusion"))
    render_audio_results(reports["baseline"], reports["pretrained"], reports["outdoor"])
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
         "Per-window spectrogram evidence in the live feed"),
        ("RF baseline vs deep model", "Implemented",
         "Forest / logistic regression / 1D-CNN; v2 fixes v1's noise shortcut"),
        ("Per-drone-model classification and RF open set", "Implemented with limits",
         "Which-drone confusion matrices; RF unknown flag works for 2 of 3 held-out drones"),
        ("Cross-dataset evaluation", "Implemented",
         "Audio: outdoor Svanström; RF: Tampere (second receiver, unseen drones)"),
        ("Probability calibration / ECE", "Implemented (RF)",
         "Temperature scaling on DroneRF, rf_trust.md"),
        ("Live feed and operator dashboard", "Implemented",
         "Dataset/upload audio replay with optional RF feed and fused state"),
        ("Learned audio/RF fusion", "Implemented (upper bound)",
         "Fusion tab; pairs are synthetic because no co-recorded corpus exists"),
    ], columns=["Proposal item", "Current status", "Where / limitation"])
    st.dataframe(coverage, hide_index=True, width="stretch")
    render_report_downloads([
        "baseline_audio.json", "pretrained_audio.json", "outdoor_audio.json",
        "noise_robustness.json", "uncertainty_audio.json", "audio_baseline.json",
        "audio_cnn.json", "audio_noise_robustness.json", "audio_noise_robustness.csv",
        "rf_detection.json",
        "training_curves.json",
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
    render_md_report("deep_audio.md", "CNN vs CNN+GRU vs transformer (from scratch, tiers 1-4)")
    render_md_report("identify.md", "Which-drone confusion matrices (audio and RF)")
    render_md_report("noise_robustness.md", "Noise robustness: full condition × severity grid")
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
    st.subheader("RF-only experiments · DroneRF")
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
    st.markdown("#### Simulated RF replay · recorded PSD windows")
    feature_dir = DATA_DIR / "rf_features"
    required_cache = (
        feature_dir / "X.npy",
        feature_dir / "index.parquet",
        feature_dir / "config.json",
    )
    if not all(path.is_file() for path in required_cache):
        st.info(
            "Replay scores are unavailable because the prepared DroneRF feature cache "
            "is missing. After placing and verifying the 23 source archives, run "
            "`python scripts/prepare_rf.py`; then run `python scripts/eval_rf_trust.py` "
            "to produce calibrated research scores. This panel never connects to an SDR."
        )
    else:
        rf_features, rf_index, rf_artifact = rf_replay_resources()
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
            precision = precision_from_rates(
                result["prevalence"], result["recall"], result["fpr"])
            f1 = (
                2 * precision * result["recall"] / (precision + result["recall"])
                if precision + result["recall"] else 0.0
            )
            accuracy = (
                result["prevalence"] * result["recall"]
                + (1 - result["prevalence"]) * (1 - result["fpr"])
            )
            rows.append({"Model": model, "Evaluation": label,
                         "PR-AUC": result["pr_auc"],
                         "PR-AUC 95% CI": result.get("pr_auc_ci95"),
                         "ROC-AUC": result["roc_auc"],
                         "Accuracy": accuracy, "Precision": precision,
                         "Recall": result["recall"], "F1": f1,
                         "FPR": result["fpr"], "Threshold": result.get("threshold"),
                         "False alarms": result.get("false_alarms"),
                         "Negative windows": result.get("negatives")})
    table = pd.DataFrame(rows)
    table["PR-AUC 95% CI"] = table["PR-AUC 95% CI"].map(
        lambda ci: f"{ci[0]:.3f}–{ci[1]:.3f}" if isinstance(ci, list) else "not reported")
    st.markdown("#### Model comparisons")
    st.dataframe(table.style.format({
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Accuracy": "{:.1%}",
        "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}",
        "FPR": "{:.2%}", "Threshold": "{:.3f}",
    }), hide_index=True, width="stretch")
    diagnostic_rows = []
    for tier, label in names.items():
        result = report["models"]["forest+level"][tier]
        diagnostic_rows.append({
            "Evaluation": label, "PR-AUC": result["pr_auc"], "ROC-AUC": result["roc_auc"],
            "Accuracy": (
                result["prevalence"] * result["recall"]
                + (1 - result["prevalence"]) * (1 - result["fpr"])
            ),
            "Recall": result["recall"], "FPR": result["fpr"],
            "Threshold": result.get("threshold"),
        })
    with st.expander("Shortcut diagnostic · forest with absolute signal level (not a candidate model)"):
        st.dataframe(pd.DataFrame(diagnostic_rows).style.format({
            "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Accuracy": "{:.1%}",
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
    st.markdown("#### RF detector versions")
    st.caption("v1 (above) called pure receiver noise 'drone' on a second receiver. v2 fixes the "
               "training data and is the deployed RF detector used by the live feed and fusion. "
               "v3 is an experiment with simulated Wi-Fi / Bluetooth / microwave negatives.")
    render_md_report("rf_tampere.md", "v1 on a new receiver (Tampere): the noise check it fails")
    render_md_report("rf_v2.md", "v2: deployed RF detector", expanded=True)
    render_md_report("rf_v3.md", "v3 experiment: Wi-Fi / Bluetooth / microwave, 1D-CNN, smoothing, timing features")
    render_md_report("rf_uncertainty.md", "RF uncertainty and unknown-drone flags")
    render_md_report("rf_trust.md", "RF probability calibration and held-out Phantom")
    render_rf_asset_inventory()
    downloads = ["rf_detection.json", "rf_v2.json", "rf_v3.json", "rf_tampere.json"]
    if dronedetect_report is not None:
        downloads.append("dronedetect_subset.json")
    if trust_report is not None:
        downloads.append("rf_trust.json")
    render_report_downloads(downloads, "rf")


def render_fusion_tab(fusion_report: dict | None):
    st.subheader("Audio + RF fusion · ablation")
    if fusion_report is None:
        st.info("Run `python scripts/eval_fusion.py` to generate the fusion ablation.")
        return
    st.markdown(
        '<div class="callout"><strong>Learned fusion layer and ablation.</strong> '
        'Each sensor score is calibrated to evidence; a learned logistic-regression layer '
        'combines audio and RF for the learned result. The evidence-sum fusion rule is also '
        'reported. Pairs are independently sampled, not synchronized recordings, so results '
        'are an upper-bound research experiment.</div>',
        unsafe_allow_html=True,
    )
    render_fusion_highlights(fusion_report)
    scenario = st.selectbox(
        "Compare methods in scenario",
        list(fusion_report["scenarios"]),
        index=0,
        key="fusion_ablation_scenario",
    )
    st.markdown(f"#### Full ablation · {scenario}")
    table = fusion_ablation_rows(fusion_report, scenario)
    st.dataframe(
        table.style.apply(_highlight_learned_fusion, axis=1).format({
            "PR-AUC": "{:.3f}",
            "Recall": "{:.1%}",
            "False-positive rate": "{:.2%}",
            "Accuracy": "{:.1%}",
            "F1": "{:.3f}",
        }, na_rep="n/a"),
        hide_index=True,
        width="stretch",
    )
    if scenario == "RF-silent drone (autonomous)":
        learned = fusion_report["scenarios"][scenario].get("learned", {})
        audio = fusion_report["scenarios"][scenario].get("audio only", {})
        st.warning(
            "Important ablation finding: on the RF-silent drone scenario, the learned "
            f"layer recall is {learned.get('recall', float('nan')):.1%}, versus "
            f"{audio.get('recall', float('nan')):.1%} for audio only. The learned layer "
            "must be treated as experimental, not as a validated deployment decision."
        )
    st.caption(
        "Thresholds are selected on validation pairs. The four outputs are No Drone, "
        "Known Drone, Unknown Drone and Low Confidence."
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


def recording_choice(manifest: pd.DataFrame | None):
    st.sidebar.header("Audio input")
    src = st.sidebar.radio(
        "Source",
        ["Svanström outdoor", "DroneAudioDataset", "ITU ARIS Lab", "Upload audio"],
                           label_visibility="collapsed")
    if src == "Upload audio":
        up = st.sidebar.file_uploader("WAV / FLAC / OGG", type=["wav", "flac", "ogg"])
        if up is None:
            return None, None
        return f"upload:{up.name}:{up.size}", up.getvalue()
    if manifest is None:
        st.sidebar.error("Audio manifest unavailable. Run scripts/build_manifest.py.")
        return None, None

    source_name = {
        "Svanström outdoor": "svanstrom",
        "DroneAudioDataset": "droneaudio",
        "ITU ARIS Lab": "itu_aris",
    }[src]
    rows = manifest[manifest["source"] == source_name].copy()
    if rows.empty:
        st.sidebar.warning(f"{src} is not present in the audio manifest.")
        return None, None

    if source_name == "svanstrom":
        kind = st.sidebar.selectbox("Category", ["DRONE", "HELICOPTER", "BACKGROUND"])
        if kind == "DRONE":
            rows = rows[rows["label"] == "drone"]
        elif kind == "HELICOPTER":
            rows = rows[rows["bg_category"] == "helicopter"]
        else:
            rows = rows[rows["bg_category"] == "outdoor_background"]
    else:
        label = st.sidebar.selectbox("Label", ["drone", "background"])
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
            category = st.sidebar.selectbox(
                "Drone model" if label == "drone" else "Background category",
                categories,
                format_func=lambda value: category_labels[value],
            )
            rows = rows[rows[category_column] == category]
        elif label == "background":
            categories = sorted(rows["bg_category"].fillna("").astype(str).unique())
            category = st.sidebar.selectbox("Background category", categories)
            rows = rows[rows["bg_category"].fillna("").astype(str) == category]

    recordings = rows.drop_duplicates("path").sort_values("path")
    st.sidebar.caption(f"{len(recordings):,} source recordings/clips in this selection.")
    relative_paths = recordings["path"].astype(str).tolist()
    selected = st.sidebar.selectbox(
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
        st.sidebar.error(f"Audio file listed in the manifest is missing: {path}")
        return None, None
    return str(path), path.read_bytes()


def render_live_monitor(manifest: pd.DataFrame | None):
    st.subheader("Simulated live feed · learned audio + RF decision")
    st.caption("One-second audio windows, replayed sequentially. Pick an RF source in the sidebar "
               "to replay held-out RF windows beside the audio. The selected fusion rule produces "
               "No Drone / Known Drone / Unknown Drone / Low Confidence. Audio and RF were never "
               "recorded together, so this is a research what-if, not paired-flight validation.")
    key, audio = recording_choice(manifest)
    feed, rf_src, fusion_method = rf_choice()
    st.sidebar.header("Replay settings")
    hop = st.sidebar.select_slider("Window hop (s)", [0.25, 0.5, 1.0], value=0.5)
    min_hits = st.sidebar.number_input("Consecutive hits to alarm", 1, 5, 2)
    layer_name = st.sidebar.radio("Grad-CAM resolution", list(X.LAYERS))
    speed = st.sidebar.select_slider("Replay speed", [1.0, 2.0, 4.0, 8.0], value=2.0,
                                     format_func=lambda s: f"{s:g}x")
    if audio is None:
        st.info("Choose an outdoor recording or upload a WAV/FLAC/OGG file in the sidebar.")
        return

    det = detector()
    df, clips, sig = analyse(key, audio, hop)
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
        df["shown_state"], df["shown_reason"] = df.fused_state, df.fused_reason
        hits = df.fused_hit
    else:
        df["shown_state"], df["shown_reason"] = df.state, df.reason
        hits = df.hit
    df["shown_label"] = df.shown_state.map(FINAL_STATE_LABEL).fillna("Low Confidence")
    if use_rf:
        df["fused_state_label"] = df.fused_state.map(FINAL_STATE_LABEL).fillna(
            "Low Confidence"
        )
    run = L.longest_run(hits)
    alarm = run >= min_hits
    st.markdown(
        f'<div class="callout">Active audio model: <strong>{det.name}</strong> · '
        f'threshold {det.threshold:.3f}. '
        + (
            f'Final-state rule: <strong>{fusion_method}</strong> fusion. '
            'Unpaired RF replay is experimental; human review remains required.'
            if use_rf else
            'RF unavailable; showing the audio-only state. Human review remains required.'
        )
        + '</div>',
        unsafe_allow_html=True,
    )
    live_slot = st.empty()
    if st.button("▶ Replay as live feed", type="primary"):
        for i in range(len(df)):
            r = df.iloc[i]
            with live_slot.container():
                c1, c2 = st.columns([1, 3])
                c1.metric("time", f"{r.t:.1f} s")
                c1.markdown(
                    f"### {STATE_ICON.get(r.shown_state, '🟡')} {r.shown_label}"
                )
                c1.caption(r.shown_reason or r.tags)
                c2.altair_chart(timeline(df, det.threshold, upto=i), width="stretch")
            time.sleep(hop / speed)
        live_slot.empty()

    verdict = "🚨 Drone candidate" if alarm else "✅ No alarm"
    title = Path(key).stem if not key.startswith("upload") else key.split(":")[1]
    st.subheader(f"{verdict} · {title}")
    m = st.columns(5)
    m[0].metric("Windows", len(df))
    m[1].metric("Above threshold" + (" (fused)" if use_rf else ""), int(hits.sum()))
    m[2].metric("Longest run", f"{run} (need {min_hits})")
    m[3].metric("Low confidence", int((df.shown_state == "low confidence").sum()))
    m[4].metric("Max score", f"{df.p.max():.3f}")
    if use_rf:
        state_counts = df.shown_label.value_counts()
        st.markdown("#### Final multimodal state by window")
        state_cards = st.columns(4)
        for column, state in zip(
            state_cards,
            ("No Drone", "Known Drone", "Unknown Drone", "Low Confidence"),
        ):
            column.metric(state, int(state_counts.get(state, 0)))
    st.audio(to_wav(sig), format="audio/wav")

    i = st.slider("Inspect window", 0, len(df) - 1, int(df.p.idxmax()),
                  format="%d", help="Defaults to the highest-scoring window")
    st.altair_chart(timeline(df, det.threshold, selected=i), width="stretch")
    if use_rf:
        st.caption(f"Audio score above; fused evidence below (RF: {rf_src}). "
                   "Dashed line = fused threshold, 1% false alarms on validation pairs.")
        st.altair_chart(fused_timeline(df, fthr), width="stretch")
    r = df.iloc[i]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"#### {STATE_ICON.get(r.shown_state, '🟡')} {r.shown_label} · "
                    f"{r.t:.1f}–{r.t + 1:.1f} s")
        if r.shown_reason:
            st.caption(f"Flagged: {r.shown_reason}")
        if use_rf:
            st.dataframe(pd.DataFrame({
                "evidence (LLR, capped)": ["audio", "RF", "fused"],
                "value": [f"{r.llr_audio:+.2f}", f"{r.llr_rf:+.2f}", f"{r.fused:+.2f}"]}),
                hide_index=True, width="stretch")
            st.caption(f"Audio-only state: {r.state}. RF score {r.rf_p:.3f}.")
        st.metric("Drone score", f"{r.p:.3f}", f"{r.p - det.threshold:+.3f} vs threshold",
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
            st.caption("RF snapshot replayed with this window (25 ms, both halves of the "
                       "2.4 GHz band, level removed).")
    with st.expander("All windows"):
        st.dataframe(df.drop(columns=["hit", "shown_state", "shown_reason",
                                      *[c for c in L.AUDIO_FLAGS if c in df]]).round(3),
                     hide_index=True, width="stretch")


def main():
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
    required_reports = (
        "baseline", "pretrained", "outdoor", "noise", "uncertainty", "rf",
        "audio_baseline", "audio_cnn", "audio_noise", "training_curves",
    )
    if any(reports[name] is None for name in required_reports):
        return
    manifest_path = DATA_DIR / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None

    hero()
    tabs = st.tabs(["Mission overview", "Audio intelligence", "RF research",
                    "Fusion", "Live feed", "Proposal checklist"])
    with tabs[0]:
        render_overview(reports, manifest)
    with tabs[1]:
        render_audio_tab(reports)
    with tabs[2]:
        render_rf_tab(
            reports["rf"],
            reports["dronedetect_subset"],
            reports["rf_trust"],
        )
    with tabs[3]:
        render_fusion_tab(reports["fusion"])
    with tabs[4]:
        render_live_monitor(manifest)
    with tabs[5]:
        render_methods_tab()


main()
