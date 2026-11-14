"""Counter-drone monitor: Streamlit dashboard (audio + optional RF fusion).

    streamlit run app/dashboard.py

Replays a recording as if it were a live microphone feed and shows, per 1 s
window, the detector's state (drone / no drone / unknown drone / low
confidence), its score against the frozen threshold, what the pretrained
encoder hears, and a Grad-CAM map of which part of the spectrogram pushed
the score towards "drone".

An RF feed can be replayed beside it: held-out RF windows of a chosen class
(DroneRF, Tampere drones never trained on, or receiver noise), one 25 ms
snapshot per audio window, scored by the RF v2 detector and fused with
data/models/fusion.joblib. Audio and RF were never recorded together, so this
is a what-if; the panel says what RF has and has not been tested on.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from dronedet import live as L
from dronedet.models import explain as X
from dronedet.paths import DATA_DIR

SVANSTROM = DATA_DIR / "raw_svanstrom" / "Audio"
REPORTS = ROOT / "reports"
STATE_COLOR = {"drone": "#d6452a", "unknown drone": "#7a4fd6",
               "low confidence": "#d99a0b", "no drone": "#8a8984"}
STATE_ICON = {"drone": "🔴", "unknown drone": "🟣", "low confidence": "🟡", "no drone": "⚪"}

st.set_page_config(page_title="Counter-drone monitor", page_icon="🛰️",
                   layout="wide")
NO_RF = "None (audio only)"


@st.cache_resource(show_spinner="Loading encoder and detector ...")
def detector() -> L.LiveDetector:
    return L.LiveDetector("detector")


@st.cache_resource(show_spinner="Loading RF detector and fusion ...")
def rf_feed():
    try:
        return L.RFFeed()
    except FileNotFoundError as e:
        return str(e)


@st.cache_data(show_spinner="Scoring windows ...", max_entries=32)
def analyse(key: str, audio: bytes, hop: float):
    y, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, y, sr)
        sig = L.load(f.name)
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


def fused_timeline(df: pd.DataFrame, thr: float):
    base = alt.Chart(df).encode(x=alt.X("t:Q", title="time (s)"))
    bars = base.mark_bar(size=8).encode(
        y=alt.Y("fused:Q", title="fused evidence (LLR)"),
        color=alt.Color("fused_state:N", title="fused state",
                        scale=alt.Scale(domain=list(STATE_COLOR),
                                        range=list(STATE_COLOR.values()))),
        tooltip=["t", alt.Tooltip("llr_audio:Q", format=".2f"),
                 alt.Tooltip("llr_rf:Q", format=".2f"),
                 alt.Tooltip("fused:Q", format=".2f"), "fused_state", "fused_reason"])
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(
        strokeDash=[4, 4], color="#52514e").encode(y="thr:Q")
    return alt.layer(bars, rule).properties(height=200)


def rf_figure(psd: np.ndarray):
    """One RF window: power spectrum of both DroneRF bands (2.400-2.480 GHz)."""
    n = psd.shape[1]
    f = np.linspace(0, 20, n)                     # MHz from each band's centre
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


def recording_choice():
    st.sidebar.header("Input")
    src = st.sidebar.radio("Source", ["Svanström outdoor recordings", "Upload audio"],
                           label_visibility="collapsed")
    if src == "Upload audio":
        up = st.sidebar.file_uploader("WAV / FLAC / OGG", type=["wav", "flac", "ogg"])
        if up is None:
            return None, None
        return f"upload:{up.name}:{up.size}", up.getvalue()
    if not SVANSTROM.exists():
        st.sidebar.warning("Svanström audio not found; run scripts/download_data.py "
                           "or upload a file.")
        return None, None
    kind = st.sidebar.selectbox("Category", ["DRONE", "HELICOPTER", "BACKGROUND"])
    files = sorted(SVANSTROM.glob(f"{kind}_*.wav"))
    f = st.sidebar.selectbox("Recording", files, format_func=lambda p: p.stem)
    return str(f), f.read_bytes()


def rf_choice():
    st.sidebar.header("RF feed")
    feed = rf_feed()
    if isinstance(feed, str):
        st.sidebar.caption(f"RF unavailable: {feed}")
        return None, NO_RF
    src = st.sidebar.selectbox("Replay RF windows from", [NO_RF, *feed.sources],
                               help="Held-out windows of this class, one 25 ms "
                                    "snapshot per audio window. A what-if: audio and "
                                    "RF were never recorded together.")
    return feed, src


def honesty_panel(det: L.LiveDetector):
    with st.expander("How far to trust this", expanded=False):
        lines = [
            f"- Detector: **{det.name}**. Threshold {det.threshold:.3f}, set for 1% "
            "false alarms on validation windows.",
        ]
        p = REPORTS / "outdoor_audio.json"
        if p.exists():
            r = json.loads(p.read_text())["models"]["pretrained balanced"]
            lines.append(
                f"- On the 90 held-out Svanström recordings it catches "
                f"**{r['recall']:.0%}** of drone windows at {r['fpr']:.1%} false "
                "alarms per window. Most outdoor drones are missed.")
        lines += [
            "- Helicopters are the main confusion: drone-vs-helicopter ROC-AUC is "
            "well below the indoor numbers.",
            "- 'Low confidence' = score within the band holding 10% of validation "
            "windows nearest the threshold, or an input unlike the training data.",
            "- 'Unknown drone' = scored as drone, but far from every training drone.",
            "- Grad-CAM cells are coarse (about 1/6 s by a quarter of the mel range).",
            "- RF is the v2 detector (reports/rf_v2.md), retrained after v1 called "
            "pure noise 'drone'. It catches Tampere drones it never trained on and "
            "ignores receiver noise, but its only real no-drone recordings are from "
            "one lab, so false alarms on real-world Wi-Fi / Bluetooth are unmeasured. "
            "A confident RF score can raise a fused alarm on its own. Audio + RF "
            "pairs are replayed, not co-recorded (reports/fusion.md).",
        ]
        st.markdown("\n".join(lines))


def main():
    det = detector()
    st.title("Counter-drone monitor")
    st.caption("Replays a recording as a live microphone feed. Decision support "
               "only: every response stays with a human operator.")

    key, audio = recording_choice()
    feed, rf_src = rf_choice()
    st.sidebar.header("Settings")
    hop = st.sidebar.select_slider("Window hop (s)", [0.25, 0.5, 1.0], value=0.5)
    min_hits = st.sidebar.number_input("Consecutive hits to alarm", 1, 5, 2)
    layer_name = st.sidebar.radio("Grad-CAM resolution", list(X.LAYERS))
    speed = st.sidebar.select_slider("Replay speed", [1.0, 2.0, 4.0, 8.0], value=2.0,
                                     format_func=lambda s: f"{s:g}x")
    honesty_panel(det)

    if audio is None:
        st.info("Pick a recording or upload a file in the sidebar.")
        return
    df, clips, sig = analyse(key, audio, hop)
    df = df.copy()
    use_rf = feed is not None and rf_src != NO_RF
    psd = None
    if use_rf:
        psd, a_rf = feed.windows(rf_src, len(df))
        fz, fthr = L.fuse(df, feed.fusion, a_rf["p"],
                          {k: a_rf[k] for k in L.AUDIO_FLAGS})
        df = pd.concat([df, fz], axis=1)
        df["rf_p"] = a_rf["p"]
        df["shown_state"], df["shown_reason"] = df.fused_state, df.fused_reason
        hits = df.fused_hit
    else:
        df["shown_state"], df["shown_reason"] = df.state, df.reason
        hits = df.hit
    run = L.longest_run(hits)
    alarm = run >= min_hits

    live_slot = st.empty()
    if st.button("▶ Replay as live feed", type="primary"):
        for i in range(len(df)):
            r = df.iloc[i]
            with live_slot.container():
                c1, c2 = st.columns([1, 3])
                c1.metric("time", f"{r.t:.1f} s")
                c1.markdown(f"### {STATE_ICON[r.shown_state]} {r.shown_state}")
                c1.caption(r.shown_reason or r.tags)
                c2.altair_chart(timeline(df, det.threshold, upto=i),
                                width="stretch")
            time.sleep(hop / speed)
        live_slot.empty()

    verdict = "🚨 DRONE ALARM" if alarm else "✅ No alarm"
    st.subheader(f"{verdict}  ·  {Path(key).stem if not key.startswith('upload') else key.split(':')[1]}")
    m = st.columns(5)
    m[0].metric("Windows", len(df))
    m[1].metric("Above threshold" + (" (fused)" if use_rf else ""), int(hits.sum()))
    m[2].metric("Longest run", f"{run} (need {min_hits})")
    m[3].metric("Low confidence", int((df.shown_state == "low confidence").sum()))
    m[4].metric("Max score", f"{df.p.max():.3f}")
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
        st.markdown(f"#### {STATE_ICON[r.shown_state]} {r.shown_state}  ·  "
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
        M, cam, _ = cam_for(key, i, clips[i], X.LAYERS[layer_name])
        st.pyplot(cam_figure(M, cam), width="stretch")
        st.caption("Bright regions pushed the score towards 'drone'. A window "
                   "scored 'no drone' can still show weak evidence.")
        if use_rf:
            st.pyplot(rf_figure(psd[i]), width="stretch")
            st.caption("RF snapshot replayed with this window (25 ms, both halves "
                       "of the 2.4 GHz band, level removed).")

    with st.expander("All windows"):
        st.dataframe(df.drop(columns=["hit", "shown_state", "shown_reason",
                                      *[c for c in L.AUDIO_FLAGS if c in df]]).round(3),
                     hide_index=True, width="stretch")


main()
