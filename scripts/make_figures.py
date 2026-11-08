#!/usr/bin/env python3
"""Figures for presenting the audio module.

    python scripts/make_figures.py

Needs the manifest, embeddings, trained head and both reports (run the rest
of the pipeline first). Writes PNGs to reports/figures/:

  01_spectrograms.png        what drones, helicopters and background look like
  02_duration_shortcut.png   why clips are looped, never zero-padded
  03_results_by_tier.png     baseline vs pretrained on the three tiers
  04_outdoor_scores.png      score distributions on the outdoor corpus
  05_detection_timeline.png  detect.py on three outdoor recordings
  06_embedding_map.png       t-SNE map of the pretrained embeddings
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import detect
from dronedet.features import audio as A
from dronedet.models import pretrained as PT
from dronedet.paths import DATA_DIR, resolve

OUT = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

# Validated categorical slots (dataviz reference palette, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
CLASS_COLOR = {"drone": BLUE, "background": ORANGE, "helicopter": AQUA}
# Sequential single-hue ramp for spectrograms: quiet = light, loud = dark.
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT / name}")


def spec_axes(ax, S, sr, hop, title):
    """Colour range runs from the clip's 2nd percentile to its peak, so each
    panel uses the full ramp whatever its loudness."""
    dur = S.shape[1] * hop / sr
    ax.imshow(S, origin="lower", aspect="auto", cmap=SPEC_CMAP,
              vmin=np.percentile(S, 2), vmax=S.max(), extent=[0, dur, 0, S.shape[0]])
    ticks = [0, 32, 64, 96, 127]
    import librosa
    hz = librosa.mel_frequencies(n_mels=A.DEFAULT.n_mels, fmin=A.DEFAULT.fmin,
                                 fmax=A.DEFAULT.fmax)
    ax.set_yticks(ticks, [f"{hz[t] / 1000:.1f}" for t in ticks])
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)


# --- 01: spectrograms --------------------------------------------------------
def fig_spectrograms(df):
    def pick(mask, k=0):
        return df[mask].iloc[k]
    da, sv = df.source == "droneaudio", df.source == "svanstrom"
    rows = [
        ("Indoor drone (Bebop)", pick(da & (df.drone_model == "bebop") & ~df.path.str.contains("mixed"), 40)),
        ("Indoor drone (Mambo)", pick(da & (df.drone_model == "membo") & ~df.path.str.contains("mixed"), 40)),
        ("Indoor helicopter (ESC-50)", pick(da & (df.bg_category == "helicopter"), 10)),
        ("Outdoor drone", pick(sv & (df.label == "drone"), 45)),
        ("Outdoor helicopter", pick(sv & (df.bg_category == "helicopter"), 45)),
        ("Outdoor background", pick(sv & (df.bg_category == "outdoor_background"), 45)),
    ]
    specs = [A.clip_features(resolve(r.path), start_s=r.start_s) for _, r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, (title, _), S in zip(axes.flat, rows, specs):
        spec_axes(ax, S, A.DEFAULT.sample_rate, A.DEFAULT.hop_length, title)
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (kHz, mel scale)")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Log-mel spectrograms of 1 s clips (darker = louder; colour scaled per clip)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, "Drones show stacked horizontal lines: rotor harmonics at fixed "
             "frequencies. Helicopters are lower and pulsed; background is broadband.",
             color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "01_spectrograms.png")


# --- 02: the duration shortcut ----------------------------------------------
def fig_durations(df):
    d = df[df.source == "droneaudio"]
    t = (d.assign(dur=d.duration.round(4))
           .groupby(["dur", "label"]).size().reset_index(name="n")
           .sort_values("n", ascending=False).head(8)
           .sort_values("n"))
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(t))
    ax.barh(y, t.n, color=[CLASS_COLOR[l] for l in t.label], height=0.7)
    ax.set_yticks(y, [f"{v:.4f} s" for v in t.dur])
    for yi, n, l in zip(y, t.n, t.label):
        ax.text(n + 60, yi, f"{n:,}  {l}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("clips")
    ax.set_xlim(0, t.n.max() * 1.25)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("Every clip duration belongs to exactly one class", loc="left")
    fig.text(0.01, -0.06, "Duration alone separates drone from background perfectly "
             "(separability 1.0). Zero-padding short clips would leak the label as a silent "
             "tail, so clips are looped instead.", color=INK2, fontsize=9, wrap=True)
    save(fig, "02_duration_shortcut.png")


# --- 03: results by tier -----------------------------------------------------
def fig_results():
    base = json.loads((REPORTS / "baseline_audio.json").read_text())["models"]["hgb"]
    pre = json.loads((REPORTS / "pretrained_audio.json").read_text())["models"]["logreg"]
    tiers = [("tier1_test", "Tier 1\nheld-out indoor\nrecordings"),
             ("tier2_test", "Tier 2\nunseen drone\nmodel (Mambo)"),
             ("tier3_test", "Tier 3\noutdoor corpus\n(new place, mic)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(len(tiers))
    w = 0.36
    for ax, key, title in [(axes[0], "recall", "Recall at 1% false-alarm rate (share of drones caught)"),
                           (axes[1], "pr_auc", "PR-AUC with 95% CI (ranking quality)")]:
        for off, res, color, name in [(-w / 2 - 0.01, base, ORANGE, "classical baseline (HGB)"),
                                      (w / 2 + 0.01, pre, BLUE, "pretrained CNN + head")]:
            vals = [res[t][key] for t, _ in tiers]
            bars = ax.bar(x + off, vals, w, color=color, label=name)
            if key == "pr_auc":
                lo = [res[t]["pr_auc_ci95"][0] for t, _ in tiers]
                hi = [res[t]["pr_auc_ci95"][1] for t, _ in tiers]
                ax.errorbar(x + off, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)],
                            fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{v:.2f}", ha="center",
                        va="bottom", color="white", fontsize=9, fontweight="bold")
        if key == "pr_auc":
            prev = [pre[t]["prevalence"] for t, _ in tiers]
            ax.scatter(x, prev, marker="_", s=900, color=INK, zorder=3,
                       label="PR-AUC of random guessing")
        ax.set_xticks(x, [lbl for _, lbl in tiers])
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_title(title, loc="left")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    rand = l1.index("PR-AUC of random guessing")
    fig.legend(h0 + [h1[rand]], l0 + [l1[rand]], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "03_results_by_tier.png")


# --- shared: deployed head on the outdoor corpus -----------------------------
def outdoor_scores():
    head = joblib.load(DATA_DIR / "models" / "pretrained_head.joblib")
    E = np.load(DATA_DIR / "embeddings" / "E.npy")
    idx = pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet")
    sv = (idx.source == "svanstrom").values
    t = idx[sv].assign(score=head["model"].predict_proba(E[sv])[:, 1])
    t["cls"] = np.where(t.label == "drone", "drone",
                        np.where(t.bg_category == "helicopter", "helicopter", "background"))
    return t, head


# --- 04: outdoor score distributions ----------------------------------------
def fig_outdoor(t, head):
    thr = head["threshold"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, cls, name in zip(axes, ["drone", "helicopter", "background"],
                             ["Outdoor drone", "Outdoor helicopter", "Outdoor background"]):
        s = t[t.cls == cls].score
        ax.hist(s, bins=bins, color=CLASS_COLOR[cls], rwidth=0.85)
        ax.axvline(thr, color=INK, lw=1.2, ls="--")
        above = (s >= thr).mean()
        ax.set_title(f"{name}: {above:.0%} of 1 s windows above threshold",
                     loc="left", fontweight="normal", fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[-1].text(thr - 0.01, axes[-1].get_ylim()[1] * 0.5,
                  f"alarm threshold {thr:.2f}\n(set on indoor data at 1% FPR)",
                  color=INK2, fontsize=8, ha="right")
    axes[-1].set_xlabel("drone probability from the pretrained detector")
    fig.suptitle("Outdoor corpus (never seen in training): where each 1 s window scores",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_outdoor_scores.png")


# --- 05: detection timeline --------------------------------------------------
def fig_timeline(t, head):
    enc = PT.Encoder()
    files = ["DRONE_005", "HELICOPTER_005", "BACKGROUND_005"]
    names = ["Outdoor drone", "Outdoor helicopter", "Outdoor background"]
    colors = [BLUE, AQUA, ORANGE]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for k, (f, name, c) in enumerate(zip(files, names, colors)):
        path = DATA_DIR / "raw_svanstrom" / "Audio" / f"{f}.wav"
        y = A.load_mono(path, A.DEFAULT.sample_rate)
        S = A.log_mel(y)
        spec_axes(axes[0, k], S, A.DEFAULT.sample_rate,
                  A.DEFAULT.hop_length, f"{name}  ({f}.wav)")
        clips, starts = detect.windows(y, A.DEFAULT.sample_rate, 0.5)
        emb, _ = enc(clips)
        p = head["model"].predict_proba(emb)[:, 1]
        centres = np.array(starts) + 0.5
        ax = axes[1, k]
        ax.axhline(head["threshold"], color=INK, lw=1, ls="--")
        ax.plot(centres, p, color=c, lw=2)
        ax.scatter(centres, p, color=c, s=18, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        hit = p >= head["threshold"]
        verdict = "DRONE DETECTED" if detect.longest_run(hit) >= 2 else "no drone"
        ax.set_title(f"→ {verdict}", loc="left", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("time (s)")
    axes[0, 0].set_ylabel("frequency (kHz)")
    axes[1, 0].set_ylabel("drone probability")
    axes[1, 2].text(9.8, head["threshold"] - 0.09, f"alarm threshold {head['threshold']:.2f}",
                    color=INK2, fontsize=8, ha="right")
    fig.suptitle("detect.py on outdoor recordings: spectrogram (top) and per-window "
                 "drone probability (bottom, 1 s windows every 0.5 s)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_detection_timeline.png")


# --- 06: embedding map ---------------------------------------------------------
def fig_embedding():
    E = np.load(DATA_DIR / "embeddings" / "E.npy")
    idx = pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet")
    cls = np.where(idx.label == "drone", "drone",
                   np.where(idx.bg_category == "helicopter", "helicopter", "background"))
    rng = np.random.default_rng(0)
    keep = np.zeros(len(idx), bool)
    for c in ("drone", "helicopter", "background"):
        for src in ("droneaudio", "svanstrom"):
            m = np.flatnonzero((cls == c) & (idx.source == src).values)
            keep[rng.choice(m, min(len(m), 600), replace=False)] = True
    X = (E[keep] - E.mean(0)) / (E.std(0) + 1e-6)
    Z = np.zeros((len(idx), 2))
    Z[keep] = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(
        PCA(50, random_state=0).fit_transform(X))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for src, marker, label in [("droneaudio", "o", "indoor"), ("svanstrom", "^", "outdoor")]:
        for c in ("background", "helicopter", "drone"):
            m = keep & (cls == c) & (idx.source == src).values
            ax.scatter(Z[m, 0], Z[m, 1], s=14 if marker == "o" else 22, marker=marker,
                       color=CLASS_COLOR[c], alpha=0.55 if marker == "o" else 0.9,
                       edgecolor=SURFACE, linewidth=0.4, label=f"{label} {c}")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8,
              markerscale=1.4)
    ax.set_title("What the frozen encoder 'hears': t-SNE map of 1024-d embeddings "
                 "(≤600 clips per group)", loc="left")
    fig.text(0.01, -0.07, "Circles = indoor training corpus, triangles = outdoor test "
             "corpus. Nearby points sound alike to the encoder; distances between "
             "far-apart clusters are not meaningful in t-SNE.\nOutdoor drones cluster "
             "mainly with each other, and sit closer to indoor drones than to background: "
             "related to what the head learned, but not the same, hence partial tier-3 recall.",
             color=INK2, fontsize=9)
    save(fig, "06_embedding_map.png")


def main() -> int:
    df = pd.read_parquet(DATA_DIR / "manifest.parquet")
    df = df[df.exclude_reason == ""].reset_index(drop=True)
    fig_spectrograms(df)
    fig_durations(df)
    fig_results()
    t, head = outdoor_scores()
    fig_outdoor(t, head)
    fig_timeline(t, head)
    fig_embedding()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
