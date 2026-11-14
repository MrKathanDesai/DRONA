#!/usr/bin/env python3
"""Noise stress test and augmentation ablation (tier 1 setting).

    python scripts/eval_noise.py            # ~5-10 min, mostly encoder passes
    python scripts/eval_noise.py --report-only   # rebuild .md/.png from .json

Every tier-1 test clip (held-out indoor recordings) is mixed with noise from
six conditions at three severities (SNR 20 / 10 / 0 dB) and re-scored. The
threshold stays where clean validation negatives put it (1% FPR), so the
table shows what happens to a deployed detector when the soundscape changes.

Three detectors, all trained on tier-1 train only:
  pretrained          the frozen encoder + logistic head, clean training
  pretrained + aug    same head, trained on clean clips plus one noisy copy
                      of each (condition uniform over the six, SNR uniform
                      0-20 dB). Fixed before looking at any result.
  baseline logreg     level-invariant log-mel statistics, clean training

Noise for the test comes only from tier-1 test negatives; noise for
augmentation only from tier-1 train negatives. A clip is never mixed with
noise from its own recording. See dronedet/features/noise.py for what
"voice" and "music" mean here.

Writes reports/noise_robustness.{json,md} and
reports/figures/09_noise_robustness.png.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dronedet.eval import metrics as E
from dronedet.eval import protocol as P
from dronedet.features import audio as A
from dronedet.features import noise as N
from dronedet.models import baseline as B
from dronedet.models import pretrained as PT
from dronedet.paths import DATA_DIR, resolve

TIER = "split_tier1"
TARGET_FPR = 0.01
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures" / "09_noise_robustness.png"
SEED = 0
N_BOOT = 300

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HEAD_COLOR = {"pretrained": BLUE, "pretrained + aug": AQUA, "baseline logreg": ORANGE}


make_head = PT.head


def _load(rel_path: str, start_s: float | None) -> np.ndarray:
    return A.load_clip(resolve(rel_path), start_s)


def load_clips(df: pd.DataFrame) -> list[np.ndarray]:
    with ProcessPoolExecutor() as pool:
        return list(pool.map(_load, df.path, df.start_s, chunksize=64))


def _melstats(y: np.ndarray) -> np.ndarray:
    return A.log_mel(y)


def baseline_feats(clips: list[np.ndarray]) -> np.ndarray:
    with ProcessPoolExecutor() as pool:
        X = np.stack(list(pool.map(_melstats, clips, chunksize=64)))
    return B.summary_features(X)


NOISE_ROOT = DATA_DIR / "raw_noise"


def bank_for(df: pd.DataFrame, clips: list[np.ndarray], split: str) -> N.NoiseBank:
    """ESC-50 negatives of this split, plus the same split's half of the
    external speech and music corpora."""
    neg = (df.label == "background").values
    c = [c for c, n in zip(clips, neg) if n]
    cats, groups = df.category[neg].tolist(), df.group_id[neg].tolist()
    for kind in N.EXTERNAL:
        ec, ek, eg = N.load_external(NOISE_ROOT, kind, split)
        c, cats, groups = c + ec, cats + ek, groups + eg
    return N.NoiseBank(c, cats, groups)


def corrupt_all(clips, groups, bank, cond, snr, rng):
    return [bank.corrupt(c, cond, snr, rng, avoid_group=g)
            for c, g in zip(clips, groups)]


def fmt_row(r):
    lo, hi = r["pr_auc_ci95"]
    return (f"{r['pr_auc']:.3f} [{lo:.3f}, {hi:.3f}] | {r['roc_auc']:.3f} | "
            f"{r['recall']:.3f} | {r['fpr']:.3f} ({r['false_alarms']}/{r['negatives']})")


def plot(results: dict) -> None:
    conds = list(N.CONDITIONS)
    sev = ["clean", *N.SEVERITY_SNR_DB]
    fig, axes = plt.subplots(2, len(conds), figsize=(2.4 * len(conds), 5.4), sharey="row",
                             facecolor=SURFACE)
    for j, cond in enumerate(conds):
        for head, color in HEAD_COLOR.items():
            rec = [results[head]["clean"]["recall"]] + \
                  [results[head][f"{cond}/{s}"]["recall"] for s in N.SEVERITY_SNR_DB]
            fpr = [results[head]["clean"]["fpr"]] + \
                  [results[head][f"{cond}/{s}"]["fpr"] for s in N.SEVERITY_SNR_DB]
            axes[0, j].plot(sev, rec, "o-", color=color, lw=2, ms=4, label=head)
            axes[1, j].plot(sev, fpr, "o-", color=color, lw=2, ms=4)
        axes[0, j].set_title(cond, color=INK, fontweight="bold", fontsize=11)
        axes[1, j].axhline(TARGET_FPR, color=INK2, lw=1, ls="--")
        for ax in axes[:, j]:
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=8)
    axes[0, 0].set_ylabel("drone recall", color=INK2)
    axes[1, 0].set_ylabel("false-alarm rate", color=INK2)
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Tier 1 under noise: threshold fixed on clean validation "
                 "(dashed = 1% target)", color=INK, fontweight="bold")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)

def findings(report: dict) -> list[str]:
    """Plain-language summary computed from the results, so it stays true
    if the script is re-run. Written after the first run was seen."""
    R = report["results"]
    pt, aug, base = R["pretrained"], R["pretrained + aug"], R["baseline logreg"]
    conds = [c for c in N.CONDITIONS]
    hi = {c: f"{c}/high" for c in conds}
    gain = {c: aug[hi[c]]["recall"] - pt[hi[c]]["recall"] for c in conds}
    worst_fpr = max(conds, key=lambda c: pt[hi[c]]["fpr"])
    base_wins = [c for c in conds if base[hi[c]]["recall"] > aug[hi[c]]["recall"]]
    low_ok = max(abs(pt[f"{c}/low"]["recall"] - pt["clean"]["recall"]) for c in conds)
    out = [
        "## What this shows", "",
        f"- **Mild noise (20 dB) barely matters.** Pretrained recall moves by at "
        f"most {low_ok:.3f} from clean ({pt['clean']['recall']:.3f}).",
        f"- **Loud noise (0 dB) cuts recall sharply** for the clean-trained "
        f"head, to {min(pt[hi[c]]['recall'] for c in conds):.2f}-"
        f"{max(pt[hi[c]]['recall'] for c in conds):.2f} across conditions.",
        f"- **Noise augmentation at 0 dB** changes recall by "
        f"{min(gain.values()):+.2f} to {max(gain.values()):+.2f} across conditions "
        f"(speech {gain.get('speech', float('nan')):+.2f}, real music "
        f"{gain.get('music_real', float('nan')):+.2f}, neither seen in training), with "
        f"clean recall unchanged ({pt['clean']['recall']:.3f} -> "
        f"{aug['clean']['recall']:.3f}). Its clean false-alarm rate is "
        f"{aug['clean']['fpr']:.1%} vs {pt['clean']['fpr']:.1%}, still within the 1% target.",
        f"- **The noise itself causes false alarms** mainly for {worst_fpr} at 0 dB "
        f"({pt[hi[worst_fpr]]['fpr']:.1%} clean-trained, "
        f"{aug[hi[worst_fpr]]['fpr']:.1%} augmented; target 1%).",
    ]
    if base_wins:
        out.append(
            "- **The classical baseline is more robust in some conditions** at 0 dB "
            f"({', '.join(base_wins)}), and keeps the lowest false-alarm rate "
            "throughout, at the cost of lower clean recall.")
    sizes = report.get("noise_pool_sizes")
    if sizes:
        out.append("- Test noise pools: " + ", ".join(
            f"{c} {n} clips" for c, n in sizes.items())
            + ". The wind pool is small, so wind results rest on few recordings.")
    return out + [""]


def write_md(report: dict) -> None:
    results, thr = report["results"], report["thresholds"]
    heads = list(results)
    conditions = [(k, None, None) for k in results[heads[0]]]
    head = ("| condition | PR-AUC [95% CI] | ROC-AUC | recall | FPR (false alarms) |\n"
            "|---|---|---|---|---|")
    summary = ["| condition | " + " | ".join(f"{h} recall / FPR" for h in heads) + " |",
               "|---|" + "---|" * len(heads)]
    for key, _, _ in conditions:
        summary.append(f"| {key} | " + " | ".join(
            f"{results[h][key]['recall']:.3f} / {results[h][key]['fpr']:.3f}"
            for h in heads) + " |")
    md = [
        "# Noise robustness (tier 1)", "",
        "Held-out indoor test recordings (tier 1), each clip mixed with noise at "
        "SNR 20 dB (low), 10 dB (medium) and 0 dB (high). The threshold is fixed "
        f"at {TARGET_FPR:.0%} FPR on **clean** validation negatives and never "
        "re-tuned, so FPR above 1% means the noise itself triggers alarms. "
        "Generated by `scripts/eval_noise.py`.", "",
        "- Noise sources: ESC-50 recordings from the tier-1 test split only "
        "(augmentation used train-split recordings only).",
        "- **voice** = laughing, coughing, crying baby, sneezing (ESC-50 has no speech).",
        "- **music** is synthetic (random chords + kick drum; ESC-50 has no music).",
        "- **mixed** = voice + traffic + wind + environment + music at equal power.",
        "- **speech** = real read speech (LibriSpeech dev-clean), **music_real** = "
        "real music (GTZAN, 10 genres). Test noise and augmentation noise come "
        "from different speakers / excerpts. Augmentation never used these two, "
        "so they also test whether it carries over to unseen noise types.",
        "- `pretrained + aug` was trained on clean clips plus one noisy copy of "
        "each (condition and SNR 0-20 dB drawn at random). Its settings were "
        "fixed before any test result was seen.", "",
        "## Recall and false-alarm rate at the frozen threshold", "",
        *summary, "",
        "![noise robustness](figures/09_noise_robustness.png)", "",
        *findings(report),
    ]
    for name in heads:
        md += [f"## {name} (threshold {thr[name]:.4f})", "", head,
               *[f"| {k} | " + fmt_row(results[name][k]) + " |" for k, _, _ in conditions], ""]
    md += ["## Caveats", "",
           "- Mixing is additive at a fixed SNR. Real masking also involves "
           "reverberation, distance and microphone response, which are not simulated.",
           "- Indoor tier-1 test only; outdoor performance is already the weak "
           "point (see `outdoor_audio.md`) and is not re-tested here.", ""]
    (REPORTS / "noise_robustness.md").write_text("\n".join(md))
    print("\n".join(md))


def main() -> int:
    if "--report-only" in sys.argv:      # rebuild the .md and figure from the .json
        report = json.loads((REPORTS / "noise_robustness.json").read_text())
        plot(report["results"])
        write_md(report)
        return 0
    rng = np.random.default_rng(SEED)
    emb = np.load(DATA_DIR / "embeddings" / "E.npy")
    df = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))
    idx = pd.read_parquet(DATA_DIR / "features" / "index.parquet")
    if not idx.path.equals(df.path):
        raise RuntimeError("feature and embedding caches disagree; rebuild both")
    feats = B.summary_features(np.load(DATA_DIR / "features" / "X.npy", mmap_mode="r"))
    tr, va, te = ((df[TIER] == s).values for s in ("train", "val", "test"))
    d_tr, d_te = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
    y_va, y_te = df.y[va].values, d_te.y.values

    print(f"  loading {tr.sum()} train + {te.sum()} test clips ...", flush=True)
    c_tr, c_te = load_clips(d_tr), load_clips(d_te)
    bank_tr, bank_te = bank_for(d_tr, c_tr, "train"), bank_for(d_te, c_te, "test")
    for cond, pool in bank_te.by_cond.items():
        if pool:
            print(f"    test noise {cond}: {len(pool)} clips")
    enc = PT.Encoder()

    # --- augmentation: one noisy copy of every train clip ------------------
    print(f"  augmenting train on {enc.device} ...", flush=True)
    conds = list(N.AUG_CONDITIONS)      # fixed before any result was seen
    aug = [bank_tr.corrupt(c, conds[rng.integers(len(conds))], rng.uniform(0, 20),
                           rng, avoid_group=g)
           for c, g in zip(c_tr, d_tr.group_id)]
    e_aug, _ = enc(aug)
    del aug

    heads = {
        "pretrained": make_head().fit(emb[tr], d_tr.y),
        "pretrained + aug": make_head().fit(np.vstack([emb[tr], e_aug]),
                                            np.concatenate([d_tr.y, d_tr.y])),
        "baseline logreg": B.MODELS["logreg"]().fit(feats[tr], d_tr.y),
    }
    thr = {}
    for name, m in heads.items():
        Fva = feats[va] if name.startswith("baseline") else emb[va]
        thr[name] = E.threshold_at_fpr(m.predict_proba(Fva)[:, 1][y_va == 0], TARGET_FPR)

    def score_all(emb_te, feat_te):
        return {n: m.predict_proba(feat_te if n.startswith("baseline") else emb_te)[:, 1]
                for n, m in heads.items()}

    results = {n: {} for n in heads}
    conditions = [("clean", None, None)] + [
        (f"{c}/{s}", c, snr) for c in N.CONDITIONS for s, snr in N.SEVERITY_SNR_DB.items()]
    for key, cond, snr in conditions:
        print(f"  {key} ...", flush=True)
        if cond is None:
            scores = score_all(emb[te], feats[te])
        else:
            noisy = corrupt_all(c_te, d_te.group_id, bank_te, cond, snr, rng)
            scores = score_all(enc(noisy)[0], baseline_feats(noisy))
        for name, s in scores.items():
            results[name][key] = E.summarize(y_te, s, d_te.group_id, thr[name], N_BOOT)

    report = {"tier": TIER, "target_fpr": TARGET_FPR, "seed": SEED,
              "snr_db": N.SEVERITY_SNR_DB,
              "noise_sources": {c: list(v) for c, v in N.CONDITIONS.items()},
              "noise_pool_sizes": {c: len(v) for c, v in bank_te.by_cond.items() if v},
              "thresholds": thr, "results": results}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "noise_robustness.json").write_text(json.dumps(report, indent=2))
    plot(results)

    write_md(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
