#!/usr/bin/env python3
"""Uncertainty flags and open-set ("unknown drone") detection for audio.

    python scripts/eval_uncertainty.py       # ~1 min, embeddings only

Settings fixed before any test result was seen:
  - review band: the 10% of validation windows nearest the threshold
  - unfamiliar: farther from training than 95% of validation windows
  - unknown drone: scored drone, but farther from training drones than 95%
    of validation drone windows
  - ensemble: 20 heads on bootstrap resamples of whole recordings

Part 1 (tiers 1, 3, 4): does flagging concentrate the errors? For each tier
the detector is built from that tier's train/val only and applied to test.
Part 2 (open set): train on Bebop recordings, then check whether the unseen
Mambo drone is flagged as unfamiliar more often than held-out Bebop.

Writes reports/uncertainty_audio.{json,md},
reports/figures/10_risk_coverage.png and the deployable tier-4 detector
data/models/detector.joblib (used by scripts/detect.py and the dashboard).
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from dronedet.eval import metrics as E
from dronedet.eval import protocol as P
from dronedet.models import pretrained as PT
from dronedet.models import uncertainty as U
from dronedet.paths import DATA_DIR


TARGET_FPR = 0.01
REVIEW_RATE = 0.10
FAMILIAR_Q = 0.95
N_ENSEMBLE = 20
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures" / "10_risk_coverage.png"
OUT = DATA_DIR / "models" / "detector.joblib"

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def head_balanced():
    return PT.head(balanced=True)


def head_weighted():          # tier 4: weights come from source_weights instead
    return PT.head(balanced=False)


def build(F, df, tr, va, make_head, weights=None) -> U.Detector:
    y, g = df.y.values, df.group_id.values
    kw = {} if weights is None else {"logisticregression__sample_weight": weights}
    head = make_head().fit(F[tr], y[tr], **kw)
    p_va = head.predict_proba(F[va])[:, 1]
    thr = E.threshold_at_fpr(p_va[y[va] == 0], TARGET_FPR)
    margin = U.margin_for_review_rate(U.logit(p_va) - U.logit(thr), REVIEW_RATE)
    fam = U.Familiarity().fit(F[tr]).calibrate(F[va], FAMILIAR_Q)
    fam_d = U.Familiarity().fit(F[tr & (y == 1)]).calibrate(F[va & (y == 1)], FAMILIAR_Q)
    ens = U.HeadEnsemble(make_head, N_ENSEMBLE).fit(F[tr], y[tr], g[tr], weights)
    return U.Detector(head, thr, margin, fam, fam_d, ens,
                      info={"target_fpr": TARGET_FPR, "review_rate": REVIEW_RATE,
                            "familiar_quantile": FAMILIAR_Q, "encoder": PT.CHECKPOINT.name})


def evaluate(det: U.Detector, F, df, te) -> tuple[dict, pd.DataFrame, dict]:
    a = det.assess(F[te])
    t = df[te].reset_index(drop=True)
    y = t.y.values
    hit = a["p"] >= det.threshold
    err = hit != (y == 1)
    flagged = a["state"] == "low confidence"
    curves = {
        "margin": -np.abs(a["margin"]),
        "ensemble spread": a["spread"],
        "distance": a["distance"],
        "random": np.random.default_rng(0).random(len(y)),
    }
    res = {
        "windows": int(len(y)),
        "error_rate": float(err.mean()),
        "flagged_rate": float(flagged.mean()),
        "error_rate_unflagged": float(err[~flagged].mean()),
        "error_rate_flagged": float(err[flagged].mean()) if flagged.any() else None,
        "errors_caught": float(err[flagged].sum() / max(err.sum(), 1)),
        "unfamiliar_rate": float(a["unfamiliar"].mean()),
        "risk_coverage": {k: U.risk_coverage(err, u) for k, u in curves.items()},
    }
    t = t.assign(state=a["state"], err=err)
    t["group"] = np.where(t.label == "drone", "drone",
                          np.where(t.source == "svanstrom", t.category, "background"))
    states = pd.crosstab(t.group, t.state).reindex(columns=list(U.STATES), fill_value=0)
    return res, states, {k: (err, u) for k, u in curves.items()}


def open_set(F, df) -> dict:
    """Train on Bebop, hold out Bebop recordings, test on Mambo."""
    tr2 = (df.split_tier2 == "train").values
    idx = np.flatnonzero(tr2)
    g = df.group_id.values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    fit_i, rest = next(gss.split(idx, groups=g[idx]))
    rest = idx[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
    cal_i, known_i = next(gss2.split(rest, groups=g[rest]))
    fit, cal, known = (np.isin(np.arange(len(df)), s)
                       for s in (idx[fit_i], rest[cal_i], rest[known_i]))
    det = build(F, df, fit, cal, head_balanced)
    y = df.y.values == 1
    mambo = (df.split_tier2 == "test").values & y
    bebop = known & y
    d_b = det.familiar_drone.distance(F[bebop])
    d_m = det.familiar_drone.distance(F[mambo])
    out = {"bebop_heldout_windows": int(bebop.sum()), "mambo_windows": int(mambo.sum()),
           "auroc_mambo_vs_bebop": float(roc_auc_score(
               np.r_[np.zeros(len(d_b)), np.ones(len(d_m))], np.r_[d_b, d_m])),
           "flag_rate_bebop": float((d_b > det.familiar_drone.cutoff).mean()),
           "flag_rate_mambo": float((d_m > det.familiar_drone.cutoff).mean())}
    for name, m in (("bebop", bebop), ("mambo", mambo)):
        s = pd.Series(det.assess(F[m])["state"]).value_counts(normalize=True)
        out[f"states_{name}"] = {k: float(s.get(k, 0.0)) for k in U.STATES}
    return out


def plot(curves: dict) -> None:
    color = {"margin": BLUE, "ensemble spread": AQUA, "distance": ORANGE, "random": MUTED}
    fig, axes = plt.subplots(1, len(curves), figsize=(13, 3.8), facecolor=SURFACE)
    for ax, (tier, cs) in zip(axes, curves.items()):
        for name, (err, u) in cs.items():
            e = err[np.argsort(u, kind="stable")].astype(float)
            cov = np.arange(1, len(e) + 1) / len(e)
            ax.plot(cov, np.cumsum(e) / np.arange(1, len(e) + 1), color=color[name],
                    lw=2 if name != "random" else 1.2,
                    ls="-" if name != "random" else "--", label=name)
        ax.set_title(tier, color=INK, fontweight="bold", fontsize=11)
        ax.set_xlabel("share of windows kept (most certain first)", color=INK2)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("error rate on kept windows", color=INK2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Risk-coverage: a useful uncertainty measure stays below 'random'",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)


def main() -> int:
    F = np.load(DATA_DIR / "embeddings" / "E.npy")
    df = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))

    report = {"target_fpr": TARGET_FPR, "review_rate": REVIEW_RATE,
              "familiar_quantile": FAMILIAR_Q, "n_ensemble": N_ENSEMBLE, "tiers": {}}
    states_md, curves, deploy = [], {}, None
    setups = {
        "tier 1 (indoor held-out)": ("split_tier1", head_balanced, False),
        "tier 3 (indoor -> outdoor)": ("split_tier3", head_balanced, False),
        "tier 4 (+ outdoor training)": ("split_tier4", head_weighted, True),
    }
    for name, (tier, make, weighted) in setups.items():
        print(f"  {name} ...", flush=True)
        tr, va, te = ((df[tier] == s).values for s in ("train", "val", "test"))
        w = P.source_weights(df[tr]) if weighted else None
        det = build(F, df, tr, va, make, w)
        res, states, curves[name] = evaluate(det, F, df, te)
        res["threshold"], res["margin_logit"] = det.threshold, det.margin
        res["states"] = states.to_dict("index")
        report["tiers"][name] = res
        states_md += [f"**{name}**", "", "| | " + " | ".join(U.STATES) + " |",
                      "|---|" + "---|" * len(U.STATES),
                      *[f"| {g} | " + " | ".join(str(int(r[s])) for s in U.STATES) + " |"
                        for g, r in states.iterrows()], ""]
        if tier == "split_tier4":
            det.info["trained_on"] = "tier 4: indoor + ITU ARIS, source-balanced"
            deploy = det

    print("  open set ...", flush=True)
    report["open_set"] = os_ = open_set(F, df)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "uncertainty_audio.json").write_text(json.dumps(report, indent=2))
    plot(curves)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy, OUT)

    rows = []
    for name, r in report["tiers"].items():
        rc = r["risk_coverage"]
        rows.append(
            f"| {name} | {r['error_rate']:.3f} | {r['flagged_rate']:.3f} | "
            f"{r['error_rate_unflagged']:.3f} | {r['errors_caught']:.3f} | "
            f"{r['unfamiliar_rate']:.3f} | "
            + " | ".join(f"{rc[k]['aurc']:.3f}" for k in
                         ("margin", "ensemble spread", "distance", "random")) + " |")
    md = [
        "# Uncertainty and open-set flags (audio)", "",
        "Each window gets one of four states: **drone**, **no drone**, **unknown "
        "drone** (scored drone but unlike every training drone) or **low "
        "confidence** (score near the threshold, or input unlike anything in "
        "training). All cut-offs come from validation data and were fixed "
        f"before testing: review band = {REVIEW_RATE:.0%} of validation windows "
        f"nearest the threshold; unfamiliar = beyond the {FAMILIAR_Q:.0%} "
        "quantile of validation distance (Mahalanobis on 64 PCA components). "
        "Generated by `scripts/eval_uncertainty.py`.", "",
        "## Do the flags catch the errors?", "",
        "Errors are counted at the frozen 1%-FPR threshold. AURC = area under "
        "the risk-coverage curve (lower is better; `random` is the no-information "
        "reference).", "",
        "| tier | error rate | flagged | error rate, unflagged | share of errors "
        "flagged | unfamiliar | AURC margin | AURC spread | AURC distance | AURC random |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows, "",
        "![risk coverage](figures/10_risk_coverage.png)", "",
        "## States by category (test windows)", "", *states_md,
        "## Open set: unseen drone model", "",
        "Familiarity fitted on Bebop drone windows from 70% of Bebop/background "
        "recordings, calibrated on 15%, tested on the other 15% (known drone) "
        "and on every Mambo window (unknown drone).", "",
        f"- Distance separates Mambo from held-out Bebop at AUROC "
        f"**{os_['auroc_mambo_vs_bebop']:.3f}**.",
        f"- Flagged as unlike any training drone: Bebop {os_['flag_rate_bebop']:.1%} "
        f"({os_['bebop_heldout_windows']} windows), Mambo {os_['flag_rate_mambo']:.1%} "
        f"({os_['mambo_windows']} windows).", "",
        "| windows | " + " | ".join(U.STATES) + " |", "|---|" + "---|" * len(U.STATES),
        *[f"| {n} | " + " | ".join(f"{os_[f'states_{n}'][s]:.1%}" for s in U.STATES) + " |"
          for n in ("bebop", "mambo")], "",
        "## What this shows", "",
        "- **The margin is the useful signal.** Ranking by distance from the "
        "threshold beats random on every tier. On tier 1, the "
        f"{report['tiers']['tier 1 (indoor held-out)']['flagged_rate']:.0%} of "
        "windows sent to review hold "
        f"{report['tiers']['tier 1 (indoor held-out)']['errors_caught']:.0%} "
        "of the errors.",
        "- **The familiarity flag does not detect the move outdoors.** Only "
        f"{report['tiers']['tier 3 (indoor -> outdoor)']['unfamiliar_rate']:.1%} "
        "of Svanström windows are flagged unfamiliar to the indoor-only model, "
        "and ranking by distance is no better than random. The outdoor shift "
        "apparently lies outside the 64 main directions of the training "
        "embeddings. A residual-based distance might catch it, but choosing it "
        "now would mean tuning on the test set, so it is left for a new test set.",
        "- **Ensemble spread adds nothing.** It is worse than the margin on "
        "every tier and no better than random outdoors, so it is shown in the "
        "dashboard as a number but not used for flags.",
        "- **Open set works in ranking, less in calibration.** Mambo is farther "
        "from known drones than held-out Bebop (AUROC above), but the "
        "flag rate on held-out Bebop is above the nominal 5%, probably "
        "because the calibration split holds few recordings. Mambo rarely scores above "
        "the drone threshold, so most Mambo windows end as low confidence, "
        "not unknown drone.", "",
        "## Caveats", "",
        "- Only two drone models exist in the indoor data, so the open-set test "
        "is a single pair (Bebop known, Mambo unknown).",
        "- The 'what this shows' notes were written after seeing these results; "
        "the cut-offs themselves were not changed.",
        f"- The tier-4 detector is saved to `data/models/{OUT.name}`; its "
        "states are what `detect.py` and the dashboard show.", "",
    ]
    (REPORTS / "uncertainty_audio.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
