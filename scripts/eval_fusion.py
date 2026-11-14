#!/usr/bin/env python3
"""Audio + RF fusion and the proposal's ablation: audio-only vs RF-only vs fused.

    python scripts/eval_uncertainty.py      # audio detector (tier 4)
    python scripts/eval_rf_uncertainty.py   # RF detector
    python scripts/eval_fusion.py

No corpus records audio and RF of the same flight, so pairs are synthetic: an
audio window and an RF window of the same class, drawn independently. That
pairing matches the independence the sum rule assumes, so fused numbers are
an upper bound on what real co-recorded data would give. Audio test windows
are the outdoor Svanström set (drone / helicopter / outdoor background); the
indoor test set is inside the tier-4 detector's training data. RF windows
come from the v2 RF detector's dataset (scripts/train_rf_v2.py): DroneRF
held-out time, held-out Tampere drones, and receiver noise. v2 was retrained
because v1 called pure noise "drone" (rf_tampere.md).

Calibration, the learned fusion layer and every threshold use validation
pairs only (audio tier-4 val x RF v2 val). Scenarios:

- both sensors: drone pairs = drone audio + drone RF
- RF-silent drone: an autonomous drone, drone audio + background RF
- audio unavailable / RF unavailable: one sensor missing (evidence 0)
- RF on a new receiver: drone RF from held-out Tampere drones only
- RF fooled?: no-drone pairs get receiver-noise RF (no transmitter). v1
  called all of it "drone" (rf_tampere.md); the v2 detector should not

Writes reports/fusion.{json,md}, reports/figures/12_fusion_ablation.png and
data/models/fusion.joblib.
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

from dronedet import fusion as FU
from dronedet.eval import metrics as E
from dronedet.eval import protocol as P
from dronedet.models import rf as M
from dronedet.paths import DATA_DIR
from make_figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, AQUA, REPORTS, save

TARGET_FPR = 0.01
REVIEW_RATE = 0.10
LLR_CAP = 5.0
N_PAIRS = 4000           # per class, per scenario
SEED = 0
# Deployed rule. The plan was the proposal's learned layer; it was switched to
# the sum after this ablation showed the learned layer, fitted on pairs that
# never contain an RF-silent drone, nearly ignores audio when RF is quiet.
# That choice used test results, and the report says so.
DEPLOY = "sum"
METHODS = ("audio only", "RF only", "OR", "AND", "sum", "learned")


class Pool:
    """One modality's windows: score, label, group, category and flags."""

    def __init__(self, a: dict, df: pd.DataFrame):
        self.p = a["p"]
        self.y = df.y.values
        self.group = df.group_id.values
        self.category = df.category.values
        self.flags = {k: a[k] for k in ("low_margin", "unfamiliar", "far_from_drones")}

    def take(self, idx):
        return {"p": self.p[idx], "group": self.group[idx],
                "category": self.category[idx],
                "flags": {k: v[idx] for k, v in self.flags.items()}}


def audio_pools():
    det = joblib.load(DATA_DIR / "models" / "detector.joblib")
    Em = np.load(DATA_DIR / "embeddings" / "E.npy")
    df = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))
    out = {}
    for split in ("val", "test"):
        m = (df.split_tier4 == split).values
        out[split] = Pool(det.assess(Em[m]), df[m].reset_index(drop=True))
    return det, out


def rf_pools():
    """RF windows scored by the deployed RF detector (v2, scripts/train_rf_v2.py):
    validation and test splits of its combined dataset. `test` holds DroneRF
    held-out time and held-out Tampere drones (positives) and DroneRF
    background (negatives); `tampere` the held-out Tampere drones alone;
    `noise` the held-out receiver-noise windows (no drone)."""
    det = joblib.load(DATA_DIR / "models" / "rf_detector.joblib")
    feats = M.MODELS[det.info.get("features", M.DEPLOY)][0]
    v2 = DATA_DIR / "rf_features_v2"
    X = np.load(v2 / "X.npy")
    df = pd.read_parquet(v2 / "index.parquet")
    real = df.source.isin(["dronerf", "tampere"]).values
    noise = df.source.str.startswith("noise").values
    masks = {"val": (df.split == "val").values,
             "test": (df.split == "test").values & real,
             "tampere": (df.split == "test").values & (df.source == "tampere").values,
             "noise": (df.split == "test").values & noise}
    out = {}
    for name, m in masks.items():
        if m.any():
            out[name] = Pool(det.assess(feats(X[m])), df[m].reset_index(drop=True))
    return det, out


def draw(rng, pool: Pool, cls: int, n: int, mask=None) -> np.ndarray:
    ok = pool.y == cls if mask is None else (pool.y == cls) & mask
    return rng.choice(np.flatnonzero(ok), n, replace=True)


def make_pairs(rng, A: Pool, R: Pool, n: int, rf_for_drone: Pool | None = None,
               rf_drone_cls: int = 1, rf_for_neg: Pool | None = None):
    """n drone pairs and n no-drone pairs. `rf_for_drone` / `rf_drone_cls`
    choose where a drone pair's RF comes from (e.g. background RF for an
    RF-silent drone)."""
    Rd, Rn = rf_for_drone or R, rf_for_neg or R
    ia = np.r_[draw(rng, A, 1, n), draw(rng, A, 0, n)]
    a = A.take(ia)
    r_pos, r_neg = Rd.take(draw(rng, Rd, rf_drone_cls, n)), Rn.take(draw(rng, Rn, 0, n))
    r = {"p": np.r_[r_pos["p"], r_neg["p"]],
         "flags": {k: np.r_[r_pos["flags"][k], r_neg["flags"][k]] for k in r_pos["flags"]}}
    return np.r_[np.ones(n, int), np.zeros(n, int)], a, r


def fit_fusion(A_val: Pool, R_val: Pool, rng):
    cal = {"audio": FU.Calibrator(LLR_CAP).fit(A_val.p, A_val.y),
           "rf": FU.Calibrator(LLR_CAP).fit(R_val.p, R_val.y)}
    y, a, r = make_pairs(rng, A_val, R_val, N_PAIRS)
    fus = {}
    for method in ("sum", "learned"):
        f = FU.Fusion(cal, method)
        L = FU.pair_llrs(f.llr({"audio": a["p"], "rf": r["p"]}), len(y))
        if method == "learned":
            f.learned = FU.fit_learned(L, y)
        f.fit_decision({"audio": a["p"], "rf": r["p"]}, y, TARGET_FPR, REVIEW_RATE)
        f.info = {"target_fpr": TARGET_FPR, "review_rate": REVIEW_RATE,
                  "llr_cap": LLR_CAP, "val_pairs_per_class": N_PAIRS}
        fus[method] = f
    return fus


def evaluate(y, a, r, groups, fus, det_a, det_r, have_audio=True, have_rf=True):
    """Every method on one set of pairs."""
    n = len(y)
    pa = a["p"] if have_audio else None
    pr = r["p"] if have_rf else None
    out = {}
    hit_a = None if pa is None else pa >= det_a.threshold
    hit_r = None if pr is None else pr >= det_r.threshold
    llr = fus["sum"].llr({"audio": pa, "rf": pr})
    if pa is not None:
        out["audio only"] = E.summarize(y, llr["audio"], groups,
                                        float(fus["sum"].calibrators["audio"].llr(
                                            np.array([det_a.threshold]))[0]))
    if pr is not None:
        out["RF only"] = E.summarize(y, llr["rf"], groups,
                                     float(fus["sum"].calibrators["rf"].llr(
                                         np.array([det_r.threshold]))[0]))
    if pa is not None and pr is not None:
        for name, h in (("OR", hit_a | hit_r), ("AND", hit_a & hit_r)):
            out[name] = {**E.operating_point(y, h.astype(float), 0.5),
                         "pr_auc": None, "pr_auc_ci95": (None, None), "roc_auc": None}
    for method in ("sum", "learned"):
        f = fus[method]
        sc = {"audio": pa, "rf": pr}
        out[method] = E.summarize(y, f.fused(sc, n), groups, f.threshold(sc))
    return out


def helicopter_fa(y, a, r, fus, det_a, det_r) -> dict:
    """False-alarm rate on helicopter pairs only (audio helicopter + RF background)."""
    heli = (y == 0) & (a["category"] == "helicopter")
    out = {"audio only": float((a["p"][heli] >= det_a.threshold).mean()),
           "RF only": float((r["p"][heli] >= det_r.threshold).mean())}
    out["OR"] = float(((a["p"] >= det_a.threshold) | (r["p"] >= det_r.threshold))[heli].mean())
    out["AND"] = float(((a["p"] >= det_a.threshold) & (r["p"] >= det_r.threshold))[heli].mean())
    for m in ("sum", "learned"):
        sc = {"audio": a["p"], "rf": r["p"]}
        out[m] = float((fus[m].fused(sc, len(y))[heli] >= fus[m].threshold(sc)).mean())
    return out


def plot(results):
    scen = [s for s in results if results[s]]
    colors = {"audio only": BLUE, "RF only": ORANGE, "OR": MUTED, "AND": GRID,
              "sum": AQUA, "learned": INK}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.13
    for k, m in enumerate(METHODS):
        xs, vals = [], []
        for i, s in enumerate(scen):
            if m in results[s]:
                xs.append(i + (k - 2.5) * w)
                vals.append(results[s][m]["recall"])
        ax.bar(xs, vals, w, color=colors[m], label=m, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(scen)), [s.replace(" (", "\n(") for s in scen], fontsize=8.5)
    ax.set_ylabel("recall at frozen threshold", color=INK2)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    ax.set_title("Fusion ablation on synthetic pairs (upper bound; RF caveat applies)",
                 color=INK, fontsize=10, loc="left", pad=28)
    fig.tight_layout()
    save(fig, "12_fusion_ablation.png")


def findings(results, heli, weights) -> list[str]:
    b, s = results["both sensors"], results["RF-silent drone (autonomous)"]
    out = [
        f"- With both sensors, fusion is as good as RF alone (recall "
        f"{b['sum']['recall']:.3f} sum, {b['RF only']['recall']:.3f} RF only). RF v2 "
        "is near-perfect on these test windows, so there is little left for audio to "
        "add here; its real negatives come from one lab, so this is RF's best case.",
        f"- RF-silent drone: the sum keeps audio's recall ({s['sum']['recall']:.3f} vs "
        f"{s['audio only']['recall']:.3f} audio only); the learned layer drops to "
        f"{s['learned']['recall']:.3f}, because it weights RF {weights[1]:.2f} vs audio "
        f"{weights[0]:.2f} and never saw a drone without a radio link.",
        f"- OR catches the most but has the most false alarms ({b['OR']['fpr']:.4f}); "
        f"AND has the fewest ({b['AND']['fpr']:.4f}) but misses whatever one sensor misses.",
        f"- Helicopter false alarms: audio only {heli['audio only']:.3f}, sum "
        f"{heli['sum']:.3f}, AND {heli['AND']:.3f}.",
    ]
    if "RF noise only (receiver noise, no drone)" in results:
        f = results["RF noise only (receiver noise, no drone)"]
        out.append(f"- Receiver noise with no drone: false alarms RF only "
                   f"{f['RF only']['fpr']:.3f}, OR {f['OR']['fpr']:.3f}, sum "
                   f"{f['sum']['fpr']:.3f}, AND {f['AND']['fpr']:.3f}; the sum is above "
                   "the 1% target here (with the v1 RF "
                   "detector every rule but AND reached 1.000). The cap would not have "
                   "saved a fooled RF sensor: the fused threshold sits below zero, so "
                   "capped RF evidence crosses it whenever audio is neutral.")
    if "RF on new receiver (Tampere drones)" in results:
        t = results["RF on new receiver (Tampere drones)"]
        out.append(f"- RF on a new receiver (Tampere drones): RF-only recall "
                   f"{t['RF only']['recall']:.3f}, sum {t['sum']['recall']:.3f}.")
    return out


def fmt(r):
    pr = "n/a" if r.get("pr_auc") is None else (
        f"{r['pr_auc']:.3f} [{r['pr_auc_ci95'][0]:.3f}, {r['pr_auc_ci95'][1]:.3f}]")
    return (f"| {pr} | {r['recall']:.3f} | {r['fpr']:.4f} ({r['false_alarms']}/{r['negatives']}) "
            f"| {r['accuracy']:.3f} | {r['f1']:.3f} |")


def main() -> int:
    rng = np.random.default_rng(SEED)
    det_a, A = audio_pools()
    det_r, R = rf_pools()
    fus = fit_fusion(A["val"], R["val"], rng)

    results, heli, states = {}, {}, {}
    scenarios = {
        "both sensors": dict(),
        "RF-silent drone (autonomous)": dict(rf_drone_cls=0),
        "audio unavailable": dict(have_audio=False),
        "RF unavailable": dict(have_rf=False),
    }
    if "tampere" in R:
        scenarios["RF on new receiver (Tampere drones)"] = dict(rf_for_drone=R["tampere"])
    if "noise" in R:
        scenarios["RF noise only (receiver noise, no drone)"] = dict(rf_for_neg=R["noise"])
    for name, opt in scenarios.items():
        y, a, r = make_pairs(rng, A["test"], R["test"], N_PAIRS,
                             opt.get("rf_for_drone"), opt.get("rf_drone_cls", 1),
                             opt.get("rf_for_neg"))
        results[name] = evaluate(y, a, r, a["group"], fus, det_a, det_r,
                                 opt.get("have_audio", True), opt.get("have_rf", True))
        if name == "both sensors":
            heli = helicopter_fa(y, a, r, fus, det_a, det_r)
            st = fus[DEPLOY].assess({"audio": a["p"], "rf": r["p"]}, len(y),
                                    {"audio": a["flags"], "rf": r["flags"]})
            states = (pd.crosstab(np.where(y == 1, "drone", "no drone"), st["state"],
                                  normalize="index").round(3).to_dict(orient="index"))
        print(f"  {name}: done", flush=True)

    joblib.dump(fus[DEPLOY], DATA_DIR / "models" / "fusion.joblib")
    weights = fus["learned"].learned.coef_.ravel().tolist()
    report = {"target_fpr": TARGET_FPR, "llr_cap": LLR_CAP, "pairs_per_class": N_PAIRS,
              "deploy": DEPLOY, "learned_weights_audio_rf": weights,
              "learned_bias": float(fus["learned"].learned.intercept_[0]),
              "scenarios": results, "helicopter_false_alarm_rate": heli,
              "states_both_sensors": states}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "fusion.json").write_text(json.dumps(report, indent=2, default=float))
    plot(results)

    head = ("| method | PR-AUC [95% CI] | recall | FPR (false alarms) | accuracy | F1 |\n"
            "|---|---|---|---|---|---|")
    md = [
        "# Audio + RF fusion: ablation", "",
        "**Read as an upper bound.** No corpus has audio and RF of the same flight, "
        "so each pair is an audio window and an RF window of the same class drawn "
        "independently, which is exactly the independence the sum rule assumes. "
        "RF is the v2 detector (`rf_v2.md`), retrained after v1 called pure noise "
        "\"drone\" (`rf_tampere.md`); its real negatives still come from one lab, "
        "so RF false alarms on real-world Wi-Fi / Bluetooth are unmeasured. Each "
        f"sensor's evidence is capped at +-{LLR_CAP:.0f} (log-likelihood ratio); the "
        "cap does not stop one confident sensor from deciding alone, because the "
        "fused threshold sits below zero. "
        "Generated by `scripts/eval_fusion.py`.", "",
        "Audio: tier-4 detector on the outdoor Svanström test set (drone, "
        "helicopter, outdoor background). RF: DroneRF held-out time and held-out "
        "Tampere drones (positives), DroneRF background (negatives). Each "
        "sensor's score is calibrated to a log-likelihood ratio on its own "
        "validation windows (balanced Platt scaling). `sum` adds the two; "
        "`learned` is the proposal's fusion layer, a logistic regression on the "
        f"two, fitted on validation pairs (weights audio {weights[0]:.2f}, RF "
        f"{weights[1]:.2f}). `OR` / `AND` combine each sensor's own frozen "
        f"decision. Fused thresholds: {TARGET_FPR:.0%} FPR on validation pairs. "
        f"{N_PAIRS} pairs per class per scenario; CIs resample audio recordings.", "",
    ]
    for name, res in results.items():
        md += [f"## {name}", "", head,
               *[f"| {m} " + fmt(res[m]) for m in METHODS if m in res], ""]
    md += ["## Helicopter false alarms (both sensors present)", "",
           "Audio confuses helicopters with drones; RF sees no drone link.", "",
           "| method | helicopter false-alarm rate |", "|---|---|",
           *[f"| {m} | {v:.3f} |" for m, v in heli.items()], "",
           "## What this shows", "",
           *findings(results, heli, weights), "",
           f"## Four-state output ({DEPLOY} fusion, both sensors)", "",
           "| true class | " + " | ".join(sorted({k for v in states.values() for k in v}))
           + " |", "|---" * (1 + len({k for v in states.values() for k in v})) + "|",
           *[f"| {c} | " + " | ".join(f"{v.get(k, 0):.3f}" for k in
                                      sorted({k for v in states.values() for k in v})) + " |"
             for c, v in states.items()], "",
           f"Share of no-drone pairs sent to review: "
           f"{states.get('no drone', {}).get('low confidence', 0):.1%}; of drone "
           f"pairs: {states.get('drone', {}).get('low confidence', 0):.1%}.", "",
           f"Deployed fusion (`{DEPLOY}`) saved to `data/models/fusion.joblib`. "
           "The plan was the learned layer; it was switched to the sum after "
           "the RF-silent scenario above, so that choice was informed by test "
           "results.", ""]
    (REPORTS / "fusion.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
