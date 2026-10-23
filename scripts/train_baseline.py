#!/usr/bin/env python3
"""Classical baseline: level-invariant log-mel summary stats + sklearn.

    python scripts/train_baseline.py

Thresholds are fixed on validation negatives at a target false-alarm rate
and then applied unchanged to test. Writes reports/baseline_audio.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from dronedet.eval import metrics as E
from dronedet.models import baseline as B
from dronedet.paths import DATA_DIR

FEATS = DATA_DIR / "features"
REPORTS = ROOT / "reports"
TARGET_FPR = 0.01
HARD = ["helicopter", "chainsaw", "vacuum_cleaner", "washing_machine",
        "engine", "airplane"]


def load():
    X = np.load(FEATS / "X.npy")
    df = pd.read_parquet(FEATS / "index.parquet")
    synth = df.group_id.str.replace(r"^synth_", "", regex=True) \
                       .str.replace(r"_?\d+$", "", regex=True)
    df["category"] = np.where(df.label == "drone", "drone",
                              df.bg_category.fillna("synth:" + synth))
    df["y"] = (df.label == "drone").astype(int)
    return X, df


def scores(model, F):
    return model.predict_proba(F)[:, 1]


def tier1(F, df, name):
    tr, va, te = (df.split_tier1 == s for s in ("train", "val", "test"))
    model = B.MODELS[name]().fit(F[tr], df.y[tr])
    s_va, s_te = scores(model, F[va]), scores(model, F[te])
    thr = E.threshold_at_fpr(s_va[df.y[va].values == 0], TARGET_FPR)
    res = E.summarize(df.y[te], s_te, df.group_id[te], thr)
    res["val_recall"] = E.operating_point(df.y[va].values, s_va, thr)["recall"]
    fa = E.false_alarms_by_category(df[te], s_te, thr)
    return res, fa


def tier2(F, df, name):
    """Train on Bebop, test on Mambo. There is no val split, so the threshold
    comes from out-of-fold scores on train, grouped by recording."""
    tr, te = df.split_tier2 == "train", df.split_tier2 == "test"
    oof = cross_val_predict(B.MODELS[name](), F[tr], df.y[tr],
                            groups=df.group_id[tr], cv=GroupKFold(5),
                            method="predict_proba")[:, 1]
    thr = E.threshold_at_fpr(oof[df.y[tr].values == 0], TARGET_FPR)
    model = B.MODELS[name]().fit(F[tr], df.y[tr])
    return E.summarize(df.y[te], scores(model, F[te]), df.group_id[te], thr)


def shortcut_probes(X, df):
    """How well does a nuisance variable alone predict the label?

    Reported as separability = max(AUC, 1 - AUC): 0.5 means useless, 1.0
    means the variable alone gives the label away. A tree is used so a
    non-monotonic relation (e.g. several class-specific durations) counts.
    These variables are kept *out* of the features; the probe shows what a
    careless pipeline could have learned instead of the drone."""
    tr, te = df.split_tier1 == "train", df.split_tier1 == "test"
    out = {}
    for name, v in [("clip_level_db", B.clip_level(X)),
                    ("duration_s", df.duration.values)]:
        v = v.reshape(-1, 1)
        m = HistGradientBoostingClassifier(max_depth=3, random_state=0)
        s = m.fit(v[tr], df.y[tr]).predict_proba(v[te])[:, 1]
        auc = float(roc_auc_score(df.y[te], s))
        out[name] = round(max(auc, 1 - auc), 3)
    return out


def md_table(t: pd.DataFrame) -> str:
    lines = ["| category | FA rate | false alarms | clips |", "|---|---|---|---|"]
    for cat, r in t.iterrows():
        lines.append(f"| {cat} | {r.fa_rate:.3f} | {int(r.false_alarms)} | {int(r.clips)} |")
    return "\n".join(lines)


def fmt(r):
    lo, hi = r["pr_auc_ci95"]
    return (f"| {r['pr_auc']:.3f} [{lo:.3f}, {hi:.3f}] | {r['roc_auc']:.3f} "
            f"| {r['recall']:.3f} | {r['fpr']:.4f} ({r['false_alarms']}/{r['negatives']}) |")


def main() -> int:
    X, df = load()
    F = B.summary_features(X)
    print(f"  features {F.shape}")

    report = {"target_fpr": TARGET_FPR, "feature_dim": F.shape[1],
              "shortcut_probes_roc_auc": shortcut_probes(X, df), "models": {}}
    fa_tables = {}
    for name in B.MODELS:
        print(f"  {name}: tier 1 ...", flush=True)
        t1, fa = tier1(F, df, name)
        print(f"  {name}: tier 2 ...", flush=True)
        t2 = tier2(F, df, name)
        report["models"][name] = {"tier1_test": t1, "tier2_test": t2}
        fa_tables[name] = fa

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "baseline_audio.json").write_text(json.dumps(report, indent=2))

    head = ("| model | split | PR-AUC [95% CI] | ROC-AUC | recall @ val-1%-FPR "
            "| test FPR (false alarms) |\n|---|---|---|---|---|---|")
    rows = []
    for name, r in report["models"].items():
        rows.append(f"| {name} | tier 1 " + fmt(r["tier1_test"]))
        rows.append(f"| {name} | tier 2 " + fmt(r["tier2_test"]))
    best = max(report["models"], key=lambda n: report["models"][n]["tier1_test"]["pr_auc"])
    fa = fa_tables[best]
    hard = fa.loc[[c for c in HARD if c in fa.index]]
    md = [
        "# Audio baseline", "",
        "Level-invariant log-mel summary statistics (per-band shape, spread, "
        "flux; 384 dims) with sklearn classifiers. The threshold is set on "
        f"validation negatives at {TARGET_FPR:.0%} FPR and applied unchanged to test. "
        "CIs resample whole recordings. Generated by `scripts/train_baseline.py`.", "",
        head, *rows, "",
        f"Test prevalence: tier 1 {report['models'][best]['tier1_test']['prevalence']:.3f}, "
        f"tier 2 {report['models'][best]['tier2_test']['prevalence']:.3f} "
        "(PR-AUC of a random classifier equals the prevalence).", "",
        "## Shortcut probes (separability of a nuisance variable alone, tier 1 test; 0.5 = useless, 1.0 = gives the label away)", "",
        *[f"- `{k}`: {v}" for k, v in report["shortcut_probes_roc_auc"].items()], "",
        f"## Tier 1 false alarms by category ({best})", "",
        "Hard confusers:", "",
        md_table(hard), "",
        "Worst 10 overall:", "",
        md_table(fa.head(10)), "",
    ]
    (REPORTS / "baseline_audio.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
