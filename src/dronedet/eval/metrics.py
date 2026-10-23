from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate on these negatives is
    at most `fpr`. Choose it on validation negatives, never on test."""
    s = np.sort(np.asarray(neg_scores))[::-1]
    k = int(np.floor(fpr * len(s)))
    if k >= len(s):
        return -np.inf
    return float(np.nextafter(s[k], np.inf))


def operating_point(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    pos, neg = y == 1, y == 0
    return {
        "threshold": thr,
        "recall": float(pred[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
        "false_alarms": int(pred[neg].sum()),
        "negatives": int(neg.sum()),
    }


def grouped_bootstrap(y, s, groups, metric, n=1000, seed=0) -> tuple[float, float]:
    """95% CI resampling whole groups (recordings), not clips: slices of one
    recording are near-duplicates, so clip-level resampling overstates
    confidence."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(y, s, groups, thr: float, n_boot: int = 1000) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "pr_auc_ci95": grouped_bootstrap(y, s, groups, average_precision_score, n_boot),
        "roc_auc": float(roc_auc_score(y, s)),
        "prevalence": float(y.mean()),
        **operating_point(y, s, thr),
    }


def false_alarms_by_category(df: pd.DataFrame, s: np.ndarray, thr: float) -> pd.DataFrame:
    """False-alarm rate for each background category at a fixed threshold."""
    neg = df.label.values == "background"
    t = pd.DataFrame({"category": df.category.values[neg],
                      "alarm": s[neg] >= thr})
    out = t.groupby("category").alarm.agg(["mean", "sum", "size"])
    out.columns = ["fa_rate", "false_alarms", "clips"]
    return out.sort_values(["fa_rate", "clips"], ascending=False)
