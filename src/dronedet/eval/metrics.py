from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:

    s = np.sort(np.asarray(neg_scores))[::-1]
    k = int(np.floor(fpr * len(s)))
    if k >= len(s):
        return -np.inf
    return float(np.nextafter(s[k], np.inf))


def operating_point(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    """Counts at a fixed threshold. Accuracy, precision and F1 are included
    because the proposal reports them, but they depend on the test set's drone
    share (12-33% here, far above real life), so PR-AUC and recall at a fixed
    false-alarm rate remain the headline numbers."""
    y, s = np.asarray(y), np.asarray(s)
    pred = s >= thr
    pos, neg = y == 1, y == 0
    tp, fp = int(pred[pos].sum()), int(pred[neg].sum())
    fn, tn = int(pos.sum()) - tp, int(neg.sum()) - fp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and precision + recall else 0.0)
    return {
        "threshold": thr,
        "recall": float(recall),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "false_alarms": fp,
        "negatives": int(neg.sum()),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "precision": float(precision),
        "f1": float(f1),
        "confusion": [[tn, fp], [fn, tp]],    # rows: true no-drone, drone
    }


def grouped_bootstrap(y, s, groups, metric, n=1000, seed=0) -> tuple[float, float]:

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
    if not vals:                       # every resample held one class only
        return float("nan"), float("nan")
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
    neg = df.label.values == "background"
    t = pd.DataFrame({"category": df.category.values[neg],
                      "alarm": s[neg] >= thr})
    out = t.groupby("category").alarm.agg(["mean", "sum", "size"])
    out.columns = ["fa_rate", "false_alarms", "clips"]
    return out.sort_values(["fa_rate", "clips"], ascending=False)
