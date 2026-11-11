"""The three evaluation tiers, shared by every model so numbers compare.

Each tier fixes its threshold at `target_fpr` on negatives the model did not
train on, then applies it unchanged to test.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from dronedet.eval import metrics as E


def with_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    synth = df.group_id.str.replace(r"^synth_", "", regex=True) \
                       .str.replace(r"_?\d+$", "", regex=True)
    df["category"] = np.where(df.label == "drone", "drone",
                              df.bg_category.fillna("synth:" + synth))
    df["y"] = (df.label == "drone").astype(int)
    return df


def _scores(model, F):
    return model.predict_proba(F)[:, 1]


def _train_val_test(F, df, tier, make_model, target_fpr):
    tr, va, te = (df[tier] == s for s in ("train", "val", "test"))
    model = make_model().fit(F[tr], df.y[tr])
    s_va, s_te = _scores(model, F[va]), _scores(model, F[te])
    thr = E.threshold_at_fpr(s_va[df.y[va].values == 0], target_fpr)
    res = E.summarize(df.y[te], s_te, df.group_id[te], thr)
    res["val_recall"] = E.operating_point(df.y[va].values, s_va, thr)["recall"]
    fa = E.false_alarms_by_category(df[te], s_te, thr)
    return res, fa, model, thr


def tier1(F, df, make_model: Callable, target_fpr: float):
    """Held-out recordings, same corpus."""
    res, fa, _, _ = _train_val_test(F, df, "split_tier1", make_model, target_fpr)
    return res, fa


def tier2(F, df, make_model: Callable, target_fpr: float):
    """Train on Bebop, test on Mambo. There is no val split, so the threshold
    comes from out-of-fold scores on train, grouped by recording."""
    tr, te = df.split_tier2 == "train", df.split_tier2 == "test"
    oof = cross_val_predict(make_model(), F[tr], df.y[tr],
                            groups=df.group_id[tr], cv=GroupKFold(5),
                            method="predict_proba")[:, 1]
    thr = E.threshold_at_fpr(oof[df.y[tr].values == 0], target_fpr)
    model = make_model().fit(F[tr], df.y[tr])
    return E.summarize(df.y[te], _scores(model, F[te]), df.group_id[te], thr)


def tier3(F, df, make_model: Callable, target_fpr: float):
    """Train on the indoor corpus, test on the outdoor one (Svanstrom). The
    threshold comes from indoor validation negatives: the outdoor corpus is
    never seen, not even to tune the operating point."""
    res, fa, _, _ = _train_val_test(F, df, "split_tier3", make_model, target_fpr)
    return res, fa


def fit_deployable(F, df, make_model: Callable, target_fpr: float):
    """The model a demo should use: tier-3 training data (all indoor
    recordings except validation), threshold from validation negatives."""
    _, _, model, thr = _train_val_test(F, df, "split_tier3", make_model, target_fpr)
    return model, thr
