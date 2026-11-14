"""The four evaluation tiers, shared by every model so numbers compare.

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


def source_weights(df: pd.DataFrame) -> np.ndarray:
    """Every (source, label) cell gets equal total weight. Tier 4 mixes an
    indoor corpus that is 88% background with ITU ARIS, which is 82% drone;
    unweighted, "sounds outdoors" would become a drone cue."""
    n = df.groupby(["source", "label"]).label.transform("size").values
    w = 1.0 / n
    return w * len(df) / w.sum()


def _fit(model, F, y, weights, weight_param):
    if weights is None:
        return model.fit(F, y)
    return model.fit(F, y, **{weight_param: weights})


def _train_val_test(F, df, tier, make_model, target_fpr, weights=None,
                    weight_param="sample_weight"):
    tr, va, te = (df[tier] == s for s in ("train", "val", "test"))
    w = None if weights is None else weights(df[tr])
    model = _fit(make_model(), F[tr], df.y[tr], w, weight_param)
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


def tier2(F, df, make_model: Callable, target_fpr: float,
          holdout: float | None = None):
    """Train on Bebop, test on Mambo. There is no val split, so the threshold
    comes from out-of-fold scores on train, grouped by recording.

    `holdout` (e.g. 0.2) is the cheaper variant for slow models: hold out that
    share of training recordings, fit once on the rest, take the threshold
    from the held-out negatives and test that same model."""
    tr, te = df.split_tier2 == "train", df.split_tier2 == "test"
    if holdout is not None:
        from sklearn.model_selection import GroupShuffleSplit
        idx = np.flatnonzero(tr.values)
        fit_i, cal_i = next(GroupShuffleSplit(1, test_size=holdout, random_state=0)
                            .split(idx, groups=df.group_id.values[idx]))
        fit_i, cal_i = idx[fit_i], idx[cal_i]
        y = df.y.values
        model = make_model().fit(F[fit_i], y[fit_i])
        s_cal = _scores(model, F[cal_i])
        thr = E.threshold_at_fpr(s_cal[y[cal_i] == 0], target_fpr)
        return E.summarize(df.y[te], _scores(model, F[te.values]), df.group_id[te], thr)
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


def tier4(F, df, make_model: Callable, target_fpr: float, weights=source_weights,
          weight_param: str = "sample_weight"):
    """Tier 3's outdoor test, with the outdoor ITU ARIS corpus added to
    training. `weights(train_df)` gives per-sample weights; `weight_param` is
    the fit() keyword that takes them (`logisticregression__sample_weight`
    for a pipeline). Returns (results, false alarms by category, model, thr)."""
    return _train_val_test(F, df, "split_tier4", make_model, target_fpr,
                           weights, weight_param)


def fit_deployable(F, df, make_model: Callable, target_fpr: float,
                   tier: str = "split_tier3", **weighting):
    """The model a demo should use: that tier's training data, threshold from
    its validation negatives. RF has no tier 3 and passes its own tier."""
    _, _, model, thr = _train_val_test(F, df, tier, make_model, target_fpr, **weighting)
    return model, thr
