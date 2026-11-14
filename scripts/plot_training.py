#!/usr/bin/env python3
"""Training curves, for the models where they exist.

    python scripts/plot_training.py

Only the RF CNN trains in epochs. `TinyCNN.fit` logs nothing, so its loop runs
here again with the same seed, scoring after every epoch on the tier 1 splits;
the final weights are checked against a plain `fit`. The other models get
learning curves instead: skill against the amount of training data, adding
whole recordings at a time. Test sets are not used. Writes
reports/training_curves.json and reports/figures/:

  07_cnn_training_curves.png  loss and PR-AUC per epoch, RF CNN
  08_learning_curves.png      PR-AUC vs training data, audio head and RF forest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from dronedet.eval import protocol as P
from dronedet.models import rf as M
from dronedet.paths import DATA_DIR
from make_figures import BLUE, ORANGE, REPORTS, save
from train_pretrained import HEADS

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)


def cnn_history(X, y, tr, va) -> tuple[dict, M.TinyCNN]:
    """`TinyCNN.fit`, scoring after each epoch. Keep the two loops in step."""
    m = M.TinyCNN()
    Xt, yt = X[tr], y[tr]
    m.classes_ = np.unique(yt)
    m.mean_ = Xt.mean(axis=0, keepdims=True)
    m.std_ = Xt.std(axis=0, keepdims=True) + 1e-6
    torch.manual_seed(m.seed)
    m.net_ = m._build()
    opt = torch.optim.Adam(m.net_.parameters(), lr=m.lr, weight_decay=1e-4)
    weight = torch.tensor((yt == 0).sum() / max((yt == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    x_t, y_t = m._tensor(Xt), torch.as_tensor(yt, dtype=torch.float32)
    x_v, y_v = m._tensor(X[va]), torch.as_tensor(y[va], dtype=torch.float32)
    gen = torch.Generator().manual_seed(m.seed)

    hist = {k: [] for k in ("train_loss", "val_loss", "train_pr_auc", "val_pr_auc")}
    for _ in range(m.epochs):
        m.net_.train()
        for idx in torch.randperm(len(x_t), generator=gen).split(m.batch):
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(m.net_(x_t[idx]).squeeze(1), y_t[idx]).backward()
            opt.step()
        m.net_.eval()
        with torch.no_grad():
            for name, xs, ys in (("train", x_t, y_t), ("val", x_v, y_v)):
                logits = m.net_(xs).squeeze(1)
                hist[f"{name}_loss"].append(float(loss_fn(logits, ys)))
                hist[f"{name}_pr_auc"].append(
                    float(average_precision_score(ys.numpy(), logits.numpy())))
    return hist, m


def learning_curve(F, df, make_model) -> dict:
    """Train on a share of each class's recordings; val is always all of val."""
    tr, va = df[df.split_tier1 == "train"], df[df.split_tier1 == "val"]
    out = {"fraction": list(FRACTIONS), "train_clips": [], "train_pr_auc": [], "val_pr_auc": []}
    for frac in FRACTIONS:
        clips, on_train, on_val = [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            keep = []
            for _, groups in tr.groupby("y").group_id:
                uniq = groups.unique()
                keep += list(rng.choice(uniq, max(2, round(frac * len(uniq))), replace=False))
            sub = tr[tr.group_id.isin(keep)]
            model = make_model().fit(F[sub.index], sub.y)
            clips.append(len(sub))
            on_train.append(average_precision_score(
                sub.y, model.predict_proba(F[sub.index])[:, 1]))
            on_val.append(average_precision_score(
                va.y, model.predict_proba(F[va.index])[:, 1]))
        out["train_clips"].append(int(np.mean(clips)))
        out["train_pr_auc"].append(float(np.mean(on_train)))
        out["val_pr_auc"].append(float(np.mean(on_val)))
        print(f"    {frac:.0%}: val PR-AUC {out['val_pr_auc'][-1]:.3f}", flush=True)
    return out


def line(ax, x, y, color, label, marker=None):
    ax.plot(x, y, color=color, lw=2, marker=marker, ms=8, mec="white", label=label)


def tidy(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="best")


def fig_cnn(h):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = np.arange(1, len(h["train_loss"]) + 1)
    line(a, ep, h["train_loss"], BLUE, "training data")
    line(a, ep, h["val_loss"], ORANGE, "validation data")
    tidy(a, "Loss falls, then stays near zero", "Epoch", "Weighted cross-entropy loss")
    line(b, ep, h["train_pr_auc"], BLUE, "training data")
    line(b, ep, h["val_pr_auc"], ORANGE, "validation data")
    tidy(b, "PR-AUC starts near its ceiling", "Epoch", "PR-AUC")
    fig.tight_layout()
    save(fig, "07_cnn_training_curves.png")


def fig_learning(audio, rf):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    axes[0].set_ylim(0.85, 1.01)
    for ax, c, title in ((axes[0], audio, "Audio: pretrained head"),
                         (axes[1], rf, "RF: random forest")):
        line(ax, c["train_clips"], c["train_pr_auc"], BLUE, "training data", "o")
        line(ax, c["train_clips"], c["val_pr_auc"], ORANGE, "validation data", "o")
        tidy(ax, title, "Training windows used", "PR-AUC")
        ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    fig.tight_layout()
    save(fig, "08_learning_curves.png")


def main() -> int:
    X = np.load(DATA_DIR / "rf_features" / "X.npy")
    rf = pd.read_parquet(DATA_DIR / "rf_features" / "index.parquet")
    rf["y"] = (rf.label == "drone").astype(int)
    S = M.shape_features(X)
    tr, va = (rf.split_tier1 == "train").values, (rf.split_tier1 == "val").values
    y = rf.y.values

    print("  RF CNN, per epoch ...", flush=True)
    hist, logged = cnn_history(S, y, tr, va)
    plain = M.TinyCNN().fit(S[tr], y[tr])
    gap = np.abs(logged.predict_proba(S[va]) - plain.predict_proba(S[va])).max()
    if gap > 1e-4:
        raise RuntimeError(f"logged loop drifted from TinyCNN.fit (gap {gap:.2g})")

    emb = np.load(DATA_DIR / "embeddings" / "E.npy")
    audio = P.with_category(pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet"))
    audio = audio.reset_index(drop=True)
    print("  audio head, learning curve ...", flush=True)
    lc_audio = learning_curve(emb, audio, HEADS["logreg"])
    print("  RF forest, learning curve ...", flush=True)
    F = M.band_features(S)
    lc_rf = learning_curve(F, rf.reset_index(drop=True), M.forest)

    head = HEADS["logreg"]()
    t1 = (audio.split_tier1 == "train").values
    head.fit(emb[t1], audio.y[t1])
    iters = int(head.named_steps["logisticregression"].n_iter_[0])

    out = {"cnn": hist, "cnn_train_windows": int(tr.sum()), "cnn_val_windows": int(va.sum()),
           "audio_head_solver_iterations": iters,
           "audio_head_learning_curve": lc_audio, "rf_forest_learning_curve": lc_rf}
    (REPORTS / "training_curves.json").write_text(json.dumps(out, indent=2))
    fig_cnn(hist)
    fig_learning(lc_audio, lc_rf)

    for e in (1, 2, 3, 5, 10, 20, 40):
        print(f"  epoch {e:2d}: train loss {hist['train_loss'][e-1]:.4f}, "
              f"val loss {hist['val_loss'][e-1]:.4f}, val PR-AUC {hist['val_pr_auc'][e-1]:.4f}")
    print(f"  best val loss epoch {int(np.argmin(hist['val_loss'])) + 1}, "
          f"audio head solver iterations {iters}, logged vs plain gap {gap:.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
