#!/usr/bin/env python3
"""RF detector v2: trained so that "drone" has to mean "a drone is transmitting".

    python scripts/prepare_rf.py          # DroneRF PSD
    python scripts/eval_rf_tampere.py     # Tampere PSD cache
    python scripts/train_rf_v2.py

v1 (scripts/train_rf.py) calls pure noise "drone" (reports/rf_tampere.md).
Three changes, all decided before any v2 result was seen:

1. **Labels.** DroneRF labels every window recorded while a drone was on as
   "drone", including 187 windows at the noise floor in both bands. Those
   teach "silence from the drone sessions = drone". They are left out of
   training and testing as drones, and scored separately as a check.
2. **Negatives from many simulated receivers.** White and coloured receiver
   noise (`data/rf_noise.py`) through both pipelines, DroneRF's 40 MS/s real
   samples and the Tampere 120 MS/s IQ conversion, so neither the noise floor
   nor the conversion can stand in for "drone". Pink noise is never trained
   on and is a held-out test family.
3. **Positives from a second receiver.** Tampere drones, split by drone
   (md5 of the name), so test drones are never seen in training.

Same features (128 level-free sub-bands) and forest as v1. Threshold at 1%
FPR on validation negatives (DroneRF background + noise).

**v2b**, added after v2's results were seen: v2 misses DroneRF's Phantom when
it is held out, because its video link fills the whole 40 MHz band flat, which
level-free shape cannot tell from a raised noise floor. v2b adds one feature,
the level difference between the two bands (`models.rf.tilt_features`), and
gives every noise negative a random band-to-band gain mismatch (+-6 dB) so the
feature cannot stand for "not a lab receiver". The rule for which variant is
deployed is fixed below (`choose`), before v2b was run. Writes
reports/rf_v2.{json,md}, the combined dataset to data/rf_features_v2/ (used by
fusion and the dashboard) and the four-state detector to
data/models/rf_detector.joblib.

What this cannot test: real non-drone emitters (Wi-Fi, Bluetooth) recorded
anywhere but DroneRF's lab. No open dataset with them was reachable.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dronedet.data import rf as R
from dronedet.data import rf_noise as RN
from dronedet.eval import metrics as E
from dronedet.models import rf as M
from dronedet.models import uncertainty as U
from dronedet.paths import DATA_DIR

OUT = DATA_DIR / "rf_features_v2"
DETECTOR = DATA_DIR / "models" / "rf_detector.joblib"
REPORTS = ROOT / "reports"
TARGET_FPR = 0.01
REVIEW_RATE = 0.10
FAMILIAR_Q = 0.95
QUIET_DB = 15.0                 # DroneRF, as in train_rf.py
QUIET_ABOVE_FLOOR_DB = 7.0      # Tampere, as in eval_rf_tampere.py
BAND_MISMATCH_DB = 6.0
VARIANTS = {"v2": "forest", "v2b": "forest+tilt"}
NOISE = {"train": (150, ("white", "coloured"), 11),
         "val": (40, ("white", "coloured"), 12),
         "test": (50, ("white", "coloured", "pink"), 13)}


def tampere_split(group: str) -> str:
    """Deterministic, name-based: about 1/5 test, 1/5 val, 3/5 train."""
    k = int(hashlib.md5(group.encode()).hexdigest(), 16) % 5
    return {0: "test", 1: "val"}.get(k, "train")


def tampere_group(drone: str) -> str:
    """Mambo control / video are one drone."""
    return drone.replace("_control", "").replace("_video", "")


def build() -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(DATA_DIR / "rf_features" / "X.npy")
    d = pd.read_parquet(DATA_DIR / "rf_features" / "index.parquet")
    quiet = ((d.level_db_low < QUIET_DB) & (d.level_db_high < QUIET_DB)).values
    a = pd.DataFrame({
        "source": "dronerf", "y": (d.label == "drone").astype(int).values,
        "drone": d.drone_model.fillna("").values, "group_id": d.group_id.values,
        "split": d.split_tier1.values, "quiet": quiet,
        "category": np.where(d.label == "drone", "drone", "rf_background")})
    a.loc[(a.y == 1) & a.quiet, "split"] = "quiet_drone"      # not a usable label

    tp = DATA_DIR / "rf_features_tampere"
    Xt = np.load(tp / "X.npy")
    t = pd.read_parquet(tp / "index.parquet")
    floor = {b: np.percentile(t[f"level_db_{b}"], 5) for b in ("low", "high")}
    tq = ((t.level_db_low < floor["low"] + QUIET_ABOVE_FLOOR_DB)
          & (t.level_db_high < floor["high"] + QUIET_ABOVE_FLOOR_DB)).values
    g = t.drone.map(tampere_group)
    b = pd.DataFrame({
        "source": "tampere", "y": 1, "drone": g.values,
        "group_id": "tampere_" + g.values, "split": g.map(tampere_split).values,
        "quiet": tq, "category": "drone"})
    b.loc[b.quiet, "split"] = "quiet_drone"

    Xs, parts = [X, Xt], [a, b]
    for split, (n, fams, seed) in NOISE.items():
        Xn, rows = RN.noise_bank(n, fams, RN.PIPELINES, seed)
        Xn = mismatch(Xn, seed)
        r = pd.DataFrame(rows)
        Xs.append(Xn)
        parts.append(pd.DataFrame({
            "source": "noise_" + r.pipeline, "y": 0, "drone": "",
            "group_id": [f"noise_{split}_{f}_{p}_{i}" for f, p, i in
                         zip(r.family, r.pipeline, r["index"])],
            "split": split, "quiet": False, "category": "noise_" + r.family}))
    n_old = DATA_DIR / "rf_features_tampere" / "noise_X.npy"     # rf_tampere.md control
    if n_old.exists():
        Xo = mismatch(np.load(n_old), 99)
        Xs.append(Xo)
        parts.append(pd.DataFrame({
            "source": "noise_tampere", "y": 0, "drone": "",
            "group_id": [f"noise_control_{i}" for i in range(len(Xo))],
            "split": "test", "quiet": False, "category": "noise_control"}))
    return np.concatenate(Xs).astype(np.float32), pd.concat(parts, ignore_index=True)


def mismatch(X: np.ndarray, seed: int) -> np.ndarray:
    """Random per-band gain offsets on noise windows (two receivers, or one
    receiver with a tilted front end, never match exactly)."""
    off = np.random.default_rng(seed + 1000).uniform(
        -BAND_MISMATCH_DB, BAND_MISMATCH_DB, (len(X), 2, 1))
    return (X + off).astype(np.float32)


def choose(res: dict) -> str:
    """Fixed before v2b was run: deploy v2b only if it is no worse on noise
    (highest false-alarm rate over noise families) and better on drones never
    trained on (mean of Tampere leave-one-out and DroneRF unseen-drone recall)."""
    def unseen(r):
        return np.mean([x["recall"] for x in r["tampere_lodo"]]
                       + list(r["dronerf_unseen_drone_recall"].values()))
    a, b = res["v2"], res["v2b"]
    return "v2b" if (b["noise_max_fpr"] <= max(a["noise_max_fpr"], 0.01)
                     and unseen(b) > unseen(a)) else "v2"


def fit(F, df, train_mask):
    return M.forest().fit(F[train_mask], df.y.values[train_mask])


def threshold(model, F, df):
    va = (df.split == "val").values & (df.y == 0).values
    return E.threshold_at_fpr(model.predict_proba(F[va])[:, 1], TARGET_FPR)


def breakdown(df_te, p, thr) -> list[dict]:
    rows = []
    for (src, cat), sub in df_te.assign(p=p).groupby(["source", "category"]):
        hit = sub.p >= thr
        rows.append({"source": src, "category": cat, "windows": int(len(sub)),
                     "called_drone": float(hit.mean()), "median_p": float(sub.p.median())})
    return rows


def lodo_tampere(F, df) -> list[dict]:
    """Leave one Tampere drone out of training (all its windows), test on it."""
    out = []
    usable = df.split.isin(["train", "val", "test"]).values
    for g in sorted(df[df.source == "tampere"].group_id.unique()):
        held = (df.group_id == g).values & usable
        tr = (df.split == "train").values | ((df.source == "tampere").values & usable & ~held)
        tr &= ~held
        model = fit(F, df, tr)
        thr = threshold(model, F, df)
        p = model.predict_proba(F[held])[:, 1]
        out.append({"drone": g.removeprefix("tampere_"), "windows": int(held.sum()),
                    "recall": float((p >= thr).mean()), "median_p": float(np.median(p))})
    return out


def dronerf_unseen(F, df) -> dict:
    """DroneRF tier 2 style: one DroneRF drone never trained on."""
    out = {}
    usable = df.split.isin(["train", "val", "test"]).values
    for d in ("bebop", "ar", "phantom"):
        held = ((df.source == "dronerf") & (df.drone == d)).values & usable
        tr = (df.split == "train").values & ~held & ~((df.source == "dronerf") & (df.drone == d)).values
        model = fit(F, df, tr)
        thr = threshold(model, F, df)
        out[d] = float((model.predict_proba(F[held])[:, 1] >= thr).mean())
    return out


def detector(model, thr, F, df) -> U.Detector:
    y = df.y.values
    tr, va = (df.split == "train").values, (df.split == "val").values
    p_va = model.predict_proba(F[va])[:, 1]
    margin = U.margin_for_review_rate(U.logit(p_va) - U.logit(thr), REVIEW_RATE)
    fam = U.Familiarity().fit(F[tr]).calibrate(F[va], FAMILIAR_Q)
    fam_d = U.Familiarity().fit(F[tr & (y == 1)]).calibrate(F[va & (y == 1)], FAMILIAR_Q)
    return U.Detector(model, thr, margin, fam, fam_d, None,
                      info={"version": 2, "target_fpr": TARGET_FPR,
                            "fingerprint": R.DEFAULT.fingerprint(), "modality": "rf"})


def md_breakdown(rows):
    return ["| source | category | windows | called drone | median score |",
            "|---|---|---|---|---|",
            *[f"| {r['source']} | {r['category']} | {r['windows']} | "
              f"{r['called_drone']:.3f} | {r['median_p']:.3f} |" for r in rows]]


def run(F, df) -> tuple[dict, object, float]:
    tr, te = (df.split == "train").values, (df.split == "test").values
    model = fit(F, df, tr)
    thr = threshold(model, F, df)
    p_te = model.predict_proba(F[te])[:, 1]
    overall = E.summarize(df.y.values[te], p_te, df.group_id.values[te], thr)
    rows = breakdown(df[te], p_te, thr)
    qd = ((df.source == "dronerf") & (df.split == "quiet_drone")).values
    qb = ((df.source == "dronerf") & df.quiet & (df.y == 0) & (df.split == "test")).values
    p_qd, p_qb = model.predict_proba(F[qd])[:, 1], model.predict_proba(F[qb])[:, 1]
    quiet = {"quiet_drone_windows": int(qd.sum()), "quiet_background_test_windows": int(qb.sum()),
             "quiet_drone_called_drone": float((p_qd >= thr).mean()),
             "quiet_background_called_drone": float((p_qb >= thr).mean()),
             "auc_quiet_drone_vs_background": float(roc_auc_score(
                 np.r_[np.ones(len(p_qd)), np.zeros(len(p_qb))], np.r_[p_qd, p_qb]))}
    res = {"threshold": thr, "test_overall": overall, "test_breakdown": rows,
           "quiet_check": quiet, "tampere_lodo": lodo_tampere(F, df),
           "dronerf_unseen_drone_recall": dronerf_unseen(F, df),
           "noise_max_fpr": max(r["called_drone"] for r in rows
                                if r["source"].startswith("noise"))}
    return res, model, thr


def md_variant(name, r) -> list[str]:
    o, q = r["test_overall"], r["quiet_check"]
    lodo_mean = float(np.mean([x["recall"] for x in r["tampere_lodo"]]))
    return [
        f"## {name}: test set (held-out DroneRF time, held-out Tampere drones, fresh noise)", "",
        f"PR-AUC {o['pr_auc']:.3f} [{o['pr_auc_ci95'][0]:.3f}, {o['pr_auc_ci95'][1]:.3f}], "
        f"recall {o['recall']:.3f}, false-alarm rate {o['fpr']:.4f} "
        f"({o['false_alarms']}/{o['negatives']}), threshold {r['threshold']:.3f}.", "",
        *md_breakdown(r["test_breakdown"]), "",
        f"### {name}: drones never trained on", "",
        f"Leave one Tampere drone out (mean recall {lodo_mean:.3f}):", "",
        "| held-out drone | windows | recall | median score |", "|---|---|---|---|",
        *[f"| {x['drone']} | {x['windows']} | {x['recall']:.3f} | {x['median_p']:.3f} |"
          for x in r["tampere_lodo"]], "",
        "One DroneRF drone left out of training: " + ", ".join(
            f"{k} {v:.3f}" for k, v in r["dronerf_unseen_drone_recall"].items()) + " recall.", "",
        f"### {name}: the checks v1 failed", "",
        f"- Pure noise, every family and pipeline, highest false-alarm rate: "
        f"**{r['noise_max_fpr']:.3f}** (v1: 1.000), including held-out pink noise "
        "and the 40 white-noise windows from `rf_tampere.md`.",
        f"- DroneRF windows at the noise floor: drone-session windows called drone "
        f"{q['quiet_drone_called_drone']:.3f} ({q['quiet_drone_windows']}), background "
        f"{q['quiet_background_called_drone']:.3f} ({q['quiet_background_test_windows']}); "
        f"rank AUC between them {q['auc_quiet_drone_vs_background']:.3f} (v1's quiet "
        "probe: 1.000).", ""]


def main() -> int:
    print("  building dataset (noise synthesis ~2 min) ...", flush=True)
    X, df = build()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X.npy", X)
    df.to_parquet(OUT / "index.parquet")

    res, models = {}, {}
    for name, feat in VARIANTS.items():
        print(f"  {name} ({feat}) ...", flush=True)
        F = M.MODELS[feat][0](X)
        res[name], model, thr = run(F, df)
        models[name] = (model, thr, F)
    deploy = choose(res)
    model, thr, F = models[deploy]
    det = detector(model, thr, F, df)
    det.info["features"] = VARIANTS[deploy]
    det.info["variant"] = deploy
    DETECTOR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(det, DETECTOR)

    splits = df[df.source == "tampere"].groupby("drone").split.agg(
        lambda s: s[s != "quiet_drone"].iloc[0] if (s != "quiet_drone").any() else "quiet")
    report = {"target_fpr": TARGET_FPR, "variants": res, "deployed": deploy,
              "deployed_features": VARIANTS[deploy],
              "tampere_split": splits.to_dict(),
              "counts": df.groupby(["source", "split", "y"]).size().reset_index()
                          .rename(columns={0: "n"}).to_dict("records")}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "rf_v2.json").write_text(json.dumps(report, indent=2, default=float))

    md = [
        "# RF detector v2: trained to require a transmission", "",
        "v1 calls pure noise \"drone\" (`rf_tampere.md`). **v2** keeps v1's features "
        "and forest and changes the data, decided before any v2 result was seen: "
        "DroneRF drone windows at the noise floor are no longer labelled drone; "
        "synthetic receiver noise (white, coloured) through both the DroneRF and "
        "the Tampere pipelines is added as "
        "negatives; Tampere drones from a second receiver are added as positives, "
        "split by drone. Pink noise is a held-out family. Threshold at 1% FPR on "
        "validation negatives. Generated by `scripts/train_rf_v2.py`.", "",
        "**v2b** was added after v2's results were seen: v2 misses DroneRF's Phantom "
        "when it is held out, because its video link fills the 40 MHz band flat and "
        "level-free shape cannot tell that from a raised noise floor. v2b adds one "
        "feature, the level difference between the two bands, and every noise "
        "window gets a random band-to-band gain mismatch (+-6 dB) so that feature "
        "cannot mean \"not a lab receiver\". Both variants are trained on this same "
        "data; v2's features remove each band's level, so the mismatch leaves v2 "
        "unchanged from its first run. The deployment rule "
        "(no worse on noise, better on unseen drones) was written before v2b ran. "
        f"**Deployed: {deploy}.**", "",
        "Tampere drones in each split: " + ", ".join(
            f"{k} ({v})" for k, v in report["tampere_split"].items()) + ". Every "
        "Tampere drone is also tested by leaving it out of training.", "",
    ]
    for name in VARIANTS:
        md += md_variant(name, res[name])
    md += [
        "## What this still cannot show", "",
        "- Real non-drone emitters (Wi-Fi, Bluetooth, microwave ovens) from anywhere "
        "but DroneRF's lab. DroneRF background is the only real negative; the "
        "datasets with more (DroneDetect, CardRF, Glüge et al. on Kaggle) need a "
        "login. Until then, a false-alarm rate on real-world 2.4 GHz traffic is "
        "unmeasured.",
        "- Tampere is an anechoic chamber, a far cleaner scene than the field.",
        "- v2b's band-difference feature would also fire on a strong non-drone "
        "transmitter in one half of the band; that is exactly the untested case above.", "",
        f"Detector (four states, `{deploy}`) saved to `data/models/{DETECTOR.name}`; "
        "dataset to `data/rf_features_v2/`.", "",
    ]
    (REPORTS / "rf_v2.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
