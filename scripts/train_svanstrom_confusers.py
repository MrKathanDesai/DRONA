#!/usr/bin/env python3
"""Compare audio heads with and without held-out outdoor Svanström negatives.

Uses cached embeddings/features. Svanström helicopter and outdoor-background
recordings are divided by recording; all Svanström drones stay test-only.
This experiment does not replace the existing tier-4 detector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from dronedet.data.manifest import split_svanstrom_confusers
from dronedet.eval import metrics as E
from dronedet.eval import protocol as P
from dronedet.models import baseline as B
from dronedet.paths import DATA_DIR
from train_outdoor import make_head, source_weights

REPORT_JSON = ROOT / "reports" / "svanstrom_confusers.json"
REPORT_MD = ROOT / "reports" / "svanstrom_confusers.md"
TARGET_FPR = 0.01
SEED = 0


def _recording_metrics(test: pd.DataFrame, scores: np.ndarray,
                       threshold: float) -> list[dict]:
    rows = test.assign(hit=scores >= threshold)
    results = []
    for category, clips in rows.sort_values("start_s").groupby("category"):
        recordings = []
        for _, recording in clips.groupby("group_id"):
            consecutive = best = 0
            for hit in recording.hit:
                consecutive = consecutive + 1 if hit else 0
                best = max(best, consecutive)
            recordings.append(best >= 2)
        results.append({
            "category": category,
            "recordings": len(recordings),
            "alarms": int(sum(recordings)),
            "alarm_rate": float(np.mean(recordings)),
        })
    return results


def _fit_and_score(features: np.ndarray, df: pd.DataFrame,
                   include_svanstrom: bool) -> dict:
    svan = df.source.eq("svanstrom").to_numpy()
    svan_negative = svan & df.label.eq("background").to_numpy()
    base_train = ~svan & df.split_tier4.eq("train").to_numpy()
    base_val = ~svan & df.split_tier4.eq("val").to_numpy()
    svan_train = svan_negative & df.confuser_split.eq("train").to_numpy()
    svan_val = svan_negative & df.confuser_split.eq("val").to_numpy()
    train = base_train | (svan_train if include_svanstrom else False)
    val = base_val | svan_val
    test = svan & (
        df.label.eq("drone").to_numpy()
        | df.confuser_split.eq("test").to_numpy()
    )

    train_groups = set(df.loc[train, "group_id"])
    val_groups = set(df.loc[val, "group_id"])
    test_groups = set(df.loc[test, "group_id"])
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("recording groups overlap between train, validation, and test")
    if df.loc[train, "y"].nunique() != 2 or df.loc[val, "y"].nunique() != 2:
        raise RuntimeError("training and validation must each contain both classes")

    weights = source_weights(df.loc[train])
    model = make_head().fit(
        features[train],
        df.loc[train, "y"],
        logisticregression__sample_weight=weights,
    )
    val_scores = model.predict_proba(features[val])[:, 1]
    threshold = E.threshold_at_fpr(
        val_scores[df.loc[val, "y"].to_numpy() == 0], TARGET_FPR
    )
    test_df = df.loc[test].copy()
    test_scores = model.predict_proba(features[test])[:, 1]
    metrics = E.summarize(
        test_df.y.to_numpy(),
        test_scores,
        test_df.group_id.to_numpy(),
        threshold,
    )
    metrics["val_recall"] = E.operating_point(
        df.loc[val, "y"].to_numpy(), val_scores, threshold
    )["recall"]
    metrics["false_alarms_by_category"] = (
        E.false_alarms_by_category(test_df, test_scores, threshold)
        .reset_index()
        .to_dict("records")
    )
    metrics["recording_level"] = _recording_metrics(
        test_df, test_scores, threshold
    )
    metrics["test_windows_by_category"] = (
        test_df.groupby("category").size().astype(int).to_dict()
    )
    metrics["test_recording_count"] = int(test_df.group_id.nunique())
    metrics["threshold"] = float(threshold)
    return metrics


def evaluate() -> dict:
    embedding_index = pd.read_parquet(DATA_DIR / "embeddings" / "index.parquet")
    feature_index = pd.read_parquet(DATA_DIR / "features" / "index.parquet")
    if not embedding_index.path.equals(feature_index.path):
        raise RuntimeError("feature and embedding caches disagree; rebuild both")
    df = P.with_category(embedding_index)
    split = split_svanstrom_confusers(df, seed=SEED)
    df["confuser_split"] = df.group_id.map(split).fillna("excluded")
    if df.loc[df.source.eq("svanstrom") & df.label.eq("drone"),
              "confuser_split"].ne("test").any():
        raise RuntimeError("Svanström drone recordings must remain test-only")

    embeddings = np.load(DATA_DIR / "embeddings" / "E.npy", mmap_mode="r")
    raw_features = np.load(DATA_DIR / "features" / "X.npy", mmap_mode="r")
    if len(df) != len(embeddings) or len(df) != len(raw_features):
        raise RuntimeError("cached features do not align with the index")

    runs = {
        "pretrained_without_svanstrom_negatives": embeddings,
        "pretrained_with_svanstrom_negatives": embeddings,
        "baseline_without_svanstrom_negatives": B.summary_features(raw_features),
        "baseline_with_svanstrom_negatives": B.summary_features(raw_features),
    }
    results = {}
    for name, features in runs.items():
        include_svanstrom = name.endswith("with_svanstrom_negatives")
        print(f"  {name} ...", flush=True)
        results[name] = _fit_and_score(features, df, include_svanstrom)

    split_counts = (
        df.loc[df.source.eq("svanstrom") & df.label.eq("background")]
        .groupby(["category", "confuser_split"])
        .group_id.nunique()
        .unstack(fill_value=0)
    )
    return {
        "title": "Svanström outdoor-confuser training experiment",
        "seed": SEED,
        "target_fpr": TARGET_FPR,
        "split": (
            "Svanström background recordings are split 60/20/20 by recording "
            "for train/validation/test; every Svanström drone recording is "
            "test-only. Both model variants use the same validation negatives "
            "for threshold selection and the same test recordings."
        ),
        "confuser_recordings_by_split": split_counts.to_dict(),
        "test_categories": sorted(df.loc[
            df.source.eq("svanstrom") & (
                df.label.eq("drone") | df.confuser_split.eq("test")
            ), "category"
        ].unique()),
        "test_recordings_by_category": (
            df.loc[df.source.eq("svanstrom") & (
                df.label.eq("drone") | df.confuser_split.eq("test")
            )]
            .groupby("category").group_id.nunique().astype(int).to_dict()
        ),
        "aircraft_status": (
            "The integrated audio manifest has no explicitly aircraft-labeled "
            "outdoor recordings. Existing airplane negatives remain in the "
            "training data; outdoor aircraft-specific training is still needed."
        ),
        "limitations": [
            "This is a recording-held-out split within Svanström, not an independent-corpus test.",
            "Svanström was used in earlier project evaluation; this split is not a blind external validation.",
            "Helicopter and outdoor-background negatives are added; the integrated manifest has no explicitly aircraft-labeled outdoor examples.",
            "This experimental run does not replace or retrain the deployed tier-4 detector.",
        ],
        "models": results,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Outdoor confuser training experiment",
        "",
        "This separate experiment adds Svanström outdoor helicopter and ambient "
        "background negatives to training while keeping its drone recordings "
        "test-only. It does not replace the deployed tier-4 detector.",
        "",
        "Thresholds are calibrated to 1% FPR on validation negatives. Both "
        "variants use the same Svanström validation negatives for threshold "
        "selection and the same recording-held-out test recordings. The test "
        "set contains 30 drone recordings and 6 per negative category (71.4% "
        "drone windows), so PR-AUC is not a deployment-prevalence estimate.",
        "",
        "| Model | Training includes Svanström negatives | PR-AUC [95% CI] | "
        "ROC-AUC | Recall | FPR | Test recordings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["models"].items():
        lo, hi = metrics["pr_auc_ci95"]
        added = "yes" if "with_svanstrom" in name else "no"
        lines.append(
            f"| {name} | {added} | {metrics['pr_auc']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] | {metrics['roc_auc']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['fpr']:.3f} | "
            f"{metrics['test_recording_count']} |"
        )
    lines.extend(["", "## False alarms and recording-level results", ""])
    for name, metrics in report["models"].items():
        lines.extend([f"### {name}", "", "| Category | Window FPR | False alarms / windows | "
                      "Recording alarm rate | Alarms / recordings |",
                      "|---|---:|---:|---:|---:|"])
        category_fpr = {
            row["category"]: row for row in metrics["false_alarms_by_category"]
        }
        recording_rates = {
            row["category"]: row for row in metrics["recording_level"]
        }
        for category in sorted(set(category_fpr) | set(recording_rates)):
            fpr = category_fpr.get(category, {})
            recording = recording_rates.get(category, {})
            window_rate = (
                f"{fpr['fa_rate']:.3f}" if "fa_rate" in fpr else "—"
            )
            window_count = (
                f"{int(fpr['false_alarms'])}/{int(fpr['clips'])}"
                if "clips" in fpr else "—"
            )
            lines.append(
                f"| {category} | {window_rate} | {window_count} | "
                f"{recording.get('alarm_rate', 0):.3f} | "
                f"{int(recording.get('alarms', 0))}/{int(recording.get('recordings', 0))} |"
            )
        lines.append("")
    lines.extend([
        "## Data gaps and limitations",
        "",
        report["aircraft_status"],
        "",
        *[f"- {limitation}" for limitation in report["limitations"]],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    report = evaluate()
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(REPORT_MD.read_text(encoding="utf-8"))
    print(f"\nWrote {REPORT_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
