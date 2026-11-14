#!/usr/bin/env python3
"""Exploratory, recording-held-out classification of available DroneDetect codes.

This is not a drone-vs-background detector: the local subset contains no
background recordings. It evaluates only the MA1/MAV filename prefixes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                            classification_report, confusion_matrix, f1_score,
                            roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from dronedet.data.dronedetect import DEFAULT, find_recordings, psd_windows
from dronedet.models.rf import band_features, logreg
from dronedet.paths import DATA_DIR

SOURCE = (
    "https://ieee-dataport.org/open-access/"
    "dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine"
)
REPORT_JSON = ROOT / "reports" / "dronedetect_subset.json"
REPORT_MD = ROOT / "reports" / "dronedetect_subset.md"
N_SPLITS = 5
THRESHOLD = 0.5


def evaluate(root: Path) -> dict:
    available = find_recordings(root)
    recordings = [r for r in available if r.mode in {"hovering", "flying"}]
    classes = sorted({recording.model_code for recording in recordings})
    if len(classes) != 2:
        raise ValueError(
            f"This limited experiment expects exactly two model codes; found {classes}"
        )
    mode_counts = {
        mode: {
            code: sum(r.mode == mode and r.model_code == code for r in recordings)
            for code in classes
        }
        for mode in ("hovering", "flying")
    }
    if any(
        not counts[classes[0]]
        or not counts[classes[1]]
        or counts[classes[0]] != counts[classes[1]]
        for counts in mode_counts.values()
    ):
        raise ValueError(f"Flight modes are not balanced across filename codes: {mode_counts}")

    feature_rows = []
    labels = []
    groups = []
    tails = []
    for recording_id, recording in enumerate(recordings):
        psd, tail_samples = psd_windows(recording.path)
        feature_rows.append(band_features(psd[:, None, :], DEFAULT.n_bands))
        labels.extend([recording.model_code] * len(psd))
        groups.extend([recording_id] * len(psd))
        tails.append(tail_samples)
        print(f"  processed {recording.path.name}: {len(psd)} windows", flush=True)

    X = np.concatenate(feature_rows)
    y = np.asarray(labels)
    groups = np.asarray(groups)
    recording_scores = np.full(len(recordings), np.nan)
    fold_by_recording = np.zeros(len(recordings), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

    for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        if train_groups & test_groups:
            raise RuntimeError("recording leakage across train and test splits")
        model = logreg()
        model.fit(X[train], y[train])
        positive_index = list(model.classes_).index(classes[1])
        scores = model.predict_proba(X[test])[:, positive_index]
        for recording_id in sorted(test_groups):
            recording_scores[recording_id] = float(
                scores[groups[test] == recording_id].mean()
            )
            fold_by_recording[recording_id] = fold

    if not np.isfinite(recording_scores).all():
        raise RuntimeError("cross-validation did not produce one prediction per recording")

    truth = np.asarray([recording.model_code for recording in recordings])
    prediction = np.where(recording_scores >= THRESHOLD, classes[1], classes[0])
    report = classification_report(
        truth, prediction, labels=classes, output_dict=True, zero_division=0
    )
    confusion = confusion_matrix(truth, prediction, labels=classes)
    mode_metrics = {}
    for mode in ("hovering", "flying"):
        keep = np.asarray([recording.mode == mode for recording in recordings])
        mode_truth = truth[keep]
        mode_prediction = prediction[keep]
        mode_metrics[mode] = {
            "recording_count": int(keep.sum()),
            "accuracy": float(accuracy_score(mode_truth, mode_prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(mode_truth, mode_prediction)
            ),
            "macro_f1": float(f1_score(mode_truth, mode_prediction, average="macro")),
        }
    report_data = {
        "title": "DroneDetect V2 matched-mode model-code classification",
        "source_url": SOURCE,
        "doi": "10.21227/5jjj-1m32",
        "scope": (
            "Exploratory classification between filename prefixes on matched flight "
            "modes; not drone detection."
        ),
        "recording_count": len(recordings),
        "available_recording_count": len(available),
        "window_count": int(len(y)),
        "window_seconds": DEFAULT.window_s,
        "feature": (
            "64-band summary of per-window, two-sided Welch log-PSD; "
            "per-window mean removed"
        ),
        "split": (
            f"{N_SPLITS}-fold stratified group CV; all windows from each recording "
            "stay in one fold"
        ),
        "threshold": THRESHOLD,
        "classes": classes,
        "metrics": {
            "recording_accuracy": float(accuracy_score(truth, prediction)),
            "recording_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "recording_macro_f1": float(f1_score(truth, prediction, average="macro")),
            "recording_roc_auc": float(
                roc_auc_score(truth == classes[1], recording_scores)
            ),
            "by_mode": mode_metrics,
            "classification_report": report,
            "confusion_matrix": confusion.tolist(),
        },
        "coverage": {
            "interference_codes": sorted({r.interference_code for r in recordings}),
            "interference": sorted({r.interference for r in recordings}),
            "model_codes": {
                code: sum(r.model_code == code for r in recordings) for code in classes
            },
            "modes": sorted({r.mode for r in recordings}),
            "mode_counts_by_code": mode_counts,
            "excluded_unmatched_mode_recordings": len(available) - len(recordings),
            "complete_windows_per_recording": sorted(
                {int(
                    (r.path.stat().st_size // np.dtype(np.complex64).itemsize)
                    // DEFAULT.samples_per_window
                ) for r in recordings}
            ),
        },
        "limitations": [
            "Only 20 of 25 local recordings are evaluated: two filename prefixes, Bluetooth-only, and matched hovering/flying modes.",
            "Five switched-on recordings are excluded because the MAV prefix has no switched-on examples.",
            "No background/no-drone captures are present, so binary detection metrics are not valid.",
            "The local subset is too small to support deployment or generalization claims.",
            "Recording-level folds prevent window leakage, but captures may share the same setup.",
            "Model codes are reported as filenames; the experiment does not infer a deployed identity.",
        ],
        "recordings": [
            {
                "file": recording.path.name,
                "model_code": recording.model_code,
                "interference": recording.interference,
                "mode": recording.mode,
                "take": recording.take,
                "fold": int(fold_by_recording[index]),
                "mean_score_for_second_class": float(recording_scores[index]),
                "predicted_code": str(prediction[index]),
                "tail_samples_dropped": int(tails[index]),
            }
            for index, recording in enumerate(recordings)
        ],
    }
    return report_data


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    labels = report["classes"]
    matrix = metrics["confusion_matrix"]
    rows = [
        "# DroneDetect V2 subset: exploratory model-code classification",
        "",
        "**Not a drone-vs-background detection result.** The local subset contains",
        "only two filename prefixes and one interference condition, with no background",
        "recordings. Five switched-on captures are excluded to compare matched modes.",
        "",
        f"Source: [IEEE DataPort DroneDetect]({report['source_url']}) "
        f"(DOI `{report['doi']}`).",
        "",
        f"- Recordings evaluated: {report['recording_count']} of "
        f"{report['available_recording_count']} local",
        f"- 20 ms windows: {report['window_count']}",
        f"- Features: {report['feature']}",
        f"- Evaluation: {report['split']}",
        f"- Coverage: {report['coverage']}",
        "",
        "## Recording-level out-of-fold metrics",
        "",
        "| Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |",
        "|---:|---:|---:|---:|",
        (
            f"| {metrics['recording_accuracy']:.3f} "
            f"| {metrics['recording_balanced_accuracy']:.3f} "
            f"| {metrics['recording_macro_f1']:.3f} "
            f"| {metrics['recording_roc_auc']:.3f} |"
        ),
        "",
        "## Matched-mode recording metrics",
        "",
        "| Mode | Recordings | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {mode} | {values['recording_count']} | {values['accuracy']:.3f} "
        f"| {values['balanced_accuracy']:.3f} | {values['macro_f1']:.3f} |"
        for mode, values in metrics["by_mode"].items()
    )
    rows += [
        "",
        "## Confusion matrix (recordings)",
        "",
        "| Actual \\ Predicted | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    rows.extend(
        f"| {label} | " + " | ".join(str(value) for value in matrix[index]) + " |"
        for index, label in enumerate(labels)
    )
    rows += ["", "## Limitations", ""]
    rows.extend(f"- {item}" for item in report["limitations"])
    rows.append("")
    return "\n".join(rows)


def main() -> int:
    root = DATA_DIR / "RF_datasets" / "datasets_2"
    report = evaluate(root)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    metrics = report["metrics"]
    print(
        f"recordings={report['recording_count']} "
        f"accuracy={metrics['recording_accuracy']:.3f} "
        f"balanced_accuracy={metrics['recording_balanced_accuracy']:.3f} "
        f"macro_f1={metrics['recording_macro_f1']:.3f} "
        f"roc_auc={metrics['recording_roc_auc']:.3f}"
    )
    print(f"wrote {REPORT_MD} and {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
