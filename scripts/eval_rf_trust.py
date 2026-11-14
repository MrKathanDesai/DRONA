#!/usr/bin/env python3
"""Calibrate DroneRF probabilities and evaluate held-out Phantom as unknown.

    python scripts/eval_rf_trust.py

Requires the complete, checksummed DroneRF feature cache from
`python scripts/prepare_rf.py`. The evaluation is within one recording session;
it is not independent-session validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dronedet.data import rf as R
from dronedet.eval import calibration as C
from dronedet.eval import metrics as E
from dronedet.eval import open_set as O
from dronedet.models import rf as M
from dronedet.paths import DATA_DIR

FEATURES = DATA_DIR / "rf_features"
REPORT_JSON = ROOT / "reports" / "rf_trust.json"
REPORT_MD = ROOT / "reports" / "rf_trust.md"
MODEL_OUT = DATA_DIR / "models" / "rf_trust.joblib"
TARGET_FPR = 0.01
KNOWN_ACCEPTANCE = 0.95


def _load_features() -> tuple[np.ndarray, pd.DataFrame, dict]:
    required = (
        FEATURES / "X.npy",
        FEATURES / "index.parquet",
        FEATURES / "config.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "RF trust evaluation requires the prepared feature cache; missing: "
            + ", ".join(str(path) for path in missing)
            + ". Run `python scripts/prepare_rf.py` after verifying all DroneRF archives."
        )
    config = json.loads((FEATURES / "config.json").read_text(encoding="utf-8"))
    if len(config.get("archives", [])) != 23:
        raise ValueError(
            f"complete DroneRF data requires 23 archives; found "
            f"{len(config.get('archives', []))}"
        )
    features = np.load(FEATURES / "X.npy", mmap_mode="r")
    index = pd.read_parquet(FEATURES / "index.parquet")
    if len(features) != len(index) or features.ndim != 3 or features.shape[1] != 2:
        raise ValueError(
            f"RF cache is not aligned or has the wrong shape: "
            f"{features.shape}, {len(index)} rows"
        )
    if not np.isfinite(features).all():
        raise ValueError("RF feature cache contains non-finite values")
    return features, index, config


def _assert_disjoint(df: pd.DataFrame, masks: list[np.ndarray]) -> None:
    groups = [set(df.loc[mask, "group_id"]) for mask in masks]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if groups[left] & groups[right]:
                raise RuntimeError("RF recording groups overlap evaluation splits")


def _binary_calibration(features: np.ndarray, df: pd.DataFrame) -> dict:
    train = df.split_tier1.eq("train").to_numpy()
    validation = df.split_tier1.eq("val").to_numpy()
    test = df.split_tier1.eq("test").to_numpy()
    _assert_disjoint(df, [train, validation, test])
    model = M.forest().fit(features[train], df.y[train])
    y_val = df.loc[validation, "y"].to_numpy(dtype=int)
    y_test = df.loc[test, "y"].to_numpy(dtype=int)
    val_raw = model.predict_proba(features[validation])[:, 1]
    test_raw = model.predict_proba(features[test])[:, 1]
    temperature = C.fit_temperature(val_raw, y_val)
    val_scaled = C.temperature_scale(val_raw, temperature)[:, 1]
    test_scaled = C.temperature_scale(test_raw, temperature)[:, 1]
    raw_threshold = E.threshold_at_fpr(val_raw[y_val == 0], TARGET_FPR)
    scaled_threshold = E.threshold_at_fpr(val_scaled[y_val == 0], TARGET_FPR)
    return {
        "temperature": temperature,
        "threshold_at_validation_fpr": scaled_threshold,
        "raw_threshold_at_validation_fpr": raw_threshold,
        "validation": {
            "raw": C.calibration_metrics(val_raw, y_val),
            "temperature_scaled": C.calibration_metrics(val_scaled, y_val),
        },
        "test": {
            "raw": C.calibration_metrics(test_raw, y_test),
            "temperature_scaled": C.calibration_metrics(test_scaled, y_test),
            "raw_operating_point": E.operating_point(
                y_test, test_raw, raw_threshold
            ),
            "temperature_scaled_operating_point": E.operating_point(
                y_test, test_scaled, scaled_threshold
            ),
        },
        "model": model,
    }


def _multiclass_open_set(features: np.ndarray, df: pd.DataFrame) -> dict:
    known_models = ("bebop", "ar")
    known_labels = ("background", *known_models)
    labels = df.drone_model.fillna("background").to_numpy()
    train = (
        df.split_tier1.eq("train").to_numpy()
        & np.isin(labels, known_labels)
    )
    validation = (
        df.split_tier1.eq("val").to_numpy()
        & np.isin(labels, known_labels)
    )
    test = df.split_tier1.eq("test").to_numpy()
    _assert_disjoint(df, [train, validation, test])
    if set(labels[train]) != set(known_labels):
        raise ValueError("known-model RF training data must include background, Bebop, and AR")
    if "phantom" not in set(labels[test]):
        raise ValueError("held-out Phantom test windows are required for open-set evaluation")

    model = M.forest().fit(features[train], labels[train])
    class_index = {label: index for index, label in enumerate(model.classes_)}
    y_val = np.asarray([class_index[label] for label in labels[validation]], dtype=int)
    raw_val = model.predict_proba(features[validation])
    temperature = C.fit_temperature(raw_val, y_val)
    calibrated_val = C.temperature_scale(raw_val, temperature)
    known_drone_indices = [class_index[name] for name in known_models]
    val_drone_mask = np.isin(labels[validation], known_models)
    val_unknownness = O.unknownness_scores(
        calibrated_val[val_drone_mask], known_drone_indices
    )
    cutoff = float(np.quantile(val_unknownness, 1 - KNOWN_ACCEPTANCE))

    raw_test = model.predict_proba(features[test])
    calibrated_test = C.temperature_scale(raw_test, temperature)
    test_labels = labels[test]
    unknown_mask = test_labels == "phantom"
    known_drone_mask = np.isin(test_labels, known_models)
    novelty_mask = unknown_mask | known_drone_mask
    novelty_scores = O.unknownness_scores(
        calibrated_test[novelty_mask], known_drone_indices
    )
    novelty_truth = unknown_mask[novelty_mask]
    open_metrics = O.open_set_metrics(novelty_scores, novelty_truth, cutoff)
    known_test_mask = np.isin(test_labels, known_labels)
    test_known_class_ids = np.asarray(
        [class_index[label] for label in test_labels[known_test_mask]], dtype=int
    )
    calibration_test = C.calibration_metrics(
        calibrated_test[known_test_mask], test_known_class_ids
    )
    prediction = model.classes_[raw_test.argmax(axis=1)]
    confusion = confusion_matrix(
        test_labels[known_test_mask],
        prediction[known_test_mask],
        labels=list(model.classes_),
    )
    return {
        "known_classes": list(model.classes_),
        "unknown_drone_model": "phantom",
        "temperature": temperature,
        "unknownness_definition": (
            "1 - maximum calibrated probability among the known drone classes "
            "(Bebop, AR); background confidence is not treated as drone familiarity."
        ),
        "acceptance_threshold_calibrated_on": (
            f"the {KNOWN_ACCEPTANCE:.0%} known-drone acceptance quantile on "
            "Tier 1 validation recordings"
        ),
        "validation_known_class_calibration": C.calibration_metrics(
            calibrated_val, y_val
        ),
        "test_known_class_calibration": calibration_test,
        "test_known_class_confusion": {
            "labels": list(model.classes_),
            "matrix": confusion.tolist(),
        },
        "open_set_test": open_metrics,
        "test_recordings": int(df.loc[test, "group_id"].nunique()),
        "model": model,
    }


def evaluate() -> dict:
    X, index, config = _load_features()
    df = R.assign_splits(index.drop(columns=[
        column for column in ("block", "group_id", "split_tier1", "split_tier2",
                              "split_tier3", "split_tier4")
        if column in index
    ]))
    if not df.path.equals(index.path):
        raise RuntimeError("RF index ordering changed while restoring validated splits")
    if len(df) != len(X):
        raise RuntimeError("RF split index and feature cache are not aligned")
    df["bg_category"] = np.where(df.label == "background", "rf_background", None)
    from dronedet.eval.protocol import with_category

    df = with_category(df)
    features = M.band_features(M.shape_features(X))

    binary = _binary_calibration(features, df)
    open_set = _multiclass_open_set(features, df)
    artifact = {
        "binary_model": binary.pop("model"),
        "binary_temperature": binary["temperature"],
        "binary_threshold": binary["threshold_at_validation_fpr"],
        "open_set_model": open_set.pop("model"),
        "open_set_temperature": open_set["temperature"],
        "unknownness_threshold": open_set["open_set_test"][
            "acceptance_unknownness_threshold"
        ],
        "known_class_indices": {
            name: int(index)
            for index, name in enumerate(open_set["known_classes"])
        },
        "feature_fingerprint": config["fingerprint"],
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)

    return {
        "title": "DroneRF calibration and unknown-signature experiment",
        "source": "DroneRF; single lab session",
        "feature": "Two-band, level-invariant 64-band summary of each 25 ms Welch PSD.",
        "split": (
            "Chronological, recording-grouped Tier 1 train/validation/test. Phantom "
            "is excluded from open-set training and validation and is evaluated only "
            "as the held-out unknown drone model."
        ),
        "binary_target_fpr": TARGET_FPR,
        "binary_calibration": binary,
        "open_set": open_set,
        "limitations": [
            "All recordings are from one lab session; this cannot establish performance on a new receiver, location, or session.",
            "Phantom is one held-out drone model, not a representative sample of unknown transmitters.",
            "Temperature scaling and unknownness cutoffs use validation recordings only; test recordings are not used to fit them.",
            "ECE is top-label expected calibration error with ten equal-width probability bins.",
            "This artifact is for the offline replay panel and research inspection; it is not connected to a live SDR.",
        ],
        "artifact": str(MODEL_OUT.relative_to(ROOT)),
    }


def render_markdown(report: dict) -> str:
    binary = report["binary_calibration"]
    opened = report["open_set"]
    lines = [
        "# DroneRF calibration and unknown-signature experiment",
        "",
        "**Research-only, single-session results.** This does not establish "
        "independent-session RF detection.",
        "",
        "## Binary drone/background probability calibration",
        "",
        "| Evaluation | ECE | Brier | NLL | Samples |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for kind in ("raw", "temperature_scaled"):
            result = binary[split][kind]
            lines.append(
                f"| {split} {kind.replace('_', ' ')} | {result['ece']:.4f} | "
                f"{result['brier_score']:.4f} | {result['nll']:.4f} | "
                f"{result['sample_count']} |"
            )
    lines.extend([
        "",
        f"Validation-frozen temperature: {binary['temperature']:.4f}. "
        f"Threshold selected for validation FPR {TARGET_FPR:.0%}: "
        f"{binary['threshold_at_validation_fpr']:.4f}.",
        "",
        "## Held-out Phantom unknown-signature probe",
        "",
        "Known training classes are background, Bebop and AR. Phantom is excluded "
        "from model fitting and calibration. Unknownness is one minus the largest "
        "calibrated probability across known drone classes.",
        "",
        "| Unknown ROC-AUC | Known acceptance | Phantom rejection | Known test | Phantom test |",
        "|---:|---:|---:|---:|---:|",
        f"| {opened['open_set_test']['unknown_roc_auc']:.3f} | "
        f"{opened['open_set_test']['known_accept_rate']:.3f} | "
        f"{opened['open_set_test']['unknown_reject_rate']:.3f} | "
        f"{opened['open_set_test']['known_count']} | "
        f"{opened['open_set_test']['unknown_count']} |",
        "",
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in report["limitations"]],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    try:
        report = evaluate()
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(REPORT_MD.read_text(encoding="utf-8"))
    print(f"\nWrote {REPORT_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
