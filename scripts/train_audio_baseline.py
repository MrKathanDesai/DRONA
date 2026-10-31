#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.parquet"
REPORT = ROOT / "reports" / "audio_baseline.json"
FIGURE = ROOT / "reports" / "figures" / "audio_baseline_confusion_matrix.png"

SAMPLE_RATE = 16_000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
RANDOM_STATE = 42


def extract_features(path: str | Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz audio, found {sample_rate} Hz")
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono audio, found {audio.shape[1]} channels")
    waveform = audio[:, 0]
    if waveform.size == 0:
        raise ValueError(f"{path}: audio file is empty")

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=sample_rate / 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate((log_mel.mean(axis=1), log_mel.std(axis=1))).astype(
        np.float32)


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        raise ValueError("Cannot choose a threshold without validation examples")
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + np.finfo(float).eps)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, probabilities: np.ndarray,
             threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, predictions, labels=[0, 1],
            target_names=["background", "drone"], output_dict=True,
            zero_division=0),
    }


def main() -> int:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Audio manifest not found at {MANIFEST}. "
            "Run scripts/build_manifest.py first.")

    manifest = pd.read_parquet(MANIFEST)
    required = {"path", "label", "group_id", "split_tier1", "exclude_reason"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    usable = manifest.loc[
        (manifest["exclude_reason"] == "")
        & manifest["split_tier1"].isin(["train", "val", "test"])
    ].copy()
    if set(usable["label"].unique()) != {"background", "drone"}:
        raise ValueError("The usable Tier 1 data must contain both classes")

    for split in ("train", "val", "test"):
        subset = usable.loc[usable["split_tier1"] == split]
        if subset.empty or subset["label"].nunique() != 2:
            raise ValueError(f"Tier 1 {split} split is empty or missing a class")

    split_groups = usable.groupby("group_id")["split_tier1"].nunique()
    if (split_groups > 1).any():
        raise ValueError("A recording group spans multiple Tier 1 splits")

    feature_cache: dict[str, np.ndarray] = {}
    features = []
    labels = []
    splits = []
    paths = usable["path"].tolist()
    for index, path in enumerate(paths, start=1):
        if path not in feature_cache:
            feature_cache[path] = extract_features(path)
        features.append(feature_cache[path])
        labels.append(usable.iloc[index - 1]["label"] == "drone")
        splits.append(usable.iloc[index - 1]["split_tier1"])
        if index % 1000 == 0 or index == len(paths):
            print(f"Extracted features for {index}/{len(paths)} clips")

    x = np.stack(features)
    y = np.asarray(labels, dtype=np.uint8)
    split_values = np.asarray(splits)
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    test_mask = split_values == "test"

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x[train_mask], y[train_mask])

    val_probabilities = model.predict_proba(x[val_mask])[:, 1]
    tuned_threshold = best_f1_threshold(y[val_mask], val_probabilities)
    test_probabilities = model.predict_proba(x[test_mask])[:, 1]
    test_default = evaluate(y[test_mask], test_probabilities, 0.5)
    test_tuned = evaluate(y[test_mask], test_probabilities, tuned_threshold)

    report = {
        "project": "DRONA audio-only classical baseline",
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": 300,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
            "input_features": (
                "64-bin log-mel spectrogram temporal means followed by "
                "64 temporal standard deviations"),
            "feature_dimensions": int(x.shape[1]),
            "sample_rate_hz": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
        },
        "split_policy": (
            "Existing manifest Tier 1 group-based train/val/test split; "
            "threshold selected on validation only"),
        "counts": {
            "usable_clips": int(len(usable)),
            "excluded_manifest_clips": int((manifest["exclude_reason"] != "").sum()),
            "clips_by_split": {
                split: {
                    label: int(((split_values == split) & (y == (label == "drone"))).sum())
                    for label in ("background", "drone")
                }
                for split in ("train", "val", "test")
            },
            "groups_by_split": {
                split: int(usable.loc[
                    usable["split_tier1"] == split, "group_id"].nunique())
                for split in ("train", "val", "test")
            },
        },
        "validation": {
            "threshold_maximizing_f1": tuned_threshold,
            "f1_at_threshold": float(f1_score(
                y[val_mask], val_probabilities >= tuned_threshold,
                zero_division=0)),
        },
        "test": {
            "threshold_0_5": test_default,
            "validation_tuned_threshold": test_tuned,
        },
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y[test_mask],
        test_probabilities >= tuned_threshold,
        labels=[0, 1],
        display_labels=["Background", "Drone"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Audio baseline — Tier 1 test, validation-tuned threshold")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)

    print("\nTier 1 test results (threshold selected using validation):")
    for name in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
        print(f"  {name:10s} {test_tuned[name]:.4f}")
    print(f"  threshold  {tuned_threshold:.6f}")
    print(f"\nReport: {REPORT}")
    print(f"Confusion matrix: {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
