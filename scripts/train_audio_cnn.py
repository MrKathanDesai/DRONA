#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.parquet"
REPORT = ROOT / "reports" / "audio_cnn.json"
FIGURE = ROOT / "reports" / "figures" / "audio_cnn_confusion_matrix.png"
CHECKPOINT = ROOT / "reports" / "models" / "audio_cnn_state_dict.pt"
BASELINE_REPORT = ROOT / "reports" / "audio_baseline.json"
BASELINE_FIGURE = ROOT / "reports" / "figures" / "audio_baseline_confusion_matrix.png"

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
RANDOM_STATE = 42
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 4
NUM_WORKERS = 0

sys.path.insert(0, str(ROOT / "scripts"))
from train_audio_baseline import best_f1_threshold, evaluate


class AudioCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs)).squeeze(1)


def extract_log_mel(path: str | Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz audio, found {sample_rate} Hz")
    if audio.shape[1] != 1:
        raise ValueError(
            f"{path}: expected mono audio, found {audio.shape[1]} channels")
    waveform = audio[:, 0]
    if waveform.size == 0:
        raise ValueError(f"{path}: audio file is empty")

    if waveform.size > WINDOW_SAMPLES:
        start = (waveform.size - WINDOW_SAMPLES) // 2
        waveform = waveform[start:start + WINDOW_SAMPLES]
    elif waveform.size < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.size))

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
    return log_mel.astype(np.float32)


def load_dataset() -> tuple[
    torch.Tensor, torch.Tensor, np.ndarray, pd.DataFrame, pd.DataFrame
]:
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
    ].copy().reset_index(drop=True)
    if set(usable["label"].unique()) != {"background", "drone"}:
        raise ValueError("The usable Tier 1 data must contain both classes")
    split_groups = usable.groupby("group_id")["split_tier1"].nunique()
    if (split_groups > 1).any():
        raise ValueError("A recording group spans multiple Tier 1 splits")
    for split in ("train", "val", "test"):
        subset = usable.loc[usable["split_tier1"] == split]
        if subset.empty or subset["label"].nunique() != 2:
            raise ValueError(f"Tier 1 {split} split is empty or missing a class")

    spectrograms = []
    for index, path in enumerate(usable["path"], start=1):
        spectrograms.append(extract_log_mel(path))
        if index % 1000 == 0 or index == len(usable):
            print(f"Extracted log-mel features for {index}/{len(usable)} clips")

    x = torch.from_numpy(np.stack(spectrograms)[:, np.newaxis, :, :])
    y = torch.from_numpy((usable["label"] == "drone").to_numpy(dtype=np.float32))
    return x, y, usable["split_tier1"].to_numpy(), usable, manifest


def probabilities_for(model: nn.Module, x: torch.Tensor,
                      indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            logits = model(x[batch_indices].to(device))
            outputs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(outputs, dtype=np.float64)


def main() -> int:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    x, y, split_values, usable, manifest = load_dataset()
    train_indices = np.flatnonzero(split_values == "train")
    val_indices = np.flatnonzero(split_values == "val")
    test_indices = np.flatnonzero(split_values == "test")

    train_labels = y[train_indices]
    positive = float(train_labels.sum())
    negative = float(len(train_labels) - positive)
    if positive == 0 or negative == 0:
        raise ValueError("Training split must contain both classes")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], device=device))

    generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        TensorDataset(x[train_indices], y[train_indices]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )
    model = AudioCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    from sklearn.metrics import average_precision_score

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(targets)
            seen += len(targets)

        val_probabilities = probabilities_for(model, x, val_indices, device)
        val_truth = y[val_indices].numpy()
        val_pr_auc = float(average_precision_score(val_truth, val_probabilities))
        val_f1_at_half = float(f1_score(
            val_truth, val_probabilities >= 0.5, zero_division=0))
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation_pr_auc": val_pr_auc,
            "validation_f1_at_0_5": val_f1_at_half,
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} "
            f"loss={epoch_row['train_loss']:.4f} "
            f"val_pr_auc={val_pr_auc:.4f} "
            f"val_f1@0.5={val_f1_at_half:.4f}")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a best validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_probabilities = probabilities_for(model, x, val_indices, device)
    val_truth = y[val_indices].numpy()
    threshold = best_f1_threshold(val_truth, val_probabilities)
    test_probabilities = probabilities_for(model, x, test_indices, device)
    test_truth = y[test_indices].numpy()
    tuned_metrics = evaluate(test_truth, test_probabilities, threshold)
    default_metrics = evaluate(test_truth, test_probabilities, 0.5)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        CHECKPOINT,
    )

    split_counts = {}
    for split in ("train", "val", "test"):
        subset = usable.loc[usable["split_tier1"] == split]
        split_counts[split] = {
            "background": int((subset["label"] == "background").sum()),
            "drone": int((subset["label"] == "drone").sum()),
            "groups": int(subset["group_id"].nunique()),
        }

    report = {
        "project": "DRONA audio CNN",
        "model": {
            "name": "AudioCNN",
            "input": (
                "64-bin log-mel spectrogram; centered 1-second waveform "
                "window, center-cropped when longer and zero-padded when shorter"),
            "input_shape": [1, N_MELS, 1 + WINDOW_SAMPLES // HOP_LENGTH],
            "sample_rate_hz": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "architecture": (
                "Three Conv2d-BatchNorm-ReLU blocks (16, 32, 64 channels), "
                "max-pooling after first two blocks, adaptive global average "
                "pooling, dropout 0.3, binary logit"),
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "loss": "BCEWithLogitsLoss with training-set positive-class weight",
            "device": str(device),
            "random_state": RANDOM_STATE,
        },
        "split_policy": (
            "Existing manifest Tier 1 recording-group split; best epoch selected "
            "by validation PR-AUC and threshold selected by validation F1"),
        "counts": {
            "usable_clips": int(len(usable)),
            "excluded_manifest_clips": int((manifest["exclude_reason"] != "").sum()),
            "clips_by_split": split_counts,
        },
        "training": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_validation_pr_auc": best_val_pr_auc,
            "history": history,
        },
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "validation": {
            "threshold_maximizing_f1": threshold,
            "f1_at_threshold": float(f1_score(
                val_truth, val_probabilities >= threshold, zero_division=0)),
        },
        "test": {
            "threshold_0_5": default_metrics,
            "validation_tuned_threshold": tuned_metrics,
        },
        "baseline_comparison": None,
    }

    if BASELINE_REPORT.is_file():
        baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        baseline_test = baseline["test"]["validation_tuned_threshold"]
        report["baseline_comparison"] = {
            "random_forest": {
                "precision": baseline_test["precision"],
                "recall": baseline_test["recall"],
                "f1": baseline_test["f1"],
                "pr_auc": baseline_test["pr_auc"],
                "roc_auc": baseline_test["roc_auc"],
            },
            "cnn": {
                key: tuned_metrics[key]
                for key in ("precision", "recall", "f1", "pr_auc", "roc_auc")
            },
            "cnn_minus_random_forest": {
                key: tuned_metrics[key] - baseline_test[key]
                for key in ("precision", "recall", "f1", "pr_auc", "roc_auc")
            },
            "note": (
                "Both models use the same Tier 1 split, but distinct input "
                "representations and validation-selected thresholds."),
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_confusion_matrix(test_truth, test_probabilities >= threshold)
    print("\nTier 1 test results (threshold selected using validation):")
    for key in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
        print(f"  {key:10s} {tuned_metrics[key]:.4f}")
    print(f"  threshold  {threshold:.6f}")
    print(f"  best epoch {best_epoch}; validation PR-AUC {best_val_pr_auc:.4f}")
    print(f"\nReport: {REPORT}")
    print(f"Confusion matrix: {FIGURE}")
    return 0


def save_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["Background", "Drone"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Audio CNN — Tier 1 test, validation-tuned threshold")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
