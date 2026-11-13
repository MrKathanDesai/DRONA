#!/usr/bin/env python3
"""Compare compact audio architectures on shared, group-safe manifest splits.

    python scripts/compare_audio_architectures.py

Uses cached 128-band log-mel features, pooled to 64 bands and 21 frames. Each
model is trained once on the indoor Tier 1 train split, then tested on both
Tier 1 and outdoor Tier 3. This comparison does not replace the deployed detector.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dronedet.eval import metrics as E
from dronedet.models.audio_architectures import MODEL_FACTORIES
from dronedet.paths import DATA_DIR

FEATURES = DATA_DIR / "features"
REPORT_JSON = ROOT / "reports" / "audio_architecture_comparison.json"
REPORT_MD = ROOT / "reports" / "audio_architecture_comparison.md"
CHECKPOINT_DIR = DATA_DIR / "models" / "audio_architecture_comparison"
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 8
PATIENCE = 2
TARGET_FPR = 0.01


class FeatureDataset(Dataset):
    def __init__(self, features, labels, indices, mean, std) -> None:
        self.features = features
        self.labels = labels
        self.indices = indices
        self.mean = mean[:, None]
        self.std = std[:, None]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.indices[index]
        values = np.asarray(self.features[row], dtype=np.float32)
        values = (values - self.mean) / (self.std + 1e-6)
        values = values.reshape(64, 2, 21, 3).mean(axis=(1, 3))
        return (
            torch.from_numpy(np.ascontiguousarray(values)),
            torch.tensor(self.labels[row], dtype=torch.float32),
        )


def _validate_alignment(index: pd.DataFrame, manifest: pd.DataFrame) -> None:
    usable = manifest.loc[manifest.exclude_reason.eq("")].reset_index(drop=True)
    if not index.path.equals(usable.path):
        raise RuntimeError(
            "feature cache does not match the current usable manifest; "
            "run scripts/make_features.py"
        )
    if index.group_id.isna().any():
        raise ValueError("manifest has missing recording group IDs")


def _feature_normalization(features, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    totals = np.zeros(features.shape[1], dtype=np.float64)
    squares = np.zeros(features.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 128):
        batch = np.asarray(features[indices[start:start + 128]], dtype=np.float64)
        totals += batch.sum(axis=(0, 2))
        squares += np.square(batch).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]
    mean = totals / count
    variance = np.maximum(squares / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _probabilities(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for inputs, _ in loader:
            logits = model(inputs.to(device))
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def _train_one(
    architecture: str,
    features,
    df: pd.DataFrame,
    device: torch.device,
) -> dict:
    train_mask = df.split_tier1.eq("train").to_numpy(copy=True)
    val_mask = df.split_tier1.eq("val").to_numpy(copy=True)
    usable = df.exclude_reason.eq("").to_numpy()
    train_mask &= usable
    val_mask &= usable
    test_masks = {
        "tier1_test": df.split_tier1.eq("test").to_numpy(copy=True) & usable,
        "tier3_outdoor": df.split_tier3.eq("test").to_numpy(copy=True) & usable,
    }
    if not train_mask.any() or not val_mask.any() or any(
        not mask.any() for mask in test_masks.values()
    ):
        raise ValueError("Tier 1 train/validation and both evaluation tests are required")

    group_sets = [set(df.loc[mask, "group_id"]) for mask in (
        train_mask, val_mask, *test_masks.values()
    )]
    for left in range(len(group_sets)):
        if any(group_sets[left] & group_sets[right]
               for right in range(left + 1, len(group_sets))):
            raise RuntimeError("recording groups overlap across architecture comparison splits")
    for name, mask in (("train", train_mask), ("validation", val_mask),
                       *test_masks.items()):
        if df.loc[mask, "label"].nunique() != 2:
            raise ValueError(f"{name} split does not contain both classes")

    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    test_indices = {
        name: np.flatnonzero(mask) for name, mask in test_masks.items()
    }
    labels = df.label.eq("drone").to_numpy(dtype=np.float32)
    mean, std = _feature_normalization(features, train_indices)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        FeatureDataset(features, labels, train_indices, mean, std),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        FeatureDataset(features, labels, val_indices, mean, std),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loaders = {
        name: DataLoader(
            FeatureDataset(features, labels, indices, mean, std),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        for name, indices in test_indices.items()
    }

    seed_offset = list(MODEL_FACTORIES).index(architecture) * 100
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    torch.manual_seed(SEED + seed_offset)
    model = MODEL_FACTORIES[architecture]().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_labels = labels[train_indices]
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    if not positives or not negatives:
        raise ValueError("Tier 1 training data is missing a class")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            examples += len(targets)

        validation = _probabilities(model, val_loader, device)
        val_truth = labels[val_indices].astype(np.uint8)
        from sklearn.metrics import average_precision_score

        val_pr_auc = float(average_precision_score(val_truth, validation))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / examples,
            "validation_pr_auc": val_pr_auc,
        })
        print(
            f"    {architecture} epoch {epoch}: "
            f"loss={total_loss / examples:.4f} val_PR_AUC={val_pr_auc:.4f}",
            flush=True,
        )
        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"{architecture} produced no validation checkpoint")
    model.load_state_dict(best_state)
    val_scores = _probabilities(model, val_loader, device)
    threshold = E.threshold_at_fpr(
        val_scores[labels[val_indices] == 0], TARGET_FPR
    )
    validation_operating_point = E.operating_point(
        labels[val_indices].astype(np.uint8), val_scores, threshold
    )
    evaluations = {}
    for name, loader in test_loaders.items():
        indices = test_indices[name]
        test_scores = _probabilities(model, loader, device)
        test_truth = labels[indices].astype(np.uint8)
        test_groups = df.loc[test_masks[name], "group_id"].to_numpy()
        metrics = E.summarize(test_truth, test_scores, test_groups, threshold)
        predictions = test_scores >= threshold
        metrics["confusion_matrix"] = confusion_matrix(
            test_truth, predictions, labels=[0, 1]
        ).tolist()
        metrics["classification_report"] = classification_report(
            test_truth,
            predictions,
            labels=[0, 1],
            target_names=["background", "drone"],
            output_dict=True,
            zero_division=0,
        )
        metrics["test_recording_groups"] = int(len(set(test_groups)))
        metrics["validation_operating_point"] = validation_operating_point
        evaluations[name] = metrics

    training = {
        "threshold": float(threshold),
        "validation_pr_auc_best": best_score,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "parameters": parameter_count,
        "training_seconds": round(time.perf_counter() - started, 2),
        "split_counts": {
            split: {
                "clips": int(mask.sum()),
                "recording_groups": int(df.loc[mask, "group_id"].nunique()),
                "background": int((df.loc[mask, "label"] == "background").sum()),
                "drone": int((df.loc[mask, "label"] == "drone").sum()),
            }
            for split, mask in (("train", train_mask), ("validation", val_mask))
        },
        "history": history,
        "threshold_policy": "1% validation-negative FPR; fixed before test scoring",
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{architecture}_tier1.pt"
    torch.save({
        "state_dict": best_state,
        "feature_mean": mean,
        "feature_std": std,
        "architecture": architecture,
        "tier": "tier1",
        "threshold": threshold,
    }, checkpoint)
    training["checkpoint"] = str(checkpoint.relative_to(ROOT))
    return {"training": training, "evaluations": evaluations}


def evaluate() -> dict:
    manifest_path = DATA_DIR / "manifest.parquet"
    index_path = FEATURES / "index.parquet"
    feature_path = FEATURES / "X.npy"
    config_path = FEATURES / "config.json"
    for path in (manifest_path, index_path, feature_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"required cached input is missing: {path}")
    manifest = pd.read_parquet(manifest_path)
    index = pd.read_parquet(index_path)
    _validate_alignment(index, manifest)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_shape = config.get("shape")
    features = np.load(feature_path, mmap_mode="r")
    if list(features.shape) != expected_shape or features.shape[1:] != (128, 63):
        raise ValueError(
            f"unexpected feature cache shape/config: {features.shape}, {expected_shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("feature cache contains non-finite values")

    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for architecture in MODEL_FACTORIES:
        print(f"  training {architecture} on Tier 1", flush=True)
        results[architecture] = _train_one(
            architecture, features, index, device
        )
    return {
        "title": "Audio architecture comparison",
        "feature": {
            "cache": "data/features/X.npy",
            "shape_per_clip": [64, 21],
            "source_shape_per_clip": list(features.shape[1:]),
            "pooling": "Adjacent mel bands averaged in pairs; every 3 time frames averaged.",
            "bands": 64,
            "frames": 21,
            "normalization": (
                "Per-mel-band mean/std fit on Tier 1 training data only; "
                "the same transform is applied to validation and both test sets."
            ),
        },
        "protocol": {
            "training": (
                "All models are trained on the same indoor Tier 1 train recordings; "
                "Tier 1 validation selects the checkpoint and threshold."
            ),
            "tier1_test": "Held-out indoor recordings; groups do not overlap train or validation.",
            "tier3_outdoor": (
                "The same frozen models and indoor-validation thresholds are applied "
                "to Svanström. This corpus was used in earlier project evaluation "
                "and is not a blind external validation."
            ),
            "threshold": (
                "Each model selects a threshold on validation negatives at "
                f"{TARGET_FPR:.0%} FPR; the threshold is frozen for test."
            ),
            "selection": (
                "Best checkpoint selected by validation PR-AUC with early stopping."
            ),
            "seed": SEED,
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
        "architectures": {
            "small_cnn": "Three compact Conv2d-BatchNorm-GELU blocks and global pooling.",
            "cnn_gru": "Two temporal Conv1d blocks followed by a bidirectional GRU.",
            "cnn_lstm": "Two temporal Conv1d blocks followed by a bidirectional LSTM.",
            "lightweight_transformer": (
                "64-dimensional mel-frame projection with two 4-head transformer "
                "encoder layers and mean pooling."
            ),
        },
        "models": results,
        "limitations": [
            "This is an experimental architecture comparison; no model is promoted to deployment.",
            "Tier 3 uses Svanström, which was used in earlier project evaluation, so it is not an untouched independent corpus.",
            "Architecture results depend on this feature representation and training protocol; the existing AudioCNN used a different 64-bin feature extraction setup.",
            "Tier 1 estimates same-corpus generalization; the outdoor tier is the key transfer check.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Audio architecture comparison",
        "",
        "All four models use the same cached log-mel inputs pooled to 64 bands and "
        "21 frames, training-only per-band normalization, group-aware splits, and a threshold fixed on "
        "validation negatives at 1% FPR. Best checkpoint is selected by validation "
        "PR-AUC. No test split is used for model selection.",
        "",
        "## Results",
        "",
        "| Evaluation | Architecture | PR-AUC [95% CI] | ROC-AUC | Recall | FPR | "
        "False alarms / negatives | Parameters | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture, model in report["models"].items():
        for evaluation, result in model["evaluations"].items():
            lo, hi = result["pr_auc_ci95"]
            lines.append(
                f"| {evaluation} | {architecture} | {result['pr_auc']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] | {result['roc_auc']:.3f} | "
                f"{result['recall']:.3f} | {result['fpr']:.3f} | "
                f"{result['false_alarms']}/{result['negatives']} | "
                f"{model['training']['parameters']:,} | "
                f"{model['training']['best_epoch']} |"
            )
    lines.extend(["", "## Protocol and caveats", ""])
    lines.extend(
        f"- **{phase}:** {description}"
        for phase, description in report["protocol"].items()
        if phase in {"training", "tier1_test", "tier3_outdoor", "threshold", "selection"}
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(["", "## Confusion matrices", ""])
    for architecture, model in report["models"].items():
        lines.extend([f"### {architecture}", ""])
        for evaluation, result in model["evaluations"].items():
            lines.append(
                f"- **{evaluation}:** rows actual (background, drone), columns "
                f"predicted (background, drone): `{result['confusion_matrix']}`."
            )
    lines.append("")
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
