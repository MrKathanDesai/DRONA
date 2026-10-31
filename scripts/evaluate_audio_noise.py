#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.parquet"
BASELINE_REPORT = ROOT / "reports" / "audio_baseline.json"
CNN_REPORT = ROOT / "reports" / "audio_cnn.json"
CNN_CHECKPOINT = ROOT / "reports" / "models" / "audio_cnn_state_dict.pt"
REPORT = ROOT / "reports" / "audio_noise_robustness.json"
CSV_REPORT = ROOT / "reports" / "audio_noise_robustness.csv"
FIGURE = ROOT / "reports" / "figures" / "audio_noise_robustness.png"

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
RANDOM_STATE = 42
BATCH_SIZE = 128
SNR_LEVELS_DB = (20, 10, 5, 0)
NOISE_TYPES = ("white", "pink")

sys.path.insert(0, str(ROOT / "scripts"))
from train_audio_baseline import best_f1_threshold, evaluate
from train_audio_cnn import AudioCNN


def _read_waveform(path: str | Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz audio, found {sample_rate} Hz")
    if waveform.ndim != 1:
        raise ValueError(
            f"{path}: expected mono audio, found shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError(f"{path}: audio file is empty")
    return waveform


def _rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def _pink_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("Pink noise requires at least two samples")
    spectrum = np.fft.rfft(rng.standard_normal(size))
    frequencies = np.fft.rfftfreq(size)
    weights = np.zeros_like(frequencies)
    weights[1:] = 1.0 / np.sqrt(frequencies[1:])
    noise = np.fft.irfft(spectrum * weights, n=size)
    noise -= noise.mean()
    noise_rms = _rms(noise)
    if noise_rms == 0:
        raise ValueError("Generated pink noise has zero energy")
    return (noise / noise_rms).astype(np.float32)


def _stable_seed(path: str | Path, noise_type: str, snr_db: int) -> int:
    relative_path = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    key = f"{relative_path}|{noise_type}|{snr_db}|{RANDOM_STATE}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little")


def add_noise(
    waveform: np.ndarray,
    snr_db: int,
    noise_type: str,
    seed: int,
    silence_reference_rms: float,
) -> tuple[np.ndarray, dict]:
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Noise input must be a non-empty mono waveform")
    if silence_reference_rms <= 0 or not math.isfinite(silence_reference_rms):
        raise ValueError("Silence reference RMS must be finite and positive")

    signal_rms = _rms(waveform)
    used_silence_reference = signal_rms == 0
    scale_reference = silence_reference_rms if used_silence_reference else signal_rms
    rng = np.random.default_rng(seed)
    if noise_type == "white":
        noise = rng.standard_normal(waveform.size).astype(np.float32)
        noise -= noise.mean()
        noise_rms = _rms(noise)
        if noise_rms == 0:
            raise ValueError("Generated white noise has zero energy")
        noise = noise / noise_rms
    else:
        noise = _pink_noise(waveform.size, rng)

    target_noise_rms = scale_reference / (10.0 ** (snr_db / 20.0))
    noisy = waveform + noise * target_noise_rms
    metadata = {
        "target_snr_db": snr_db,
        "signal_rms": signal_rms,
        "noise_rms": target_noise_rms,
        "used_training_drone_reference_for_silence": used_silence_reference,
        "achieved_snr_db": (
            None if used_silence_reference
            else float(20.0 * math.log10(signal_rms / target_noise_rms))
        ),
    }
    return noisy.astype(np.float32), metadata


def _mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SAMPLE_RATE / 2,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def _baseline_features(waveform: np.ndarray) -> np.ndarray:
    log_mel = _mel_spectrogram(waveform)
    return np.concatenate((log_mel.mean(axis=1), log_mel.std(axis=1))).astype(
        np.float32)


def _cnn_window(waveform: np.ndarray) -> np.ndarray:
    if waveform.size > WINDOW_SAMPLES:
        start = (waveform.size - WINDOW_SAMPLES) // 2
        return waveform[start:start + WINDOW_SAMPLES]
    if waveform.size < WINDOW_SAMPLES:
        return np.pad(waveform, (0, WINDOW_SAMPLES - waveform.size))
    return waveform


def _probabilities_for_cnn(
    model: AudioCNN,
    spectrograms: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(spectrograms), BATCH_SIZE):
            batch = torch.from_numpy(
                spectrograms[start:start + BATCH_SIZE, np.newaxis, :, :]
            ).to(device)
            probabilities.extend(
                torch.sigmoid(model(batch)).cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def _validate_artifacts() -> tuple[dict, dict, float, float]:
    for path in (MANIFEST, BASELINE_REPORT, CNN_REPORT, CNN_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Required model/data artifact not found: {path}")
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    cnn_report = json.loads(CNN_REPORT.read_text(encoding="utf-8"))
    baseline_threshold = float(baseline["validation"]["threshold_maximizing_f1"])
    cnn_threshold = float(cnn_report["validation"]["threshold_maximizing_f1"])
    if not 0 <= baseline_threshold <= 1 or not 0 <= cnn_threshold <= 1:
        raise ValueError("Saved validation thresholds must be probabilities in [0, 1]")
    return baseline, cnn_report, baseline_threshold, cnn_threshold


def main() -> int:
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_report, cnn_report, rf_threshold, cnn_threshold = _validate_artifacts()

    manifest = pd.read_parquet(MANIFEST)
    required = {"path", "label", "group_id", "split_tier1", "exclude_reason"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    usable = manifest.loc[
        (manifest["exclude_reason"] == "")
        & manifest["split_tier1"].isin(["train", "val", "test"])
    ].copy().reset_index(drop=True)
    split_overlap = usable.groupby("group_id")["split_tier1"].nunique()
    if (split_overlap > 1).any():
        raise ValueError("A recording group spans multiple Tier 1 splits")

    train = usable.loc[usable["split_tier1"] == "train"].reset_index(drop=True)
    val = usable.loc[usable["split_tier1"] == "val"].reset_index(drop=True)
    test = usable.loc[usable["split_tier1"] == "test"].reset_index(drop=True)
    if any(frame.empty or frame["label"].nunique() != 2 for frame in (train, val, test)):
        raise ValueError("Every Tier 1 split must contain both classes")

    train_drone = train.loc[train["label"] == "drone"]
    training_drone_rms = np.asarray([
        _rms(_read_waveform(path)) for path in train_drone["path"]
    ])
    positive_training_rms = training_drone_rms[training_drone_rms > 0]
    if positive_training_rms.size == 0:
        raise ValueError("Training drone recordings have no measurable signal energy")
    silence_reference_rms = float(np.median(positive_training_rms))

    print(f"Evaluation device: {device}")
    print(f"Training-only drone RMS reference for silent clips: {silence_reference_rms:.6f}")
    print("Rebuilding the fixed Random Forest on the clean training split...")
    x_train = np.stack([
        _baseline_features(_read_waveform(path)) for path in train["path"]
    ])
    y_train = (train["label"] == "drone").to_numpy(dtype=np.uint8)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)

    x_val = np.stack([
        _baseline_features(_read_waveform(path)) for path in val["path"]
    ])
    y_val = (val["label"] == "drone").to_numpy(dtype=np.uint8)
    rf_val_probs = rf_model.predict_proba(x_val)[:, 1]
    reconstructed_rf_threshold = best_f1_threshold(y_val, rf_val_probs)
    if not math.isclose(
        reconstructed_rf_threshold, rf_threshold, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError(
            "Rebuilt Random Forest validation threshold does not match the saved "
            f"baseline ({reconstructed_rf_threshold} != {rf_threshold})")
    del x_train, x_val, y_train, y_val, rf_val_probs

    cnn_model = AudioCNN().to(device)
    state = torch.load(CNN_CHECKPOINT, map_location=device, weights_only=True)
    cnn_model.load_state_dict(state)
    cnn_model.eval()

    y_test = (test["label"] == "drone").to_numpy(dtype=np.uint8)
    test_paths = test["path"].tolist()
    result_rows = []
    condition_results = {}
    silent_count = 0

    conditions = [("clean", None, None)]
    conditions.extend(
        (noise_type, snr_db, noise_type)
        for noise_type in NOISE_TYPES
        for snr_db in SNR_LEVELS_DB
    )

    for condition_name, snr_db, noise_type in conditions:
        rf_features = []
        cnn_spectrograms = []
        condition_silent_count = 0
        print(
            f"Preparing {condition_name}"
            + (f" {snr_db} dB" if snr_db is not None else " reference")
            + f" ({len(test_paths)} held-out clips)")
        for index, path in enumerate(test_paths, start=1):
            waveform = _read_waveform(path)
            if condition_name == "clean":
                rf_waveform = waveform
                cnn_waveform = _cnn_window(waveform)
                if _rms(waveform) == 0:
                    condition_silent_count += 1
            else:
                seed = _stable_seed(path, str(noise_type), int(snr_db))
                rf_waveform, metadata = add_noise(
                    waveform, int(snr_db), str(noise_type), seed,
                    silence_reference_rms)
                cnn_clean = _cnn_window(waveform)
                cnn_waveform, cnn_metadata = add_noise(
                    cnn_clean, int(snr_db), str(noise_type), seed,
                    silence_reference_rms)
                if metadata["used_training_drone_reference_for_silence"]:
                    condition_silent_count += 1
                if metadata["used_training_drone_reference_for_silence"] != (
                    cnn_metadata["used_training_drone_reference_for_silence"]
                ):
                    raise ValueError(
                        f"Silent-reference mismatch between model inputs for {path}")

            rf_features.append(_baseline_features(rf_waveform))
            cnn_spectrograms.append(_mel_spectrogram(cnn_waveform))
            if index % 500 == 0 or index == len(test_paths):
                print(f"  prepared {index}/{len(test_paths)}")

        rf_probabilities = rf_model.predict_proba(
            np.stack(rf_features))[:, 1]
        cnn_probabilities = _probabilities_for_cnn(
            cnn_model, np.stack(cnn_spectrograms), device)
        model_probabilities = {
            "random_forest": (rf_probabilities, rf_threshold),
            "cnn": (cnn_probabilities, cnn_threshold),
        }
        condition_results[condition_name] = {}
        for model_name, (probabilities, threshold) in model_probabilities.items():
            metrics = evaluate(y_test, probabilities, threshold)
            condition_results[condition_name][model_name] = metrics
            result_rows.append({
                "condition": condition_name,
                "snr_db": snr_db,
                "model": model_name,
                **{key: metrics[key] for key in (
                    "threshold", "precision", "recall", "f1", "pr_auc",
                    "roc_auc")},
                "false_positives": int(metrics["confusion_matrix"][0][1]),
                "false_negatives": int(metrics["confusion_matrix"][1][0]),
            })
        if condition_name == "clean":
            silent_count = condition_silent_count

    for row in result_rows:
        clean_metrics = condition_results["clean"][row["model"]]
        row["f1_delta_from_clean"] = row["f1"] - clean_metrics["f1"]
        row["pr_auc_delta_from_clean"] = row["pr_auc"] - clean_metrics["pr_auc"]

    saved_baseline_f1 = baseline_report["test"]["validation_tuned_threshold"]["f1"]
    saved_cnn_f1 = cnn_report["test"]["validation_tuned_threshold"]["f1"]
    reproduced_clean = {
        "random_forest_f1_matches_saved_report": math.isclose(
            condition_results["clean"]["random_forest"]["f1"],
            saved_baseline_f1, abs_tol=1e-12, rel_tol=0.0),
        "cnn_f1_matches_saved_report": math.isclose(
            condition_results["clean"]["cnn"]["f1"],
            saved_cnn_f1, abs_tol=1e-12, rel_tol=0.0),
    }
    if not all(reproduced_clean.values()):
        raise ValueError(
            "Clean test scores did not reproduce saved reports: "
            f"{reproduced_clean}")

    report = {
        "project": "DRONA controlled synthetic audio-noise robustness evaluation",
        "protocol": {
            "dataset": "Existing audio manifest, Tier 1 held-out test groups",
            "test_clips": int(len(test)),
            "test_source_groups": int(test["group_id"].nunique()),
            "test_labels": {
                "background": int((test["label"] == "background").sum()),
                "drone": int((test["label"] == "drone").sum()),
            },
            "models": (
                "Saved CNN checkpoint and Random Forest reconstructed with the "
                "original fixed training configuration; neither model is trained "
                "on noisy test clips"),
            "thresholds": (
                "Fixed validation-selected thresholds from clean training runs; "
                "no threshold tuning on test/noisy conditions"),
            "noise": {
                "types": list(NOISE_TYPES),
                "snr_db": list(SNR_LEVELS_DB),
                "seed_policy": (
                    "Deterministic SHA-256 seed per filename, noise type, and SNR"),
                "silence_count": silent_count,
                "silence_policy": (
                    "Zero-energy clips use the median RMS of non-silent training "
                    "drone clips as the noise reference; these are noise-only "
                    "conditions and do not have a physically defined SNR."),
                "warning": (
                    "White and pink noise are controlled synthetic conditions, "
                    "not recordings of real speech, traffic, wind, or music."),
            },
            "rf_model_reconstruction": {
                "n_estimators": 300,
                "class_weight": "balanced_subsample",
                "random_state": RANDOM_STATE,
                "validation_threshold_matches_saved_report": True,
            },
            "clean_score_reproduction": reproduced_clean,
            "features": (
                "Random Forest receives the complete original-duration waveform, "
                "matching baseline preprocessing. CNN receives its centered "
                "one-second window, zero-padded if short, matching CNN preprocessing."),
        },
        "results": result_rows,
        "full_metrics_by_condition": condition_results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result_rows).to_csv(CSV_REPORT, index=False)
    save_plot(result_rows)
    print("\nNoise robustness results:")
    print(pd.DataFrame(result_rows)[[
        "condition", "snr_db", "model", "precision", "recall", "f1", "pr_auc",
        "f1_delta_from_clean",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nJSON report: {REPORT}")
    print(f"CSV report: {CSV_REPORT}")
    print(f"Plot: {FIGURE}")
    return 0


def save_plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"random_forest": "#3569a8", "cnn": "#d47722"}
    for axis, metric, title in (
        (axes[0], "f1", "Drone F1"),
        (axes[1], "pr_auc", "PR-AUC"),
    ):
        for model_name in ("random_forest", "cnn"):
            subset = [row for row in rows if row["model"] == model_name]
            clean_value = next(
                row[metric] for row in subset if row["condition"] == "clean")
            axis.axhline(
                clean_value,
                color=colors[model_name],
                linestyle="--",
                alpha=0.45,
                label=f"{model_name.replace('_', ' ').title()} clean",
            )
            for noise_type in NOISE_TYPES:
                noisy = sorted(
                    (row for row in subset if row["condition"] == noise_type),
                    key=lambda row: row["snr_db"],
                )
                axis.plot(
                    [row["snr_db"] for row in noisy],
                    [row[metric] for row in noisy],
                    marker="o",
                    color=colors[model_name],
                    linestyle="-" if noise_type == "white" else ":",
                    label=f"{model_name.replace('_', ' ').title()} {noise_type}",
                )
        axis.set_title(title)
        axis.set_xlabel("Input SNR (dB); lower means more noise")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
