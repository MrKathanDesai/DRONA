# Counter-Drone Detection (Audio + RF): Project Summary

**Goal:** detect drones from their sound and their radio signals, and measure honestly how well that works outside the conditions the model was trained on.

## 1. Where things stand

| Module | Status |
|---|---|
| **Audio** | Works well indoors and on unseen drones. Outdoors it is weak: it catches about 22–33% of drone windows and is confused by helicopters. |
| **RF** | Pipeline finished, but the near-perfect scores **cannot be trusted** as detection results (see section 7). |
| **Noise robustness** | Done. Holds at 20 dB SNR; at 0 dB recall drops to 0.40–0.74. Noise augmentation helps in every condition. |
| **Uncertainty / open set** | Done. Four states per window. The review band catches most indoor errors; the unseen-drone flag works (AUROC 0.81), but the "unfamiliar input" flag misses the move outdoors. |
| **Dashboard** | Done. `streamlit run app/dashboard.py`: live-feed replay, states, Grad-CAM. |
| **Fusion** | Not started, and should wait until RF has a second dataset. |

> **Do not claim "99–100% RF detection accuracy".**

## 2. Data used

### Audio

All audio combined: **18,093 clips**.

| Dataset | Use | Contents |
|---|---|---|
| **DroneAudioDataset** | Training (indoor) | Parrot Bebop and Mambo. 1,332 drone clips and about 10,370 background clips, 1 s each. The drone clips come from only 312 real recordings, because each 5 s recording was cut into 1 s slices. |
| **ESC-50 metadata** | Labels | Categories for the background clips, including helicopter, chainsaw, vacuum, washing machine, engine and aeroplane. |
| **Svanström** | Outdoor test only | 90 10-second recordings: 30 drone, 30 helicopter, 30 background. Cut into 900 1 s windows. Never used for training or tuning. |
| **ITU ARIS Lab** | Training (outdoor) | 4,491 drone and 1,000 background 1 s clips. No documented helicopters. |

### RF

**DroneRF:** Bebop, AR and Phantom drones plus background, from one lab session.

- 227 segments of 0.25 s each: 84 Bebop, 81 AR, 21 Phantom, 41 background
- Two receivers cover the lower and upper halves of the 2.4 GHz band
- 23 archives, 4 GB, checksums verified

## 3. Processing

### Audio

- Resample to 16 kHz mono and use fixed 1 s clips. Fixed length matters: clip duration alone gave the label away (AUC 1.0) in the raw data.
- **Baseline input:** 128-band log-mel spectrogram.
- **Pretrained model input:** RMS-normalised clip, 64-band log-mel spectrogram.

### RF

- Each segment is split into 10 windows of 25 ms.
- Each window becomes a log power spectrum (Welch, 2048-point FFT), giving 1025 values per band. Result: 2,270 windows (1,860 drone, 410 background).
- Each band's average level is removed, so loudness cannot act as a shortcut.
- Classical models then average the spectrum into 64 sub-bands per band (128 numbers per window).

## 4. Models and training

### Audio baseline

- **Input:** 3 summary statistics per mel band (384 numbers), level removed.
- **Models:** logistic regression (C=0.1) and gradient boosting.

### Audio main model: frozen encoder + small trained head

```text
1 s clip → log-mel spectrogram → MobileNetV1 (frozen) → 1024 numbers → logistic regression → drone score
```

- **Encoder:** PANNs MobileNetV1, a 4.2M-parameter CNN already trained on AudioSet. It is frozen, and we do not train it.
- **Head:** logistic regression (C=0.01, balanced classes), the only part trained. About 1,025 weights. Inputs are standardised using training data only.
- **Why frozen:** near-duplicate clips would let a 4.2M-parameter network memorise recordings. An MLP head was also tried and was no better.
- **Tier 4 version:** weights are balanced by dataset and class, because ITU ARIS is 82% drone and the indoor data 88% background. Otherwise "sounds outdoors" would become a drone cue.

### RF models

| Model | Details |
|---|---|
| **Logistic regression** | 128 inputs, C=0.1 |
| **Random forest** (saved model) | 300 trees, minimum 2 samples per leaf, balanced classes |
| **1D-CNN** | About 12k parameters. 4 conv layers (8, 16, 32, 32 filters, kernel 7, stride 2), global pooling, dropout 0.3. 40 epochs, batch 64, Adam, learning rate 0.001. |
| **Forest + level** | Keeps absolute level. Used only to probe for shortcuts. |

## 5. How it was evaluated

### Leakage control

Slices of one recording never cross splits. Audio is grouped by recording. RF uses time-ordered blocks of 3 consecutive segments. A validator stops the pipeline on any leak.

### Test levels (tiers)

| Tier | Test |
|---|---|
| **Tier 1** | Held-out recordings, same dataset |
| **Tier 2** | Drone model never trained on. Audio: train Bebop, test Mambo. RF: each drone held out in turn. |
| **Tier 3** | Audio: train indoor, test outdoor Svanström |
| **Tier 4** | Audio: tier 3 test, with ITU ARIS added to training |

RF has no tier 3 or 4 because it has only one lab session.

### Metrics

| Metric | Meaning |
|---|---|
| **ROC-AUC** | Chance a drone window scores above a non-drone window. Random = 0.5. |
| **PR-AUC** | Precision across all recall levels. Random = share of drones in the test set (0.333 on tiers 3 and 4, not 0). |
| **Recall** | Share of drone windows caught. |
| **FPR** | Share of non-drone windows that raise an alarm. |

- The threshold is set at 1% FPR on validation data, then frozen for test.
- Confidence intervals resample whole recordings.
- A recording raises an alarm only on 2 consecutive positive windows.

**Metrics not used:**

- **Accuracy:** always saying "no drone" scores 0.667 on tiers 3 and 4.
- **Precision and F1:** these change with the test set's drone share, which is far higher here than in real life.
- **RMSE:** this is for predicting numbers, not classes.

## 6. Audio results

Format: PR-AUC / recall at the frozen threshold / false alarms.

| Tier | Baseline (gradient boosting) | Pretrained head |
|---|---|---|
| **Tier 1** | 0.981 / 0.935 / 3 of 1759 | 0.968 / 0.878 / 10 of 1759 |
| **Tier 2** | 0.891 / 0.393 / 14 of 2113 | 0.929 / 0.647 / 22 of 2113 |
| **Tier 3 (outdoor)** | 0.702 / 0.197 / 7 of 600 | 0.649 / 0.330 / 19 of 600 |

Tier 3 baseline logistic regression: 0.735 / 0.073 / 0 of 600.

### Tier 4 (ITU ARIS added, tested on Svanström)

| Model | PR-AUC | Recall | False alarms |
|---|---|---|---|
| **Pretrained, balanced (planned)** | 0.640 [0.503–0.755] | 0.220 | 6 of 600 |
| Pretrained, pooled | 0.659 | 0.240 | 5 of 600 |
| Baseline, balanced | 0.619 | 0.047 | 0 of 600 |

### Alarms per recording

| Category | Tier 3 | Tier 4 |
|---|---|---|
| Drone | 15 of 30 | 14 of 30 |
| Helicopter | 2 of 30 | 1 of 30 |
| Outdoor background | 0 of 30 | 0 of 30 |

### What this shows

- Indoor scores overstate real performance. Moving outdoors is the hard step.
- The pretrained head handles unseen drones better than the baseline, but does not beat it on outdoor PR-AUC.
- Outdoor training data did not improve PR-AUC (0.640 vs 0.649; the intervals overlap). False alarms did land on target (1.0% vs 3.2%), but recall fell.
- The model learns ITU ARIS well (validation PR-AUC 0.995). It fails to carry that over to Svanström.
- **Main weakness:** the pretrained model's drone vs helicopter ROC-AUC is only about 0.58 (baseline about 0.74). Trained on ITU ARIS alone, helicopters score almost like drones (0.90 vs 0.99). Outdoor helicopter and aircraft negatives are the likely next fix.

## 7. RF results

Format: PR-AUC / recall at the frozen threshold. False alarms: 0–1 of 50 on tier 1, 0 of 80 on tier 2.

| Model | Held-out time | Unseen Bebop | Unseen AR | Unseen Phantom |
|---|---|---|---|---|
| **Logistic regression** | 0.998 / 0.985 | 0.999 / 0.977 | 1.000 / 0.994 | 0.580 / 0.076 |
| **Random forest** | 1.000 / 1.000 | 1.000 / 0.988 | 1.000 / 0.991 | 1.000 / 0.976 |
| **1D-CNN** | 1.000 / 0.978 | 1.000 / 0.926 | 1.000 / 0.967 | 1.000 / 0.990 |

Other results (forest, tier 1):

- **Per segment:** 27 of 27 drone segments alarmed, 0 of 5 background.
- **Identifying the drone type:** 292 of 320 test windows correct, mostly confusing Bebop with AR.

### Why these scores cannot be trusted

Windows with no visible transmission were tested on their own (below 15 dB in both bands, 344 train and 56 test). **The forest still separated drone from background at AUC 1.000.**

A model cannot find a drone in a window with no visible drone signal, so it is most likely recognising the recording session or receiver: background was recorded separately. A very weak drone signal under the cut-off is not ruled out.

- Removing signal level did not fix this.
- DroneRF alone cannot fix it, because its background is one session.
- The results do not show that the CNN beats the classical models.
- Level alone gives AUC 0.908 (low band) and 0.504 (high band).

## 7b. Noise robustness (tier 1)

Tier-1 test clips mixed with noise from test-split ESC-50 recordings at 20 / 10 / 0 dB SNR. The threshold stays at its clean-validation value. Recall / false-alarm rate:

| Condition at 0 dB | Pretrained | Pretrained + noise augmentation | Baseline |
|---|---|---|---|
| Clean (reference) | 0.878 / 0.006 | 0.882 / 0.009 | 0.833 / 0.003 |
| Voice | 0.396 / 0.011 | 0.498 / 0.011 | 0.698 / 0.003 |
| Traffic | 0.502 / 0.019 | 0.547 / 0.020 | 0.469 / 0.005 |
| Wind | 0.743 / 0.006 | 0.796 / 0.009 | 0.657 / 0.003 |
| Environment | 0.412 / 0.011 | 0.506 / 0.014 | 0.527 / 0.005 |
| Music (synthetic) | 0.522 / 0.087 | 0.641 / 0.042 | 0.567 / 0.010 |
| Mixed | 0.478 / 0.023 | 0.596 / 0.024 | 0.392 / 0.002 |

- At 20 dB recall changes by at most 0.016.
- Augmentation (one noisy copy of each train clip, settings fixed in advance) raises 0 dB recall in every condition, with clean recall unchanged.
- Loud music is the only noise that causes many false alarms.
- ESC-50 has no speech or music: "voice" is laughing / coughing / crying / sneezing, and "music" is synthetic chords + drum.

## 7c. Uncertainty and open set

Each window is **drone**, **no drone**, **unknown drone** or **low confidence**. All cut-offs come from validation and were fixed before testing.

| Tier | Error rate | Flagged low confidence | Error rate on the rest | Share of errors flagged |
|---|---|---|---|---|
| Tier 1 | 0.020 | 12.7% | 0.005 | 80% |
| Tier 3 | 0.244 | 41.8% | 0.218 | 48% |
| Tier 4 | 0.267 | 11.6% | 0.245 | 19% |

- The distance-to-threshold (margin) is the useful signal. Ensemble spread and distance-from-training add nothing.
- **Open set:** trained on Bebop, the distance to known drones separates Mambo from held-out Bebop at AUROC 0.81 (35.6% vs 15.5% flagged).
- **Gap:** only 1.7% of outdoor windows are flagged "unlike training data", so that flag does not detect the move outdoors.

## 8. Known limits

- The pretrained head's C and embedding layer were chosen from a small grid while looking at all three tiers, including tier 3. Those rows are mildly optimistic.
- Tier 4 reused those settings and was not tuned on Svanström, but tier 3 test data had already been seen.
- The test sets are 12–33% drone, far more than real life. At 1% drone windows, tier 4 precision would fall from 0.917 to about 0.18.
- `scripts/detect.py` and the dashboard use the tier 4 detector; outdoor recall is still only about 22% of windows.
- Probabilities are not calibrated (no temperature scaling / ECE); the flags rely on ranks only.
- Noise mixing is additive at fixed SNR: no reverberation, distance or microphone effects.
- `data/models/rf_forest.joblib` is not a usable real-world detector.
- Steps 2a–3b are committed; the RF, noise, uncertainty and dashboard work is not committed yet.

## 9. What is left

1. Add a second RF dataset with different sessions (DroneDetect or CardRF; both need a free IEEE DataPort login), then repeat the quiet-window check.
2. Add outdoor helicopter and aircraft negatives to audio training.
3. Build audio + RF fusion once step 1 gives trustworthy RF scores.
4. Optional: probability calibration (ECE), an audio drone-model confusion matrix, and a statement in the report on the CNN / CNN+LSTM / transformer comparison, which was replaced by the frozen-encoder approach.

## 10. Result files

| Path | Contents |
|---|---|
| `reports/baseline_audio.md` | Audio baseline |
| `reports/pretrained_audio.md` | Pretrained model, tiers 1–3 |
| `reports/outdoor_audio.md` | Tier 4 |
| `reports/rf_detection.md` | RF results and warning |
| `reports/noise_robustness.md` | Noise stress test and augmentation |
| `reports/uncertainty_audio.md` | Uncertainty flags and open set |
| `app/dashboard.py` | Streamlit monitor |
| `reports/figures/` | Charts |
| `data/models/` | Saved models |

```bash
python scripts/train_baseline.py
python scripts/train_pretrained.py
python scripts/train_outdoor.py
python scripts/train_rf.py
python scripts/detect.py data/raw_svanstrom/Audio/DRONE_001.wav
```
