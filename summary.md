# Counter-Drone Detection Using Audio and RF Signals

## Project summary

**Objective:** detect drones from their acoustic and radio-frequency signatures, and evaluate how well detection holds up beyond the conditions seen in training.

| Module | Status | Main outcome |
|---|---|---|
| **Audio** | Implemented and evaluated | Strong indoors and on an unseen drone model; outdoor detection remains limited, mainly by helicopter confusion |
| **RF** | Implemented and evaluated | Scores of about 1.000, but our validation check shows they cannot yet be accepted as true detection performance |
| **Noise robustness** | Implemented and evaluated | Unaffected at 20 dB SNR; recall falls to 0.40–0.74 at 0 dB; noise augmentation helps in every condition |
| **Uncertainty and open set** | Implemented and evaluated | Low-confidence band holds 80% of indoor errors in 13% of windows; unseen drone model flagged at AUROC 0.81 |
| **Dashboard** | Implemented | Streamlit replay of recordings with per-window state and Grad-CAM |
| **Audio + RF fusion** | Planned | Depends on validating RF on a second dataset |

## Key findings

1. **Evaluation conditions matter more than the headline score.** The same audio model drops from PR-AUC 0.968 on held-out indoor recordings to 0.649 outdoors.
2. **Pretrained features generalise better to new drones.** On an unseen drone model, recall rose from 0.393 to 0.647 compared with the baseline.
3. **Outdoor training data alone was not enough.** It reduced false alarms to the 1% target, but PR-AUC did not improve and recall fell.
4. **Near-perfect RF scores can come from th recording setup.** Our quiet-window test shows the RF models may be separating recording sessions, not drone signals.

## Data

| Dataset | Type | Role | Contents |
|---|---|---|---|
| **DroneAudioDataset** + ESC-50 labels | Audio | Training | Indoor Bebop and Mambo drones; 1,332 drone and about 10,370 background clips |
| **ITU ARIS Lab** | Audio | Training | Outdoor; 4,491 drone and 1,000 background clips |
| **Svanström multi-sensor** | Audio | Test only | Outdoor; 30 drone, 30 helicopter and 30 background recordings |
| **DroneRF** | RF | Training and test | One lab session; 227 segments covering Bebop, AR, Phantom and background |

## Method

### Audio

```text
1 s clip → log-mel spectrogram → MobileNetV1 (frozen) → 1024-d embedding → logistic regression → drone score
```

- **Encoder:** PANNs MobileNetV1, pretrained on AudioSet (4.2M parameters) and kept frozen.
- **Classifier:** logistic regression trained on the embeddings. Only this head is trained, because near-duplicate clips make a full network prone to memorising recordings.
- **Baseline:** level-invariant log-mel statistics with logistic regression and gradient boosting.
- **Outdoor training:** sample weights balance each dataset and class, so that outdoor sound does not become a proxy for "drone".

### RF

```text
25 ms window → log power spectrum (two bands) → level removed → classifier → drone score
```

We compared logistic regression, a random forest (300 trees) and a small 1D-CNN (about 12k parameters).

## Evaluation design

| Tier | What it tests | Audio | RF |
|---|---|---|---|
| **1** | Held-out data, same dataset | Held-out recordings | Held-out time blocks |
| **2** | Unseen drone model | Train Bebop, test Mambo | Each drone held out in turn |
| **3** | New environment | Train indoor, test outdoor | Not possible with one session |
| **4** | Outdoor training data added | Same outdoor test as tier 3 | Not possible with one session |

- **Leakage control:** slices of one recording never cross splits, and an automated validator enforces this.
- **Primary metrics:** PR-AUC and ROC-AUC measure ranking quality. Recall and false-positive rate are reported at a threshold fixed at 1% FPR on validation data, then applied unchanged to test.
- **Why not accuracy or F1:** accuracy is misleading here, since always predicting "no drone" scores 0.667 on tiers 3 and 4. Precision and F1 depend on the test set's drone share, which is well above real-world rates.
- **Uncertainty:** 95% confidence intervals resample whole recordings.

## Audio results

Each cell shows PR-AUC / recall / false alarms.

| Tier | Baseline (gradient boosting) | Pretrained model |
|---|---|---|
| **1: Held-out recordings** | 0.981 / 0.935 / 3 of 1,759 | 0.968 / 0.878 / 10 of 1,759 |
| **2: Unseen drone model** | 0.891 / 0.393 / 14 of 2,113 | 0.929 / 0.647 / 22 of 2,113 |
| **3: Outdoor test** | 0.702 / 0.197 / 7 of 600 | 0.649 / 0.330 / 19 of 600 |
| **4: Outdoor data added** | 0.619 / 0.047 / 0 of 600* | 0.640 / 0.220 / 6 of 600 |

\* Tier 4 baseline is logistic regression; its tier 3 equivalent scored 0.735 / 0.073 / 0 of 600.

**Per-recording alarms (2 consecutive positive windows, pretrained model):**

| Category | Tier 3 | Tier 4 |
|---|---|---|
| Drone | 15 of 30 | 14 of 30 |
| Helicopter | 2 of 30 | 1 of 30 |
| Outdoor background | 0 of 30 | 0 of 30 |

The pretrained model separates drones from helicopters poorly (ROC-AUC about 0.58, against about 0.74 for the baseline). Outdoor helicopter and aircraft examples are the likely next improvement.

## RF results

Each cell shows PR-AUC / recall.

| Model | Held-out time | Unseen Bebop | Unseen AR | Unseen Phantom |
|---|---|---|---|---|
| **Logistic regression** | 0.998 / 0.985 | 0.999 / 0.977 | 1.000 / 0.994 | 0.580 / 0.076 |
| **Random forest** | 1.000 / 1.000 | 1.000 / 0.988 | 1.000 / 0.991 | 1.000 / 0.976 |
| **1D-CNN** | 1.000 / 0.978 | 1.000 / 0.926 | 1.000 / 0.967 | 1.000 / 0.990 |

### Validation check: quiet-window test

Scores this high on unseen drones prompted a further check. We trained and tested using only windows with no visible transmission, and **the random forest still separated drone from background at AUC 1.000**.

This points to the recording session or receiver, since DroneRF's background was recorded separately from the drones. A very weak drone signal below the cut-off is not ruled out. The RF results above therefore show that the pipeline works, but not that it would detect drones in the field, and they do not establish that one model is better than another.

## Noise, uncertainty and the dashboard

- **Noise:** held-out indoor clips mixed with voice, traffic, wind, environment, synthetic music and a mix of all five, at 20 / 10 / 0 dB, with the threshold unchanged. Noise augmentation of the head raised 0 dB recall by 0.04–0.12 in every condition. Loud music raised false alarms to 8.7% (4.2% with augmentation). Details: `reports/noise_robustness.md`.
- **Uncertainty:** each window is drone, no drone, unknown drone or low confidence, with every cut-off fixed on validation. The flag that marks inputs unlike the training data caught only 1.7% of outdoor windows, so it does not detect the indoor-to-outdoor shift. Details: `reports/uncertainty_audio.md`.
- **Dashboard:** `streamlit run app/dashboard.py`.

## Limitations

- The pretrained model's regularisation strength and embedding layer were selected while viewing all three tiers, so its results are mildly optimistic. Tier 4 reused these settings without further tuning.
- Test sets are 12–33% drone, far above real-world rates, so real-world precision would be lower.
- DroneRF has a single background session, so it cannot resolve the quiet-window result by itself.
- RF test sets contain only 50–80 background windows, so false-positive rates are coarse.

## Next steps

1. Validate RF on a second dataset with different recording sessions (DroneDetect or CardRF), and repeat the quiet-window test.
2. Add outdoor helicopter and aircraft examples to audio training.
3. Build audio and RF fusion once RF performance is validated.
4. Calibrate probabilities (ECE) and explain in the report why the frozen encoder replaced the planned CNN / CNN+LSTM / transformer comparison.
