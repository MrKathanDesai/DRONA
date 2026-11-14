# Audio architecture comparison

All four models use the same cached log-mel inputs pooled to 64 bands and 21 frames, training-only per-band normalization, group-aware splits, and a threshold fixed on validation negatives at 1% FPR. Best checkpoint is selected by validation PR-AUC. No test split is used for model selection.

## Results

| Evaluation | Architecture | PR-AUC [95% CI] | ROC-AUC | Recall | FPR | False alarms / negatives | Parameters | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tier1_test | small_cnn | 0.976 [0.952, 0.992] | 0.996 | 0.898 | 0.005 | 8/1759 | 23,585 | 8 |
| tier3_outdoor | small_cnn | 0.814 [0.692, 0.911] | 0.835 | 0.283 | 0.000 | 0/600 | 23,585 | 8 |
| tier1_test | cnn_gru | 0.985 [0.970, 0.996] | 0.997 | 0.951 | 0.006 | 11/1759 | 56,129 | 6 |
| tier3_outdoor | cnn_gru | 0.616 [0.449, 0.768] | 0.674 | 0.120 | 0.012 | 7/600 | 56,129 | 6 |
| tier1_test | cnn_lstm | 0.980 [0.960, 0.994] | 0.996 | 0.931 | 0.007 | 12/1759 | 62,401 | 4 |
| tier3_outdoor | cnn_lstm | 0.647 [0.488, 0.776] | 0.684 | 0.140 | 0.003 | 2/600 | 62,401 | 4 |
| tier1_test | lightweight_transformer | 0.984 [0.968, 0.995] | 0.993 | 0.935 | 0.007 | 12/1759 | 72,641 | 7 |
| tier3_outdoor | lightweight_transformer | 0.607 [0.440, 0.751] | 0.724 | 0.120 | 0.003 | 2/600 | 72,641 | 7 |

## Protocol and caveats

- **training:** All models are trained on the same indoor Tier 1 train recordings; Tier 1 validation selects the checkpoint and threshold.
- **tier1_test:** Held-out indoor recordings; groups do not overlap train or validation.
- **tier3_outdoor:** The same frozen models and indoor-validation thresholds are applied to Svanström. This corpus was used in earlier project evaluation and is not a blind external validation.
- **threshold:** Each model selects a threshold on validation negatives at 1% FPR; the threshold is frozen for test.
- **selection:** Best checkpoint selected by validation PR-AUC with early stopping.
- This is an experimental architecture comparison; no model is promoted to deployment.
- Tier 3 uses Svanström, which was used in earlier project evaluation, so it is not an untouched independent corpus.
- Architecture results depend on this feature representation and training protocol; the existing AudioCNN used a different 64-bin feature extraction setup.
- Tier 1 estimates same-corpus generalization; the outdoor tier is the key transfer check.

## Confusion matrices

### small_cnn

- **tier1_test:** rows actual (background, drone), columns predicted (background, drone): `[[1751, 8], [25, 220]]`.
- **tier3_outdoor:** rows actual (background, drone), columns predicted (background, drone): `[[600, 0], [215, 85]]`.
### cnn_gru

- **tier1_test:** rows actual (background, drone), columns predicted (background, drone): `[[1748, 11], [12, 233]]`.
- **tier3_outdoor:** rows actual (background, drone), columns predicted (background, drone): `[[593, 7], [264, 36]]`.
### cnn_lstm

- **tier1_test:** rows actual (background, drone), columns predicted (background, drone): `[[1747, 12], [17, 228]]`.
- **tier3_outdoor:** rows actual (background, drone), columns predicted (background, drone): `[[598, 2], [258, 42]]`.
### lightweight_transformer

- **tier1_test:** rows actual (background, drone), columns predicted (background, drone): `[[1747, 12], [16, 229]]`.
- **tier3_outdoor:** rows actual (background, drone), columns predicted (background, drone): `[[598, 2], [264, 36]]`.
