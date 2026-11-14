# Outdoor confuser training experiment

This separate experiment adds Svanström outdoor helicopter and ambient background negatives to training while keeping its drone recordings test-only. It does not replace the deployed tier-4 detector.

Thresholds are calibrated to 1% FPR on validation negatives. Both variants use the same Svanström validation negatives for threshold selection and the same recording-held-out test recordings. The test set contains 30 drone recordings and 6 per negative category (71.4% drone windows), so PR-AUC is not a deployment-prevalence estimate.

| Model | Training includes Svanström negatives | PR-AUC [95% CI] | ROC-AUC | Recall | FPR | Test recordings |
|---|---:|---:|---:|---:|---:|---:|
| pretrained_without_svanstrom_negatives | no | 0.867 [0.767, 0.943] | 0.695 | 0.250 | 0.017 | 42 |
| pretrained_with_svanstrom_negatives | yes | 0.948 [0.901, 0.978] | 0.864 | 0.213 | 0.000 | 42 |
| baseline_without_svanstrom_negatives | no | 0.859 [0.740, 0.946] | 0.720 | 0.063 | 0.000 | 42 |
| baseline_with_svanstrom_negatives | yes | 0.884 [0.781, 0.955] | 0.757 | 0.040 | 0.000 | 42 |

## False alarms and recording-level results

### pretrained_without_svanstrom_negatives

| Category | Window FPR | False alarms / windows | Recording alarm rate | Alarms / recordings |
|---|---:|---:|---:|---:|
| drone | — | — | 0.500 | 15/30 |
| helicopter | 0.033 | 2/60 | 0.000 | 0/6 |
| outdoor_background | 0.000 | 0/60 | 0.000 | 0/6 |

### pretrained_with_svanstrom_negatives

| Category | Window FPR | False alarms / windows | Recording alarm rate | Alarms / recordings |
|---|---:|---:|---:|---:|
| drone | — | — | 0.467 | 14/30 |
| helicopter | 0.000 | 0/60 | 0.000 | 0/6 |
| outdoor_background | 0.000 | 0/60 | 0.000 | 0/6 |

### baseline_without_svanstrom_negatives

| Category | Window FPR | False alarms / windows | Recording alarm rate | Alarms / recordings |
|---|---:|---:|---:|---:|
| drone | — | — | 0.100 | 3/30 |
| helicopter | 0.000 | 0/60 | 0.000 | 0/6 |
| outdoor_background | 0.000 | 0/60 | 0.000 | 0/6 |

### baseline_with_svanstrom_negatives

| Category | Window FPR | False alarms / windows | Recording alarm rate | Alarms / recordings |
|---|---:|---:|---:|---:|
| drone | — | — | 0.033 | 1/30 |
| helicopter | 0.000 | 0/60 | 0.000 | 0/6 |
| outdoor_background | 0.000 | 0/60 | 0.000 | 0/6 |

## Data gaps and limitations

The integrated audio manifest has no explicitly aircraft-labeled outdoor recordings. Existing airplane negatives remain in the training data; outdoor aircraft-specific training is still needed.

- This is a recording-held-out split within Svanström, not an independent-corpus test.
- Svanström was used in earlier project evaluation; this split is not a blind external validation.
- Helicopter and outdoor-background negatives are added; the integrated manifest has no explicitly aircraft-labeled outdoor examples.
- This experimental run does not replace or retrain the deployed tier-4 detector.
