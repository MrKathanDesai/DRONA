# DroneRF calibration and unknown-signature experiment

**Research-only, single-session results.** This does not establish independent-session RF detection.

## Binary drone/background probability calibration

| Evaluation | ECE | Brier | NLL | Samples |
|---|---:|---:|---:|---:|
| validation raw | 0.0188 | 0.0108 | 0.0267 | 330 |
| validation temperature scaled | 0.0028 | 0.0055 | 0.0099 | 330 |
| test raw | 0.0166 | 0.0111 | 0.0261 | 320 |
| test temperature scaled | 0.0068 | 0.0054 | 0.0089 | 320 |

Validation-frozen temperature: 0.3030. Threshold selected for validation FPR 1%: 0.0316.

## Held-out Phantom unknown-signature probe

Known training classes are background, Bebop and AR. Phantom is excluded from model fitting and calibration. Unknownness is one minus the largest calibrated probability across known drone classes.

| Unknown ROC-AUC | Known acceptance | Phantom rejection | Known test | Phantom test |
|---:|---:|---:|---:|---:|
| 0.771 | 0.042 | 1.000 | 240 | 30 |

## Limitations

- All recordings are from one lab session; this cannot establish performance on a new receiver, location, or session.
- Phantom is one held-out drone model, not a representative sample of unknown transmitters.
- Temperature scaling and unknownness cutoffs use validation recordings only; test recordings are not used to fit them.
- ECE is top-label expected calibration error with ten equal-width probability bins.
- This artifact is for the offline replay panel and research inspection; it is not connected to a live SDR.
