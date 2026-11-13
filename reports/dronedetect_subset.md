# DroneDetect V2 subset: exploratory model-code classification

**Not a drone-vs-background detection result.** The local subset contains
only two filename prefixes and one interference condition, with no background
recordings. Five switched-on captures are excluded to compare matched modes.

Source: [IEEE DataPort DroneDetect](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine) (DOI `10.21227/5jjj-1m32`).

- Recordings evaluated: 20 of 25 local
- 20 ms windows: 2000
- Features: 64-band summary of per-window, two-sided Welch log-PSD; per-window mean removed
- Evaluation: 5-fold stratified group CV; all windows from each recording stay in one fold
- Coverage: {'interference_codes': ['01'], 'interference': ['bluetooth'], 'model_codes': {'MA1': 10, 'MAV': 10}, 'modes': ['flying', 'hovering'], 'mode_counts_by_code': {'hovering': {'MA1': 5, 'MAV': 5}, 'flying': {'MA1': 5, 'MAV': 5}}, 'excluded_unmatched_mode_recordings': 5, 'complete_windows_per_recording': [100]}

## Recording-level out-of-fold metrics

| Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---:|---:|---:|---:|
| 1.000 | 1.000 | 1.000 | 1.000 |

## Matched-mode recording metrics

| Mode | Recordings | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| hovering | 10 | 1.000 | 1.000 | 1.000 |
| flying | 10 | 1.000 | 1.000 | 1.000 |

## Confusion matrix (recordings)

| Actual \ Predicted | MA1 | MAV |
|---|---:|---:|
| MA1 | 10 | 0 |
| MAV | 0 | 10 |

## Limitations

- Only 20 of 25 local recordings are evaluated: two filename prefixes, Bluetooth-only, and matched hovering/flying modes.
- Five switched-on recordings are excluded because the MAV prefix has no switched-on examples.
- No background/no-drone captures are present, so binary detection metrics are not valid.
- The local subset is too small to support deployment or generalization claims.
- Recording-level folds prevent window leakage, but captures may share the same setup.
- Model codes are reported as filenames; the experiment does not infer a deployed identity.
