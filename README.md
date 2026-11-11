# dronedet

Acoustic drone detection — a dataset-only project. No hardware required.

## Status

| Step | What | State |
|---|---|---|
| 1 | Data foundation: manifest, splits, leakage tests | **done** |
| 2 | Features + classical baseline | **done** — see `reports/baseline_audio.md` |
| 2b | Tier 3: Svanström outdoor audio into the manifest | **done** |
| 3 | Pretrained detector (frozen AudioSet CNN + small head), tiered evaluation, `detect.py` demo | **done** — see `reports/pretrained_audio.md` |
| 4 | Augmentation (outdoor noise, music masking), SNR curves | next |
| 5 | Calibration (temperature scaling, ECE) | |
| 6 | Open-set head | |
| 7 | Tracking / fusion layer (classical AI half) | |

## Setup

macOS ships a Homebrew Python that refuses system-wide installs (PEP 668),
so the project uses a virtual environment. Do this once:

```bash
cd /Users/kathandesai/drone
python3 -m venv .venv
source .venv/bin/activate          # every new terminal needs this line
pip install -r requirements.txt
```

You know it worked when your prompt shows `(.venv)`. Then:

```bash
python scripts/build_manifest.py        # builds data/manifest.parquet
python scripts/make_features.py         # caches log-mel features (~10 s)
python scripts/train_baseline.py        # baseline + report (~2 min)
python scripts/make_embeddings.py       # frozen encoder, one pass (~30 s)
python scripts/train_pretrained.py      # trains only the head (~15 s)
python scripts/detect.py data/raw_svanstrom/Audio/DRONE_001.wav
python -m pytest tests -q               # 36 tests
```

In VS Code, press ⇧⌘P → "Python: Select Interpreter" → pick the one inside
`.venv`, so the editor and the terminal agree.

The raw corpora and the pretrained checkpoint are not committed. Fetch
everything (~985 MB) with `python scripts/download_data.py`; see
`data/README.md` for provenance.

## What the data is

11,704 one-second clips at 16 kHz, from Al-Emadi et al. (IWCMC 2019).

- **1,332 drone clips** — Parrot Bebop and Mambo, recorded indoors, from 312 parent recordings
- **10,372 background clips** — all 50 ESC-50 environmental classes at 200 each, plus 372 synthetic (silence, white/pink noise, taps)
- **Imbalance 7.79 : 1** against the positive class, so accuracy is a meaningless metric here
- **Hard confusers included**: 200 each of helicopter, chainsaw, vacuum cleaner, washing machine, engine, airplane

## The three decisions that make results trustworthy

**Splits are by parent recording, never by clip.** The dataset cuts every
5-second recording into five 1-second slices, and those slices are
near-duplicates. A random clip split puts the same flight in train and test;
the model memorises and the score lies. `group_id` is the leakage unit, and
the negatives get the same treatment — ESC-50 clips were sliced the same way,
so all segments of one source clip share a group.

Split assignment is a hash of `group_id`, so it is identical on every machine
and across every run, with no seeds to get wrong.

**Labels come from metadata, not filename arithmetic.** The background
filenames keep ESC-50's naming with a segment digit appended, so you *can*
recover the class as `last_field // 10` — but that disagrees with ESC-50's own
metadata on about 20 of 10,000 files. We join on `(src_file, take)` against
`esc50.csv` instead, and a test asserts zero mismatches.

**Clips are never zero-padded.** Clip duration identifies the class almost
perfectly here: drone clips are 1.024 s, background clips 1.0217 / 0.975 /
0.960 s and so on, and every distinct duration belongs to one class. Pad
short clips with silence and the length of the silent tail gives away the
label. `fix_length` centre-crops long clips and loops short ones, and a test
confirms it: the same noise at two lengths gives the same features. That
test was checked to fail under zero-padding.

How much this matters: a classifier given *only* clip duration separates
the classes perfectly (separability 1.0), and one given only overall
loudness reaches 0.79. Neither enters the features — the baseline's are
level-invariant by construction, and a test checks it. The loop splice was
checked too: cutting full-length drone test clips to background lengths
and looping them moves the median score by 0.0000.

## Baseline results (step 2)

Level-invariant log-mel summary statistics with gradient boosting; the
threshold is fixed at 1% false-alarm rate on validation, then applied to
test. CIs resample whole recordings.

| split | PR-AUC [95% CI] | recall | test false alarms |
|---|---|---|---|
| tier 1 (held-out recordings) | 0.981 [0.954, 0.998] | 0.935 | 3 / 1759 |
| tier 2 (train Bebop, test Mambo) | 0.891 [0.845, 0.927] | 0.393 | 14 / 2113 |

Tier 1 looks nearly solved; tier 2 shows why that is not the whole story.
At the same operating point, recall on an unseen drone model collapses from
94% to 39%: the baseline has partly learned *Bebop*, not *drone*. Tier 1
also has zero helicopter / engine / airplane false alarms, which more
likely reflects indoor-vs-ESC-50 recording differences than real
discrimination. Tier 3 (outdoor, different hardware) is the test that
decides.

## Pretrained detector (step 3)

No model is trained from scratch. A frozen PANNs MobileNetV1 (AudioSet,
4.2 M parameters, 24 MB) turns each loudness-normalised 1 s clip into a
1024-d embedding, and a logistic-regression head is fitted on top in
seconds. Same tiers, same 1%-FPR operating point as the baseline:

| split | baseline (best) PR-AUC / recall | pretrained PR-AUC / recall |
|---|---|---|
| tier 1 (held-out recordings) | 0.981 / 0.935 | 0.968 / 0.878 |
| tier 2 (unseen drone model) | 0.891 / 0.393 | **0.929 / 0.647** |
| tier 3 (outdoor corpus, 1 s windows) | 0.735 / 0.073 | 0.649 / 0.330 |

The pretrained features are what carry over to a drone the head never
heard: recall on Mambo goes from 39% to 65%. Outdoors both models are weak;
the baseline's threshold does not transfer at all (7% recall), the
pretrained one keeps 33% at a 3% false-alarm rate. Per 10 s recording, with
`detect.py`'s rule (two consecutive positive windows), it detects 15 / 30
outdoor drones with 2 / 30 helicopter and 0 / 30 background false alarms,
having never heard outdoor audio. Closing that gap is what augmentation
(step 4) is for. The embedding layer and C came from a small grid, so the
pretrained rows are mildly optimistic.

## Splits

- **tier 1** — held-out recordings, same corpus. The everyday split (70/15/15 by group).
- **tier 2** — held-out drone *model*: train on Bebop, test on Mambo. Answers "does it detect drones, or only the drones it saw?"
- **tier 3** — held-out *corpus*: train on all indoor recordings, test on the 90 outdoor Svanström recordings (drone / helicopter / background), cut into 1 s windows grouped by recording. The threshold comes from indoor validation; the outdoor set is never used for training or tuning.

## Layout

```
src/dronedet/
  contracts.py          shared types
  paths.py              data root (override with DRONEDET_DATA)
  data/manifest.py      build + validate the manifest
  features/audio.py     load, fix length, log-mel
  models/baseline.py    level-invariant summary features + sklearn models
  models/pretrained.py  frozen PANNs MobileNetV1 encoder (no torchlibrosa)
  eval/metrics.py       threshold at FPR, grouped bootstrap, per-category FA
  eval/protocol.py      tier 1 / 2 / 3 protocol shared by every model
scripts/download_data.py
scripts/build_manifest.py
scripts/make_features.py
scripts/train_baseline.py
scripts/make_embeddings.py
scripts/train_pretrained.py
scripts/detect.py             run the detector on any audio file
reports/baseline_audio.{md,json}
reports/pretrained_audio.{md,json}
tests/test_manifest.py        13 tests, incl. an injected-leak check
tests/test_audio_features.py  12 tests, incl. cache-reproduces-from-code
tests/test_baseline.py        5 tests, incl. level invariance
tests/test_pretrained.py      6 tests, incl. level invariance of embeddings
data/                   gitignored — raw corpora, checkpoint, manifest, caches
```

`tests/test_manifest.py::test_validate_catches_injected_leak` deliberately
corrupts a split and asserts the validator raises. A guard nobody has watched
fail is not a guard.
