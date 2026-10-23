# dronedet

Acoustic drone detection — a dataset-only project. No hardware required.

## Status

| Step | What | State |
|---|---|---|
| 1 | Data foundation: manifest, splits, leakage tests | **done** |
| 2 | Features + classical baseline | **done** — see `reports/baseline_audio.md` |
| 2b | Tier 3: Svanström outdoor audio into the manifest | next |
| 3 | CNN detector, tiered evaluation | |
| 4 | Augmentation, SNR + distance curves | |
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
python -m pytest tests -q               # 27 tests
```

In VS Code, press ⇧⌘P → "Python: Select Interpreter" → pick the one inside
`.venv`, so the editor and the terminal agree.

The raw corpus is not committed. Fetch it with:

```bash
cd data
git clone --depth 1 https://github.com/saraalemadi/DroneAudioDataset.git raw_droneaudio
rm -rf raw_droneaudio/.git
curl -o esc50_meta.csv https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv
cd ..
```

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

## Splits

- **tier 1** — held-out recordings, same corpus. The everyday split (70/15/15 by group).
- **tier 2** — held-out drone *model*: train on Bebop, test on Mambo. Answers "does it detect drones, or only the drones it saw?"
- **tier 3** — held-out *corpus*. Column exists; needs a second dataset to fill.

## Layout

```
src/dronedet/
  contracts.py          shared types
  paths.py              data root (override with DRONEDET_DATA)
  data/manifest.py      build + validate the manifest
  features/audio.py     load, fix length, log-mel
  models/baseline.py    level-invariant summary features + sklearn models
  eval/metrics.py       threshold at FPR, grouped bootstrap, per-category FA
scripts/build_manifest.py
scripts/make_features.py
scripts/train_baseline.py
reports/baseline_audio.{md,json}
tests/test_manifest.py        11 tests, incl. an injected-leak check
tests/test_audio_features.py  11 tests, incl. cache-reproduces-from-code
tests/test_baseline.py        5 tests, incl. level invariance
data/                   gitignored — raw corpus, manifest, feature cache
```

`tests/test_manifest.py::test_validate_catches_injected_leak` deliberately
corrupts a split and asserts the validator raises. A guard nobody has watched
fail is not a guard.
