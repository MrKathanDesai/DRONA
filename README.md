# dronedet

Acoustic drone detection — a dataset-only project. No hardware required.

## Status

| Step | What | State |
|---|---|---|
| 1 | Data foundation: manifest, splits, leakage tests | **done** |
| 2 | Log-mel features + classical audio baseline | **done** |
| 3 | CNN detector, tiered evaluation | **done** |
| 4 | Initial synthetic-noise SNR evaluation | **done** |
| 5 | Calibration (temperature scaling, ECE) | |
| 6 | Open-set head | |
| 7 | RF model and audio/RF fusion | |
| 8 | Monitoring dashboard and integrated evaluation | |

Step 2 extracts 128 summary features from each usable clip (64 log-mel-bin
means and standard deviations) and trains a class-balanced Random Forest on
the existing group-based Tier 1 split. Test metrics use a threshold selected
on validation data. The implementation and reproducible results are described
in [PROGRESS.md](./PROGRESS.md); run it with
`python scripts/train_audio_baseline.py`.

Step 3 trains a compact PyTorch CNN on fixed one-second, 64-bin log-mel
spectrograms using the same recording-group split. It selects the best epoch
and probability threshold using validation data, saves metrics and weights
under `reports/`, and evaluates once on the held-out test set. Run it with
`python scripts/train_audio_cnn.py`.

The initial noise evaluation adds deterministic white and pink noise at
20, 10, 5, and 0 dB SNR to the held-out Tier 1 test clips. It keeps both
models fixed and reuses their clean-validation thresholds. See
[PROGRESS.md](./PROGRESS.md) and `reports/audio_noise_robustness.json`; run it
with `python scripts/evaluate_audio_noise.py`. These synthetic tests do not
replace testing with independent recordings of speech, traffic, wind, or music.

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
python -m pytest tests -q               # 11 tests
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

## The two decisions that make results trustworthy

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

## Splits

- **tier 1** — held-out recordings, same corpus. The everyday split (70/15/15 by group).
- **tier 2** — held-out drone *model*: train on Bebop, test on Mambo. Answers "does it detect drones, or only the drones it saw?"
- **tier 3** — held-out *corpus*. Column exists; needs a second dataset to fill.

## Layout

```
src/dronedet/
  contracts.py          shared types
  data/manifest.py      build + validate the manifest
scripts/build_manifest.py
tests/test_manifest.py  11 tests, incl. an injected-leak check
data/                   gitignored — raw corpus + manifest.parquet
```

`tests/test_manifest.py::test_validate_catches_injected_leak` deliberately
corrupts a split and asserts the validator raises. A guard nobody has watched
fail is not a guard.
