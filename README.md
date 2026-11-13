# dronedet

Acoustic drone detection — a dataset-only project. No hardware required.

## Status

| Step | What | State |
|---|---|---|
| 1 | Data foundation: manifest, splits, leakage tests | **done** |
| 2 | Features + classical baseline | **done** — see `reports/baseline_audio.md` |
| 2b | Tier 3: Svanström outdoor audio into the manifest | **done** |
| 3 | Pretrained detector (frozen AudioSet CNN + small head), tiered evaluation, `detect.py` demo | **done** — see `reports/pretrained_audio.md` |
| 3a | CNN+GRU, CNN+LSTM and lightweight-transformer comparison | **done (experimental)** — shared feature input; indoor holdout and Svanström evaluation, no model promoted to deployment |
| 3b | Tier 4: outdoor drones (ITU ARIS) added to training, Svanström still test-only | **done, no PR-AUC gain** — see `reports/outdoor_audio.md` |
| 3c | Outdoor confuser training experiment with Svanström negatives | **partial** — helicopter and outdoor-background negatives tested; no outdoor aircraft-labeled recordings available |
| 4 | Noise stress test (6 conditions x 3 SNRs) + noise-augmentation ablation | **done** — see `reports/noise_robustness.md` |
| 5 | Uncertainty flags (review band, familiarity, ensemble) | **done** — see `reports/uncertainty_audio.md` |
| 6 | Open-set "unknown drone" (train Bebop, test Mambo) | **done, AUROC 0.81** — same report |
| 6b | Audio/RF results dashboard: all local audio replay sources, metrics, uncertainty/noise results, RF inventory and report downloads | **audio results available; RF research-only; learned fusion not trained** — `streamlit run app/dashboard.py` |
| 7 | Validate additional RF folders, then independent-session RF evaluation and (later) audio + RF fusion | in progress — RF calibration/open-set/replay code is ready but awaits the prepared DroneRF cache; partial DroneDetect V2 work is classification, not detection |

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
python scripts/train_outdoor.py         # tier 4, needs ITU ARIS
python scripts/eval_noise.py            # noise stress test + augmentation (~10 min)
python scripts/eval_uncertainty.py      # flags, open set, saves the detector (~1 min)
python scripts/prepare_rf.py            # download and prepare DroneRF; requires bsdtar or 7-Zip
python scripts/eval_rf_trust.py         # RF calibration + Phantom held-out unknown test
python scripts/detect.py data/raw_svanstrom/Audio/DRONE_001.wav
streamlit run app/dashboard.py          # live-feed replay with Grad-CAM
python scripts/make_figures.py          # presentation figures -> reports/figures/
python -m pytest tests -q               # 68 tests
```

On Windows, `prepare_rf.py` also supports 7-Zip. Install it with
`winget install --id 7zip.7zip -e`; the script detects the standard
`C:\Program Files\7-Zip\7z.exe` location automatically. Preparing DroneRF
downloads about 4 GB of archives and processes them into the local feature
cache.

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

## Outdoor training data (tier 4)

Adding ITU ARIS outdoor drones to training did not improve the unseen outdoor
test: PR-AUC 0.640 vs 0.649, CIs fully overlapping. The false-alarm rate did
land on target (1.0% vs 3.2%), but recall fell to 0.220; at tier 3's 3.2% FPR it
is 0.303 vs 0.330. The model learns ITU ARIS well (validation PR-AUC 0.995), so
this is a transfer failure, not a training failure. Trained on ITU ARIS alone,
it scores Svanström helicopters almost like drones (median 0.90 vs 0.99):
rotor sound without aircraft negatives is what it learns. `detect.py` now uses the tier-4
detector (it holds false alarms at the 1% target); `--head tier3` gives the old one.

### Outdoor confuser training experiment

Run `python scripts/train_svanstrom_confusers.py` after building the audio
embedding and feature caches. This separate experiment trains comparison heads
with and without a group-disjoint portion of Svanström's helicopter and outdoor
background negatives. Svanström drone recordings remain test-only, and both
variants use the same validation negatives and held-out test recordings.
The test set has 30 drone recordings and six per negative category, so its
71.4% drone prevalence makes PR-AUC unsuitable as a deployment estimate. The
pretrained head's window ROC-AUC rose from 0.695 to 0.864 and PR-AUC from
0.867 to 0.948; recall fell from 0.250 to 0.213. On the small held-out negative subset, helicopter false
alarms fell from 2/60 to 0/60 windows. These are exploratory within-corpus
results, not independent validation, and the experiment does not replace the
tier-4 detector. The integrated audio manifest has no explicitly
aircraft-labeled outdoor recordings, so aircraft-specific training remains
outstanding. Full results:
`reports/svanstrom_confusers.md`.

### Audio architecture comparison

Run `python scripts/compare_audio_architectures.py` to compare a compact CNN,
CNN+GRU, CNN+LSTM and a lightweight transformer. They share the same cached
log-mel features (pooled to 64 bands × 21 frames), training-only
normalization, group-safe indoor train/validation data, and a 1%-FPR threshold
fixed on validation negatives. Each trained model is evaluated on both the
indoor holdout and Svanström, without retraining or threshold changes. This is
an experiment, not a deployment model; Svanström was used in earlier work and
is not a blind independent-corpus test. CNN+GRU led the indoor comparison
(PR-AUC 0.985, recall 0.951), while the compact CNN led on Svanström
(PR-AUC 0.814, recall 0.283); the differing results show no single architecture
is a clear deployment winner. The older 2D CNN/PANNs results use different
features and are reported separately. See
`reports/audio_architecture_comparison.md`.

## Noise robustness and augmentation (step 4)

Tier-1 test clips mixed with voice, traffic, wind, environment, synthetic
music and a mix of all five, at 20 / 10 / 0 dB SNR, with the threshold left
where clean validation put it. At 20 dB nothing changes; at 0 dB recall falls
from 0.88 to 0.40-0.74. Training the head on one noisy copy of each clip
raises 0 dB recall in every condition (+0.04 to +0.12) without hurting clean
recall. Loud synthetic music is the one masker that also causes false alarms
(8.7%, 4.2% with augmentation). ESC-50 has no speech or music, so "voice" is
laughing / coughing / crying and "music" is synthetic.

## Uncertainty and open set (steps 5-6)

Each window is **drone**, **no drone**, **unknown drone** or **low
confidence**. The review band (10% of validation windows nearest the
threshold) is the useful signal: on tier 1 it holds 80% of the errors in 13%
of the windows. The distance-from-training flag catches an unseen drone
model (Mambo vs held-out Bebop, AUROC 0.81) but **not** the move outdoors
(1.7% of outdoor windows flagged). Ensemble spread added nothing.

## RF (DroneRF)

Every RF model scores near 1.000, including on drones it never trained on, and
**those scores do not measure detection.** On noise-floor windows with nothing
visibly transmitting, a forest still separates drone from background perfectly,
which points to the recording session or receiver. See `reports/rf_detection.md`.
RF needs a second corpus before any number here means anything.
The local `RF_Dataset2` CSV folder is an extracted copy of one DroneRF archive
(all 20 names and file sizes match). Download metadata identifies the 25 files
in `datasets_2` as a partial IEEE DataPort DroneDetect V2 copy (DOI
`10.21227/5jjj-1m32`). The official description specifies interleaved-float I/Q
from BladeRF, 1.2e8 complex samples per 2-second recording, and a 2.4375 GHz
center frequency. Of the 25 local recordings, 20 matched hovering/flying
recordings were evaluated in five-fold recording-grouped validation. The
MA1-vs-MAV prefix classification scored 1.000 on the recorded metrics, but this
small result is exploratory, is not drone-vs-background detection, and does
not establish deployment performance. Five switched-on recordings are excluded
because MAV has no matching examples. There are no background recordings. The
46 `IEEE_DATA` files resemble the same naming/format but lack source metadata.
See `reports/dronedetect_subset.md` for scope and limitations.
To run the deliberately limited two-prefix, recording-held-out experiment:
`python scripts/eval_dronedetect_subset.py`. It reports model-code classification
on matched hovering/flying recordings only; five switched-on recordings are
excluded because the MAV code lacks that mode. It does not report
drone-vs-background detection.

## Splits

- **tier 1** — held-out recordings, same corpus. The everyday split (70/15/15 by group).
- **tier 2** — held-out drone *model*: train on Bebop, test on Mambo. Answers "does it detect drones, or only the drones it saw?"
- **tier 3** — held-out *corpus*: train on all indoor recordings, test on the 90 outdoor Svanström recordings (drone / helicopter / background), cut into 1 s windows grouped by recording. The threshold comes from indoor validation; the outdoor set is never used for training or tuning.
- **tier 4** — tier 3's test set, with ITU ARIS outdoor drones added to train and val, split by its source intervals.

## Layout

```
src/dronedet/
  contracts.py          shared types
  paths.py              data root (override with DRONEDET_DATA)
  data/manifest.py      build + validate the manifest
  data/rf.py            DroneRF labels, time-ordered splits, streaming, PSD
  features/audio.py     load, fix length, log-mel
  models/baseline.py    level-invariant summary features + sklearn models
  models/pretrained.py  frozen PANNs MobileNetV1 encoder (no torchlibrosa)
  eval/metrics.py       threshold at FPR, grouped bootstrap, per-category FA
  eval/protocol.py      tier 1 / 2 / 3 protocol shared by every model
  features/noise.py     noise conditions, SNR mixing, synthetic music
  models/uncertainty.py review band, familiarity, ensemble, the four states
  models/explain.py     Grad-CAM through the frozen encoder + linear head
  live.py               window-by-window runtime shared by detect.py and the app
app/dashboard.py              Streamlit monitor (replay, states, Grad-CAM)
scripts/download_data.py
scripts/prepare_rf.py         DroneRF: download (4 GB, resumable), then PSD features
scripts/build_manifest.py
scripts/make_features.py
scripts/train_baseline.py
scripts/make_embeddings.py
scripts/train_pretrained.py
scripts/train_outdoor.py      tier 4
scripts/train_rf.py           RF detectors
scripts/eval_noise.py         noise stress test + augmentation ablation
scripts/eval_uncertainty.py   uncertainty flags, open set, saves data/models/detector.joblib
scripts/detect.py             run the detector on any audio file
scripts/make_figures.py       six presentation figures (spectrograms, results, timelines)
reports/baseline_audio.{md,json}
reports/pretrained_audio.{md,json}
reports/outdoor_audio.{md,json}
reports/noise_robustness.{md,json}
reports/uncertainty_audio.{md,json}
reports/rf_detection.{md,json}
tests/test_manifest.py        13 tests, incl. an injected-leak check
tests/test_audio_features.py  12 tests, incl. cache-reproduces-from-code
tests/test_baseline.py        5 tests, incl. level invariance
tests/test_pretrained.py      6 tests, incl. level invariance of embeddings
tests/test_rf.py              14 tests, incl. an injected-leak check
tests/test_noise.py           10 tests, incl. exact SNR
tests/test_uncertainty.py     8 tests, incl. Grad-CAM logit = head logit
data/                   gitignored — raw corpora, checkpoint, manifest, caches
```

`tests/test_manifest.py::test_validate_catches_injected_leak` deliberately
corrupts a split and asserts the validator raises. A guard nobody has watched
fail is not a guard.
