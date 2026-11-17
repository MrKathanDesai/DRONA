# dronedet

Acoustic + RF drone detection, built and evaluated on public datasets. No special hardware:
the live page runs on a laptop microphone. Every experiment has a results page in `reports/`.

```bash
python -m console                  # live detection page, http://localhost:8080
streamlit run app/dashboard.py     # results and evaluation dashboard
```

**AI Lab project, Ahmedabad University** (instructor: Prof. Maitrik Shah).
Team: Vidhi Rojasara (AU2644022) and Kathan Desai (AU2644002).

| Deliverable | Where |
|---|---|
| Results, one page per experiment | `reports/*.md` (raw numbers in `reports/*.json`) |
| Live detection page | `python -m console` |
| Results dashboard | `streamlit run app/dashboard.py` |

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
| 6b | Streamlit dashboard: project briefing (readiness checked from files on disk, how a prediction is made, the four states, headline results, limits), replay detector (recording-level result first, per-sensor evidence, Grad-CAM, optional recorded-RF replay), fusion, audio, RF, proposal checklist; System / Light / Dark theme. `detect.py` on the tier-4 detector | **done** — `streamlit run app/dashboard.py`; see "Reading the dashboard" below |
| 7 | Audio model comparison: CNN vs CNN+GRU vs CNN+LSTM vs transformer | **done** — see `reports/deep_audio.md`, `reports/audio_architecture_comparison.md` |
| 8 | Which-drone confusion matrices (audio, RF) | **done** — see `reports/identify.md` |
| 9 | RF uncertainty + unknown-signature flags | **done** — see `reports/rf_uncertainty.md` |
| 10 | RF on a new receiver (Tampere) + white-noise control | **done: RF v1 calls pure noise "drone"** — see `reports/rf_tampere.md` |
| 10b | RF v2: relabelled silence, receiver-noise negatives, second-receiver drones | **done: noise is "no drone", 9/10 unseen drones caught** — see `reports/rf_v2.md` |
| 11 | Audio + RF fusion, ablation, dashboard RF panel | **evaluated (unpaired-data upper bound)** — evidence sum deployed; the learned layer is evaluated but not deployed (0.2% recall on RF-silent drones vs 18.4% for the sum); see `reports/fusion.md`. Rebuilt and re-verified on 14 Nov 2026 (`rf_detector.joblib`, `fusion_models.joblib`); the dashboard replays recorded RF windows and shows the fused decision |
| 12 | Real speech + music in the noise test | **done** — in `reports/noise_robustness.md` |
| 12b | RF v3 experiment: simulated Wi-Fi / Bluetooth / microwave negatives, forest vs 1D-CNN, 3-of-5 smoothing, time-frequency features | **done** — see `reports/rf_v3.md` |
| 12c | RF probability calibration + held-out Phantom (Vidhi) | **done** — see `reports/rf_trust.md` |
| 13 | Audio baseline + 2D CNN, synthetic white/pink noise, Svanström confuser training, DroneDetect subset (Vidhi) | **done** — see `reports/audio_cnn.json`, `reports/svanstrom_confusers.md`, `reports/dronedetect_subset.md` |
| 14 | Live detection page: microphone / file, recorded radio panel, flags | **done** — `python -m console` |
| 14b | Live alarm rule calibrated on validation recordings | **done: alarm on the first drone window** — see `reports/live_calibration.md` |
| 14c | Detector trained on simulated speaker -> microphone playback | **done, opt-in** — see `reports/mic_robustness.md` |
| 15 | Music masking benchmark: drones under real music, -10 to +10 dB; stationary-component front-end (negative result) vs music-aware training | **done: music-aware training more than doubles detection at 0 and -5 dB** — see `reports/masking.md` |
| 16 | Live page redesigned as an operations console: sensor tiles, site coverage map, activity timeline, incident queue with playback and CSV export | **done** — `python -m console` |
| 18 | Outdoor aircraft validation on AeroSonicDB (ADS-B-labelled overflights, held-out recording sessions) | **done: 0 of 99 aircraft and 0 of 317 outdoor-ambience recordings raise an alarm** — see `reports/aircraft.md` |
| 17 | Operator-in-the-loop learning: labelled incidents retrain the detector behind a champion/challenger safety gate, with versions and rollback | **done** — `src/dronedet/feedback.py`, `scripts/retrain_feedback.py` |
| open | Independent RF validation with background recordings (KU Leuven / CardRF need a request or login) and paired audio + RF for fusion (no public co-recorded set found) | needs data |
| next | Real Wi-Fi / Bluetooth recordings (CardRF / DroneDetect) to measure RF false alarms on real traffic | needs a DataPort / CardRF login |

## Reading the dashboard

**What it is.** A research dashboard over recorded datasets. Nothing in it is
connected to a live microphone array or an RF receiver, and nothing in it is
field-validated. (The live microphone page is the separate console,
`python -m console`.)

**The four states** (per 1-second window, and summarised per recording):

| State | Meaning |
|---|---|
| No Drone | The available sensor evidence is below the decision threshold. Not proof that no drone is physically present. |
| Known Drone | Drone-like evidence above the threshold, without the unknown-signature flag. "Resembles the drones in training", **not** an identification of the drone model. |
| Unknown Drone | Drone-like evidence above the threshold, with a signature unlike every training drone. |
| Low Confidence | Ambiguous, near the review boundary, or unlike the training data: a person should review it. |

A recording raises **Known/Unknown Drone** only after N consecutive drone
windows (sidebar setting, default 2); isolated hits or repeated
low-confidence windows give **Low Confidence**. This is the same rule as the
console (`console/server.py`). The rule lives in `src/dronedet/status.py`.

**Scores.** The audio score (0–1) and the RF score are model scores, **not
calibrated probabilities**. Fused values are log-likelihood ratios (positive
favours drone). Thresholds are set for 1% false alarms on validation data.

**What a result is, and is not:**

- *Single-window prediction*: one 1-s window; never an alarm on its own.
- *Recording-level summary*: the consecutive-window rule above, on one recording.
- *Experimental replay*: with an RF source selected, recorded RF **test**
  windows of the chosen class are replayed beside the audio. They were never
  recorded with that audio, so a fused result is an unpaired what-if, not a
  synchronized multimodal decision.
- *Field-validated outcome*: none exists yet (needs field recordings).

**Reports vs. artifacts.** A report in `reports/` (e.g. `fusion.md`,
`rf_v2.md`) can exist without the model file it describes. The briefing page
checks `data/` directly. On a fresh copy of the project, the RF and fusion
models must be rebuilt in this order:

```bash
python scripts/prepare_rf.py --verify          # 23 DroneRF archives, SHA-256
python scripts/prepare_rf.py --process-only --workers 3   # -> data/rf_features/ (1-3 h; ~0.4 GB RAM per worker)
python scripts/train_rf.py                     # -> rf_forest.joblib (v1, diagnostic only)
python scripts/eval_rf_trust.py                # -> rf_trust.joblib  (RF research tab)
python scripts/download_data.py --only-extra   # Tampere RF (5.3 GB) + speech/music noise (1.6 GB)
python scripts/eval_rf_tampere.py              # -> data/rf_features_tampere/
python scripts/train_rf_v2.py                  # -> rf_detector.joblib + data/rf_features_v2/
python scripts/eval_fusion.py                  # -> fusion.joblib (sum) + fusion_models.joblib (all rules)
```

`fusion_models.joblib` holds every rule and names the deployed one (the sum);
a project that only has the older `fusion.joblib` still replays with the sum.

**Audio input** (the **ACOUSTIC-01** tile on the *Replay detector* page; the RF feed and fusion rule are in the **RADIO-01** tile, and Theme and Replay settings in the collapsible sidebar):

- dataset recordings: Svanström outdoor, DroneAudioDataset, ITU ARIS;
- **any audio or video file**: WAV, MP3, M4A, FLAC, OGG, AAC, MP4, MOV, WEBM,
  MKV, AVI, … (video is used for its sound only);
- **a link**: YouTube and most video sites (via `yt-dlp`), or a direct link to
  an audio/video file;
- **the microphone**: record a few seconds in the browser.

Everything is decoded to 16 kHz mono (`src/dronedet/media.py`, bundled ffmpeg
from `imageio-ffmpeg`); at most the first 5 minutes are analysed. Expect
misses on outdoor or internet clips: outdoors the detector catches about a
fifth of drone windows (see the results). Use only clips you may use.

**Look.** The dashboard uses the operations console's design (the approved
UI in `console/static/index.html`): the "DRONA" top bar with a state badge
(STANDBY until a clip is analysed, then that clip's final state), sensor
status dots and a UTC clock; hairline panels; small uppercase labels; the
console's colours. The first page, *Replay detector*, uses the console's
three-column layout: sensor tiles (ACOUSTIC-01, RADIO-01, MODEL) with their
controls on the left, the site sensor-coverage map and acoustic spectrum in
the middle (with the full result below), and an incident list with CSV export
on the right.

**Theme.** The dashboard opens in **Dark**. Sidebar → Theme switches to System
(follows the OS/browser) or Light. The choice is kept in the URL
(`?theme=light`), so it survives reruns and reloads; it drives Streamlit's own
theme setting (`.streamlit/config.toml` defines both palettes), so tables,
charts and widgets switch with it.

**Detail level.** The Audio and RF tabs show their headline result and the
full reports; the long experiment tables are behind a *Show detailed tables*
switch. Only values stored in `reports/` are shown (nothing recomputed in the
dashboard). Each model's training settings are in its training script under
`scripts/`.

## Setup

macOS ships a Homebrew Python that refuses system-wide installs (PEP 668),
so the project uses a virtual environment. Do this once:

```bash
cd DRONA                      # the cloned repository
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
python scripts/train_deep.py            # CNN / CNN+GRU / transformer (~1 h awake)
python scripts/eval_identify.py         # which-drone confusion matrices
python scripts/prepare_rf.py            # DroneRF: 4 GB download + PSD features
python scripts/train_rf.py              # RF detectors + quiet-window check
python scripts/eval_rf_uncertainty.py   # RF v1 flags and open set
python scripts/eval_rf_tampere.py       # RF v1 on a new receiver + white-noise control
python scripts/train_rf_v2.py           # RF v2, saves data/models/rf_detector.joblib
python scripts/eval_fusion.py           # ablation; saves sum + learned fusion models
python scripts/train_rf_v3.py           # RF v3 experiment (simulated Wi-Fi/BT/microwave), report only
python scripts/eval_rf_trust.py         # RF calibration + Phantom held-out unknown test
python scripts/train_audio_baseline.py  # Vidhi: random-forest audio baseline
python scripts/train_audio_cnn.py       # Vidhi: 2D audio CNN
python scripts/compare_audio_architectures.py  # CNN / CNN+GRU / CNN+LSTM / transformer
python scripts/evaluate_audio_noise.py  # synthetic white/pink noise test
python scripts/train_svanstrom_confusers.py    # outdoor helicopter/background negatives
python scripts/eval_dronedetect_subset.py      # DroneDetect subset (needs Vidhi's local copy)
python scripts/detect.py data/raw_svanstrom/Audio/DRONE_001.wav
streamlit run app/dashboard.py          # live-feed replay, Grad-CAM, optional RF feed
python scripts/make_figures.py          # presentation figures -> reports/figures/
python -m pytest tests -q               # 87 tests
```

On Windows, `prepare_rf.py` also supports 7-Zip. Install it with
`winget install --id 7zip.7zip -e`; the script detects the standard
`C:\Program Files\7-Zip\7z.exe` location automatically. Preparing DroneRF
downloads about 4 GB of archives and processes them into the local feature
cache.

In VS Code, press ⇧⌘P → "Python: Select Interpreter" → pick the one inside
`.venv`, so the editor and the terminal agree.

The raw corpora and the pretrained checkpoint are not committed. Fetch
everything (~985 MB) with `python scripts/download_data.py`; add `--extra`
for real speech/music noise (1.6 GB) and the Tampere RF recordings (5.3 GB).
See `data/README.md` for provenance. Long runs: keep the Mac awake
(`caffeinate -i python scripts/train_deep.py`), or sleep stretches them to hours.

## Live detection page (operations console)

```bash
pip install -r console/requirements.txt
python -m console                       # http://localhost:8080
python -m console --mic-model speaker   # mic uses the playback-trained detector
```

- **Acoustic sensor (primary):** microphone or an audio file (played out loud
  and checked as it plays). Browser voice processing is off so rotor hum is
  not filtered out. One decision per second; the alarm rule (first drone
  window, held for three) was chosen on validation recordings
  (`reports/live_calibration.md`).
- **Radio sensor (secondary):** a laptop cannot receive drone radio (its Wi-Fi
  chip only hands over decoded Wi-Fi packets), so this replays recorded
  2.4 GHz captures from the held-out test split, scored by RF v2, judged
  independently of the sound. Live radio needs a software-defined radio
  covering 2.4 GHz (HackRF One, ADALM-Pluto).
- **Site map, timeline, incidents:** sensor coverage lights up on contact
  (the sensors are omnidirectional, so no bearing is shown); every detection
  becomes an incident with its sources, audio playback and CSV export.
- **Session review:** the current recording's sound is kept in memory (up to
  15 minutes, dropped when a new source starts) and drawn as a strip of
  per-second calls. Drag across any stretch to replay it and mark it *drone*
  (for example a drone the detector missed) or *not drone*.
- **Learning from the operator:** each detection is reviewed as *Drone* or
  *Not drone*; *Report missed drone* marks the last 8 s as a drone; any
  stretch can be marked from the session strip. Only labelled stretches are
  stored (`data/feedback/`). Retraining (`python scripts/retrain_feedback.py`,
  or `POST /retrain` on the running console, which swaps the model in) rebuilds
  the detector head on the original training data plus every label, in
  seconds, and promotes the new model only if it passes a gate: no more than
  0.02 drone recall lost and no more false alarms on the outdoor held-out
  benchmark, and at least as good on the labelled clips under
  incident-grouped cross-validation. Every promoted model is versioned
  (`data/models/versions/`, `registry.json`) and can be rolled back.

Hosting: `python -m console --host 0.0.0.0`, or
`docker build -t drona . && docker run -p 8080:8080 -v "$PWD/data:/app/data" drona`.
Browsers allow the microphone only on `https://` or `localhost`.

The Streamlit app (`streamlit run app/dashboard.py`) stays as the separate
results and evaluation dashboard.

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
Run on another receiver (Tampere anechoic-chamber recordings, 11 new drones),
the forest calls every window "drone" — and calls **pure white noise** "drone"
too. Its rule is "not the DroneRF background recording" (`reports/rf_tampere.md`).
**RF v2** (`scripts/train_rf_v2.py`, `reports/rf_v2.md`) fixes the data, not the
model: DroneRF drone windows with no visible transmission are no longer labelled
"drone"; synthetic receiver noise through both pipelines is added as negatives;
Tampere drones from a second receiver are added as positives, split by drone. v2
calls noise and silent windows "no drone" and catches 9 of 10 Tampere drones it
never trained on. Still unmeasured: false alarms on real Wi-Fi / Bluetooth
recorded outside DroneRF's lab.

## Fusion

Each sensor's score is calibrated to a log-likelihood ratio on its own
validation data, capped, and summed (`src/dronedet/fusion.py`); the proposal's
learned layer (logistic regression on the two) is reported alongside. Pairs are
synthetic, since no corpus records both. With RF v2, fusion follows RF when
both see a drone, keeps most of audio's recall for an RF-silent drone (the
learned layer does not), and cuts helicopter false alarms. With RF v1, receiver
noise drove every rule but AND to 100% false alarms. See `reports/fusion.md`.

**RF v3** (`scripts/train_rf_v3.py`, `reports/rf_v3.md`) is an experiment, not
deployed: it adds simulated Wi-Fi, microwave and (held-out) Bluetooth signals as
negatives, compares the forest with a 1D-CNN, tests 3-of-5 smoothing and adds
time-frequency (timing) features on raw Tampere IQ.

**DroneDetect subset** (Vidhi): RF needs a second corpus before any number here means anything.
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
  eval/protocol.py      tier 1-4 protocol shared by every model (sample weights, holdout)
  data/rf_tampere.py    Tampere IQ -> DroneRF's two bands and PSD
  data/rf_noise.py      synthetic receiver noise (no transmitter), both pipelines
  models/deep.py        CNN, CNN+GRU, tiny AST; sklearn-style trainer
  fusion.py             calibration to LLR, capped sum / learned layer, four states
  features/noise.py     noise conditions, SNR mixing, synthetic music
  models/uncertainty.py review band, familiarity, ensemble, the four states
  models/explain.py     Grad-CAM through the frozen encoder + linear head
  live.py               window-by-window runtime, RF replay feed, fusion
app/dashboard.py              Streamlit monitor (replay, states, Grad-CAM, RF + fused)
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
scripts/train_deep.py         CNN vs CNN+GRU vs transformer, all tiers
scripts/eval_identify.py      which-drone confusion matrices
scripts/eval_rf_uncertainty.py RF v1 flags, saves data/models/rf_detector_dronerf.joblib
scripts/eval_rf_tampere.py    RF v1 on a new receiver, white-noise control
scripts/train_rf_v2.py        RF v2 (deployed), saves data/models/rf_detector.joblib
scripts/eval_fusion.py        fusion ablation; saves sum + learned models for replay
scripts/detect.py             run the detector on any audio file
scripts/make_figures.py       six presentation figures (spectrograms, results, timelines)
reports/baseline_audio.{md,json}
reports/pretrained_audio.{md,json}
reports/outdoor_audio.{md,json}
reports/noise_robustness.{md,json}
reports/uncertainty_audio.{md,json}
reports/rf_detection.{md,json}
reports/deep_audio.{md,json}, identify, rf_uncertainty, rf_tampere, rf_v2, fusion
tests/test_manifest.py        13 tests, incl. an injected-leak check
tests/test_audio_features.py  12 tests, incl. cache-reproduces-from-code
tests/test_baseline.py        5 tests, incl. level invariance
tests/test_pretrained.py      6 tests, incl. level invariance of embeddings
tests/test_rf.py              14 tests, incl. an injected-leak check
tests/test_noise.py           10 tests, incl. exact SNR
tests/test_uncertainty.py     8 tests, incl. Grad-CAM logit = head logit
tests/test_rf_tampere.py      3 tests, incl. a tone lands in the right band and bin
tests/test_fusion_deep.py     9 tests, incl. per-sensor-set false-alarm rate, level invariance
tests/test_rf_v2.py           8 tests, incl. noise has no carrier, tilt ignores common gain
data/                   gitignored — raw corpora, checkpoint, manifest, caches
```

`tests/test_manifest.py::test_validate_catches_injected_leak` deliberately
corrupts a split and asserts the validator raises. A guard nobody has watched
fail is not a guard.
