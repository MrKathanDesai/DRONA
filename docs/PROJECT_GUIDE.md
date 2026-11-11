# dronedet — Project Guide (audio module)

A complete walkthrough of the acoustic half of the *Counter-Drone Detection &
Classification* project: what every part does, how it works, why it was built
that way, what happens to the data at each step, and where every output lands.

RF is handled separately by the other teammate; this document covers audio
only. Status as of 27 Oct 2026.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [How to run everything](#2-how-to-run-everything)
3. [The data](#3-the-data)
4. [Data manipulation, step by step](#4-data-manipulation-step-by-step)
5. [Code walkthrough](#5-code-walkthrough)
6. [How models are evaluated](#6-how-models-are-evaluated)
7. [Results so far](#7-results-so-far)
8. [Where every output is generated](#8-where-every-output-is-generated)
9. [Tests: what they guard against](#9-tests-what-they-guard-against)
10. [Design decisions, and why](#10-design-decisions-and-why)
11. [Known limitations](#11-known-limitations)
12. [Progress against the proposal and next steps](#12-progress-against-the-proposal-and-next-steps)

---

## 1. The big picture

The goal of the audio module is to listen to a short audio clip and say
**"drone" or "no drone"**, with a score, in a way that still works on drones,
places and microphones it has never heard.

The pipeline:

```
raw audio files (3 sources)
        │
        ▼
 build_manifest.py ──► manifest.parquet      one row per 1 s clip: label, recording
        │                                    group, split for each of 3 test tiers
        ├──────────────────────────┐
        ▼                          ▼
 make_features.py           make_embeddings.py
 log-mel spectrograms       frozen pretrained CNN (PANNs MobileNetV1)
 (12602, 128, 63)           1024-d embedding per clip (12602, 1024)
        │                          │
        ▼                          ▼
 train_baseline.py          train_pretrained.py
 hand-made features +       logistic regression head on embeddings
 sklearn classifiers        (only the head is trained)
        │                          │
        ▼                          ▼
 reports/baseline_audio.md  reports/pretrained_audio.md
                                   │
                                   ▼
                       data/models/pretrained_head.joblib
                                   │
                                   ▼
                       detect.py  ──► run on any .wav file
```

Two detectors exist and are scored by the **same** evaluation code:

| Detector | Input | Model | Trained? |
|---|---|---|---|
| Classical baseline | 384 hand-made statistics of the log-mel spectrogram | logistic regression / gradient boosting | yes, seconds–minutes |
| Pretrained detector | 1024-d embedding from a CNN pretrained on AudioSet | logistic regression head | only the head, ~15 s |

No neural network is trained from scratch anywhere in the project.

---

## 2. How to run everything

One-time setup (macOS Homebrew Python refuses system-wide installs, so a
virtual environment is used):

```bash
cd /Users/kathandesai/drone
python3 -m venv .venv
source .venv/bin/activate          # needed in every new terminal
pip install -r requirements.txt
```

Full pipeline, in order:

```bash
python scripts/download_data.py        # fetch datasets + pretrained checkpoint (~985 MB)
python scripts/build_manifest.py       # -> data/manifest.parquet
python scripts/make_features.py        # -> data/features/       (~10 s)
python scripts/train_baseline.py       # -> reports/baseline_audio.*   (~2 min)
python scripts/make_embeddings.py      # -> data/embeddings/     (~25 s on Apple GPU)
python scripts/train_pretrained.py     # -> reports/pretrained_audio.*, data/models/  (~15 s)
python scripts/detect.py data/raw_svanstrom/Audio/DRONE_005.wav   # demo
python -m pytest tests -q              # 36 tests
```

`python scripts/download_data.py --verify` checks what is present without
downloading anything.

Dependencies (`requirements.txt`): pandas, pyarrow, soundfile, pytest,
librosa, scipy, scikit-learn, matplotlib, torch, joblib.

---

## 3. The data

Nothing in `data/` is committed to git; `scripts/download_data.py` is the
record of where everything came from.

### 3.1 Sources

| # | Dataset | Folder | Size | Role |
|---|---|---|---|---|
| 1 | **DroneAudioDataset** (Al-Emadi et al., IWCMC 2019) | `data/raw_droneaudio/` | 803 MB | Training corpus: drones and background, recorded indoors |
| 2 | **ESC-50 metadata** (Piczak 2015) | `data/esc50_meta.csv` | 92 KB | Labels only: the true category of each background clip |
| 3 | **Svanström multi-sensor dataset** (Halmstad, 2021) | `data/raw_svanstrom/Audio/` | 152 MB | Outdoor test-only corpus (tier 3) |
| 4 | **PANNs MobileNetV1 checkpoint** (Kong et al., 2020) | `data/pretrained/` | 23 MB | Frozen pretrained encoder + AudioSet label names |

### 3.2 What is inside

**DroneAudioDataset** (`Binary_Drone_Audio/`), 16 kHz mono:

| Folder | Clips | Independent recordings | Content |
|---|---|---|---|
| `yes_drone/` | 1,332 | 312 | Parrot **Bebop** (666) and **Mambo** (666), indoors |
| `unknown/` | 10,372 | 2,003 | All 50 ESC-50 categories × 200, plus 372 synthetic clips (silence, white/pink noise, taps, dishes…) |

Imbalance: **7.79 background clips per drone clip**, so plain accuracy is
meaningless here (always saying "no drone" scores 88.6%).

Hard confusers are included on purpose, 200 clips each: helicopter,
chainsaw, vacuum cleaner, washing machine, engine, airplane.

**Svanström outdoor corpus**: 90 recordings, 44.1 kHz **stereo**, 10 s
each, recorded outdoors with different hardware:

| Class | Recordings | 1 s windows |
|---|---|---|
| DRONE | 30 | 300 |
| HELICOPTER | 30 | 300 |
| BACKGROUND | 30 | 300 |

### 3.3 Traps found in the data

These were found by inspecting the data, and each one drove a design decision
(see §10).

1. **Clips are slices of longer recordings.** Every 5 s recording was cut
   into five 1 s slices before publication, so 1,332 drone clips are really
   only 312 independent recordings. The slices are near-duplicates.
2. **Clip duration gives the label away.** Drone clips are 1.024 s;
   background clips are 1.0217 / 0.9752 / 0.9597 s, and so on. Every distinct
   duration belongs to one class. A classifier given *only* the duration
   separates the classes perfectly (separability **1.0**).
3. **Loudness nearly gives the label away.** Overall clip level alone reaches
   separability **0.79**.
4. **Filename arithmetic is wrong for some labels.** The background category
   can be decoded as `last_field // 10`, but that disagrees with ESC-50's own
   metadata on about 20 of 10,000 files.
5. **Half the drone clips are "mixed" with noise** (`mixed_*` files, 670
   clips). It was checked (cross-correlation against every clean clip) that
   they are **not** noisy copies of published clean clips, so they cause no
   leakage.
6. **Two clips are shorter than 0.3 s**: too short to be meaningful.

---

## 4. Data manipulation, step by step

What happens to one clip on its way to a model:

| Step | What | Where | Why |
|---|---|---|---|
| 1 | Index the file: label, recording group, source, drone model, category, duration | `manifest.py` | One table every stage reads |
| 2 | Recover the background category from ESC-50 metadata | `manifest.py: negative_info` | Trap 4 |
| 3 | Group slices by parent recording (`group_id`) | `manifest.py` | Trap 1, the leakage unit |
| 4 | Assign splits by hashing `group_id` | `manifest.py: assign_tier1` | Same split on every machine, no seeds |
| 5 | Mark clips < 0.3 s as excluded (kept, not deleted) | `manifest.py: build` | Trap 6, an auditable exclusion |
| 6 | Cut outdoor recordings into ten 1 s windows (`start_s`) | `manifest.py: build_svanstrom` | Same unit as the training clips |
| 7 | Load → stereo to mono → resample to 16 kHz | `audio.py: load_mono` | One format for both corpora |
| 8 | Cut the window at `start_s` (outdoor) or use the whole clip | `audio.py: load_clip` | |
| 9 | Fix length to exactly 1 s: centre-crop long, **loop** short, never zero-pad | `audio.py: fix_length` | Trap 2 |
| 10a | **Baseline path:** 128-band log-mel spectrogram → 384 level-invariant statistics | `audio.py: log_mel`, `baseline.py` | Trap 3 |
| 10b | **Pretrained path:** RMS-normalise → upsample to 32 kHz → 64-band log-mel → frozen CNN → 1024-d | `pretrained.py` | Trap 3; the encoder expects 32 kHz |

### 4.1 The manifest

`data/manifest.parquet` has **12,604 rows** (11,704 indoor clips + 900
outdoor windows) and these columns:

| Column | Example | Meaning |
|---|---|---|
| `path` | `raw_droneaudio/Binary_Drone_Audio/yes_drone/B_S2_D1_067-bebop_000_.wav` | Relative to `data/`, so it works on any machine |
| `label` | `drone` / `background` | The target |
| `group_id` | `B_S2_D1_067`, `esc_172649_A`, `svan_DRONE_005` | Parent recording; all its slices share it |
| `source` | `droneaudio` / `svanstrom` | Corpus |
| `drone_model` | `bebop` / `membo` / empty | Used for tier 2 |
| `bg_category` | `helicopter`, `outdoor_background`, … | From ESC-50 metadata, or the Svanström class |
| `sample_rate` | 16000 / 44100 | Native rate of the file |
| `duration` | 1.024 | Seconds (for outdoor rows: the window length, 1.0) |
| `start_s` | NaN / 0.0 … 9.0 | NaN = whole clip; a number = window start inside a long recording |
| `exclude_reason` | `""` or `duration < 0.3s` | Why a row is excluded, if it is |
| `split_tier1/2/3` | `train` / `val` / `test` / `excluded` | Split for each evaluation tier |

Split sizes (clips):

| Tier | train | val | test | Meaning of the test set |
|---|---|---|---|---|
| tier 1 | 900 drone / 7,166 bg | 187 / 1,445 | 245 / 1,759 | Held-out indoor recordings |
| tier 2 | 666 / 8,257 | — | 666 / 2,113 | Mambo drones (never seen) |
| tier 3 | 1,145 / 8,925 | 187 / 1,445 | 300 / 600 | Outdoor Svanström corpus |

### 4.2 Two spectrogram front ends

| | Baseline log-mel | Pretrained encoder log-mel |
|---|---|---|
| Sample rate | 16 kHz | 32 kHz (upsampled from 16 kHz) |
| FFT / hop | 1024 (64 ms) / 256 (16 ms) | 1024 (32 ms) / 320 (10 ms) |
| Mel bands | 128, 20–8000 Hz | 64, 50–14000 Hz (checkpoint's own filterbank) |
| Shape per 1 s clip | (128, 63) | (101, 64) |
| Level handling | raw dB kept; statistics made level-invariant | waveform RMS-normalised before the spectrogram |

Everything is first resampled to 16 kHz, which band-limits it to 8 kHz.
Because of that, the 44.1 kHz outdoor recordings and the 16 kHz indoor clips
look the same to the encoder: the outdoor files can't be told apart just by
having more high-frequency content.

---

## 5. Code walkthrough

```
src/dronedet/
  contracts.py          shared types
  paths.py              data root (override with DRONEDET_DATA)
  data/manifest.py      build + validate the manifest
  features/audio.py     load, window, fix length, log-mel
  models/baseline.py    level-invariant summary features + sklearn models
  models/pretrained.py  frozen PANNs MobileNetV1 encoder
  eval/metrics.py       threshold at FPR, grouped bootstrap, per-category false alarms
  eval/protocol.py      tier 1 / 2 / 3 protocol shared by every model
scripts/                runnable entry points (one per pipeline stage)
tests/                  36 tests
```

### 5.1 `paths.py`: where the data lives

```python
DATA_DIR = Path(os.environ.get(
    "DRONEDET_DATA", Path(__file__).resolve().parents[2] / "data"))

def resolve(rel_path: str) -> Path:
    return DATA_DIR / rel_path
```

Manifest paths are relative, and `resolve()` turns them into real paths. Set
`DRONEDET_DATA=/somewhere/else` to keep the data on another disk (or in
Colab's Drive) without touching the code.

### 5.2 `data/manifest.py`: the index of every clip

**Grouping drone slices by recording.** The filename before the first `-` is
the parent recording:

```python
def positive_group(filename: str) -> str:
    return filename.split("-")[0]
# "B_S2_D1_067-bebop_000_.wav" and "B_S2_D1_067-bebop_004_.wav" -> "B_S2_D1_067"
```

**Background labels from metadata, not filename arithmetic:**

```python
def negative_info(filename: str, esc: dict) -> tuple[str, str | None]:
    m = ESC_NAME.match(filename)          # e.g. 1-172649-A-403.wav
    if not m:                             # synthetic clip (noise, silence…)
        stem = re.sub(r"\d+$", "", Path(filename).stem)
        return "synth_" + stem, None
    _, src, take, _ = m.groups()
    row = esc.get((src, take))            # join on (src_file, take)
    group = f"esc_{src}_{take}"           # all slices of one ESC-50 clip
    return group, (row["category"] if row else None)
```

The negatives get the same grouping as the positives, because ESC-50 clips
were sliced the same way.

**Deterministic splits by hashing the group:**

```python
def _hash_frac(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF      # uniform in [0, 1]

def assign_tier1(group_id: str, val=0.15, test=0.15, salt="tier1") -> str:
    f = _hash_frac(group_id, salt)
    if f < test:
        return "test"
    if f < test + val:
        return "val"
    return "train"
```

The split is computed from the group's name, so every slice of a recording
lands in the same split automatically, and the split is identical on every
machine, with no random seed that could be forgotten.

**Tier 2 holds out a whole drone model:**

```python
def assign_tier2(row) -> str:
    if row.label == "drone":
        return "test" if row.drone_model == "membo" else "train"
    f = _hash_frac(row.group_id, "tier2")
    return "test" if f < 0.2 else "train"
```

**Tier 3 uses all indoor data for training and the outdoor corpus for testing:**

```python
df["split_tier3"] = np.where(df.split_tier1 == "val", "val", "train")
```

```python
def build_svanstrom(root, window_s=1.0, data_dir=DATA_DIR):
    for p in sorted(root.glob("*.wav")):
        kind = p.stem.rsplit("_", 1)[0]               # DRONE / HELICOPTER / BACKGROUND
        label, cat = SVAN_LABELS[kind]
        info = sf.info(str(p))
        for k in range(int(info.duration // window_s)):   # 10 windows per file
            rows.append(dict(path=..., label=label,
                             group_id="svan_" + p.stem,   # one group per recording
                             source="svanstrom", start_s=k * window_s, ...))
    df["split_tier1"] = df["split_tier2"] = "excluded"   # never used in tiers 1-2
    df["split_tier3"] = "test"                            # never trained on
```

Indoor rows reuse the tier 1 validation split, so the tier 3 threshold is set
on indoor data the model did not train on. The outdoor corpus is never
touched for training *or* tuning.

**Validation fails loudly on leakage:**

```python
def validate(df):
    for tier in TIERS:
        overlap = (df[df[tier] != "excluded"]
                     .groupby("group_id")[tier].nunique()
                     .loc[lambda s: s > 1])            # a group in >1 split = leak
        ...
    dup = df.duplicated(["path", "start_s"])           # same window twice
    ...                                                # + absolute paths, one-class splits
    if problems:
        raise LeakageError(...)
```

`scripts/build_manifest.py` calls `build()` + `build_svanstrom()`, runs
`validate()`, then writes `data/manifest.parquet` (plus a human-readable
`manifest.csv`) and prints a summary of counts, groups and splits.

### 5.3 `features/audio.py`: loading and the log-mel spectrogram

**One config object, fingerprinted:**

```python
@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_fft: int = 1024          # 64 ms: resolves rotor harmonics ~16 Hz apart
    hop_length: int = 256      # 16 ms -> 63 frames per second
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0

    def fingerprint(self) -> str:      # hash of all settings
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
```

The fingerprint is stored next to the feature cache; a test fails if the
config changes and the cache is not rebuilt. The long 64 ms FFT gives fine
frequency resolution, because drone sound is a stack of narrow rotor
harmonics.

**Loading, any format to 16 kHz mono:**

```python
def load_mono(path, sample_rate):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)                                   # stereo -> mono
    if sr != sample_rate:                                # 44.1 kHz -> 16 kHz
        g = gcd(sr, sample_rate)
        y = resample_poly(y, sample_rate // g, sr // g).astype(np.float32)
    return y
```

`resample_poly` is an exact rational-ratio resampler with anti-aliasing (a
test checks that a 1 kHz tone at 44.1 kHz still peaks in the same mel band).

**Cutting one window:**

```python
def load_clip(path, start_s=None, cfg=DEFAULT):
    y = load_mono(path, cfg.sample_rate)
    if start_s is not None and not np.isnan(start_s):
        a = int(round(start_s * cfg.sample_rate))
        y = y[a:a + cfg.clip_samples]
    return fix_length(y, cfg.clip_samples)
```

`None`/NaN means "whole clip" (indoor rows); a number means "the window
starting here" (outdoor rows). An earlier draft treated `start_s = 0` as
"whole clip", which would have centre-cropped a 10 s file to its 4.5–5.5 s
segment instead of 0–1 s; a test now checks that window 0 is really the
first second.

**Fixing the length without leaking the label:**

```python
def fix_length(y, n):
    """Centre-crop long clips; loop short ones. Never zero-pad."""
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    return np.resize(y, n)          # repeats the clip to fill n samples
```

If short clips were padded with silence, the length of the silent tail would
reveal the class (trap 2). Looping fills the clip with the same sound
instead. This was checked: cutting full drone test clips to background
lengths and looping them moves the median score by 0.0000.

**The log-mel spectrogram:**

```python
def log_mel(y, cfg=DEFAULT):
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0)
    return (10.0 * np.log10(S + 1e-10)).astype(np.float32)    # dB, (128, 63)
```

The absolute level is deliberately kept here; deciding how to handle level
is left to each model.

`scripts/make_features.py` runs `clip_features()` on every usable clip in
parallel (`ProcessPoolExecutor`) and writes `data/features/X.npy` (12602 ×
128 × 63, float32, ~400 MB), `index.parquet` (row *i* of X = row *i* of the
index) and `config.json` (config fingerprint + manifest hash). A cache with no
`config.json` is moved aside to `features_legacy/` rather than overwritten.

### 5.4 `models/baseline.py`: the classical baseline

```python
def summary_features(X):
    """(N, mels, frames) dB -> (N, 3 * mels), invariant to overall level."""
    level = X.mean(axis=(1, 2), keepdims=True)
    shape = (X - level).mean(axis=2)                      # spectral shape per band
    spread = X.std(axis=2)                                # variation over time
    flux = np.abs(np.diff(X, axis=2)).mean(axis=2)        # frame-to-frame change
    return np.concatenate([shape, spread, flux], axis=1)  # 3 x 128 = 384 dims
```

Adding a constant number of dB to a clip (a louder recording or a closer
source) changes none of the three statistics, so the model cannot use
loudness (trap 3). A test adds 17 dB and checks the features are identical.

```python
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
    "hgb": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0),
}
```

`class_weight="balanced"` compensates for the 7.8 : 1 imbalance. Models are
stored as factories (lambdas), so each tier gets a fresh, untrained copy.

`scripts/train_baseline.py` runs both models through tiers 1–3 and also runs
the **shortcut probes**: it trains a tiny tree on *only* clip loudness, and
on *only* duration, to measure how much a careless pipeline could cheat. These
two variables are never given to the real models.

### 5.5 `models/pretrained.py`: the frozen pretrained encoder

**What it is:** PANNs MobileNetV1, a CNN trained by Kong et al. on Google's
AudioSet (2 million clips, 527 sound classes). It has 4.2 M parameters and a
24 MB checkpoint. It is used **frozen**: its weights never change. Only the
small classifier on top is trained.

**Why this model:** it is small enough to run on a laptop, it was trained on
16 kHz-compatible log-mels, its checkpoint is public with a checksum, and it
has already learned general sound structure (engines, propellers, insects,
wind) from millions of clips. That is what helps on drones it has not heard.

**The architecture** is rebuilt layer for layer, so the checkpoint loads
exactly:

```python
def _conv_dw(inp, oup, stride):       # depthwise-separable block
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.AvgPool2d(stride),
        nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

class MobileNetV1(nn.Module):
    def forward(self, logmel):                    # (B, 1, T=101, mels=64)
        x = self.bn0(logmel.transpose(1, 3)).transpose(1, 3)
        x = self.features(x).mean(dim=3)          # 14 conv blocks, average over frequency
        x = x.max(dim=2).values + x.mean(dim=2)   # max + mean over time -> (B, 1024)
        tags = torch.sigmoid(self.fc_audioset(F.relu(self.fc1(x))))   # 527 AudioSet tags
        return x, tags
```

The layer order (conv → avgpool → batchnorm → relu) was confirmed by
reading the checkpoint's key names (`features.1.0` conv, `features.1.2`
batchnorm, `features.1.4` pointwise conv…).

**Which layer is used as the embedding:** the pooled 1024-d conv output
`x`, *before* `fc1`. Three choices were compared (pooled, `fc1` output,
AudioSet tag logits). `fc1` is specialised for AudioSet's 527 tags, whereas
the pooled features are more general. On the unseen-drone test (tier 2), the
pooled features gave recall 0.65 versus 0.41.

**Loading safely:**

```python
sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
self.mel_w = sd["logmel_extractor.melW"].numpy()      # reuse the checkpoint's own mel filterbank
missing, unexpected = self.net.load_state_dict(sd, strict=False)
stray = [k for k in unexpected
         if not k.startswith(("spectrogram_extractor", "logmel_extractor"))]
if missing or stray:
    raise RuntimeError(f"checkpoint mismatch: {missing} {stray}")
```

`weights_only=True` means the checkpoint can only contain tensors, not
executable code. Any mismatched layer raises an error instead of silently
running with random weights.

**The front end** replaces the original `torchlibrosa` dependency with numpy:

```python
def logmel(self, y16k):
    rms = float(np.sqrt(np.mean(y16k ** 2)))
    y = y16k * (TARGET_RMS / rms) if rms > 1e-8 else y16k     # loudness removed
    y = resample_poly(y, 2, 1).astype(np.float32)             # 16 kHz -> 32 kHz
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=320, window="hann",
                            center=True, pad_mode="reflect")) ** 2
    return 10.0 * np.log10(np.maximum(S.T @ self.mel_w, 1e-10))   # (101, 64) dB
```

- **RMS normalisation** scales every clip to the same loudness (about
  −26 dBFS), so loudness (trap 3) can't reach the classifier. Comparing with
  and without it, normalisation clearly helped on tier 2. Digital silence
  (`rms ≈ 0`) is left as is rather than divided by zero.
- **Using the checkpoint's own mel filterbank** means the spectrogram matches
  what the network was trained on, without having to rebuild it from
  memory.

**Sanity check run before trusting it:** the encoder's own AudioSet tags
were printed for known clips. Outdoor helicopter → "Helicopter 0.88", rain →
"Rain 0.70". That confirms the weights and front end are correct.

`scripts/make_embeddings.py` loads every clip in parallel, encodes them in
batches of 128 on the Apple GPU (MPS) or the CPU, and writes
`data/embeddings/E.npy` (12602 × 1024), `tags.npy` (12602 × 527 AudioSet
probabilities), `index.parquet` and `config.json`.

### 5.6 `eval/metrics.py`: the measuring tools

**Threshold at a target false-alarm rate:**

```python
def threshold_at_fpr(neg_scores, fpr):
    s = np.sort(np.asarray(neg_scores))[::-1]         # highest first
    k = int(np.floor(fpr * len(s)))
    if k >= len(s):
        return -np.inf
    return float(np.nextafter(s[k], np.inf))           # just above the (k+1)-th highest
```

A detector needs a yes/no cut-off. It is set so that at most 1% of
*validation* background clips would raise an alarm, and then applied
unchanged to test. Choosing it on test would be cheating.

**Grouped bootstrap confidence interval:**

```python
def grouped_bootstrap(y, s, groups, metric, n=1000, seed=0):
    ...
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))            # resample recordings
        idx = np.concatenate([members[g] for g in pick])        # take all their clips
        vals.append(metric(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
```

Five slices of one recording are one piece of evidence, not five.
Resampling whole recordings gives honest (wider) error bars. A test checks
that duplicating every clip ten times does not narrow the interval.

**`false_alarms_by_category`** reports, at the chosen threshold, what
fraction of each background category (helicopter, chainsaw, …) was wrongly
called a drone.

### 5.7 `eval/protocol.py`: the three tiers, shared by every model

```python
def _train_val_test(F, df, tier, make_model, target_fpr):
    tr, va, te = (df[tier] == s for s in ("train", "val", "test"))
    model = make_model().fit(F[tr], df.y[tr])                   # train
    s_va, s_te = _scores(model, F[va]), _scores(model, F[te])
    thr = E.threshold_at_fpr(s_va[df.y[va].values == 0], target_fpr)   # threshold on val
    res = E.summarize(df.y[te], s_te, df.group_id[te], thr)            # score on test
    ...
```

- `tier1` / `tier3` use this train → val threshold → test path.
- `tier2` has no validation split (all Mambo is test), so the threshold comes
  from **out-of-fold** scores on train, using `GroupKFold(5)` so no
  recording is split across folds.
- `fit_deployable` returns the tier 3 model and threshold, i.e. trained on all
  indoor recordings except validation. This is what the demo uses.

Any feature matrix `F` plus any model factory can be passed in, which is
what guarantees the baseline and the pretrained model are compared fairly.

### 5.8 `scripts/train_pretrained.py`: training the head

```python
HEADS = {
    "logreg": lambda: make_pipeline(StandardScaler(),
        LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000)),
    "mlp": lambda: make_pipeline(StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2,
                      early_stopping=True, random_state=0, max_iter=300)),
}
DEPLOY = "logreg"
```

- `C=0.01` is strong regularisation. With 1024 features and only 312 drone
  recordings, a weakly regularised model memorises the indoor recordings and
  does worse outdoors.
- The MLP (one hidden layer of 128) was tried as the "shallow" alternative.
  It did not beat logistic regression, so the simpler model is deployed.

The script also evaluates tier 3 **per recording** with the same rule as
the demo:

```python
def recording_level(F, df, model, thr, min_hits=2):
    # alarm for a 10 s recording only if >= 2 consecutive 1 s windows fire
```

It saves `{"model", "threshold", "target_fpr", "encoder"}` to
`data/models/pretrained_head.joblib`.

### 5.9 `scripts/detect.py`: the demo

```bash
python scripts/detect.py path/to/file.wav [--hop 0.5] [--min-hits 2]
```

1. Loads any audio file and converts it to 16 kHz mono.
2. Slides a 1 s window every `--hop` seconds.
3. Encodes each window with the frozen CNN and scores it with the saved head.
4. Prints a timeline: time, probability, a bar, DRONE if above threshold, and
   the encoder's top 2 AudioSet tags (a free sanity check of what it hears).
5. Reports "DRONE DETECTED" only if at least `--min-hits` consecutive windows
   fire, which removes one-off false alarms.

Real output on an outdoor drone recording:

```
  data/raw_svanstrom/Audio/DRONE_005.wav  (10.0 s, 10 windows, threshold 0.841)

     0.0s  0.893 ##################   DRONE  Vehicle, Siren
     1.0s  0.985 #################### DRONE  Insect, Music
     ...
     9.0s  0.952 ###################  DRONE  Vehicle, Siren

  DRONE DETECTED: 10/10 windows above threshold, longest run 10 (need 2); max p 0.998
```

and on an outdoor helicopter (correctly rejected, though the encoder
recognises it as a helicopter):

```
     5.0s  0.548 ###########                 Helicopter, Vehicle
  no drone: 0/10 windows above threshold, longest run 0 (need 2); max p 0.548
```

### 5.10 `scripts/download_data.py`: reproducible downloads

Fetches each source only if missing, verifies counts (23,408 WAVs in
DroneAudioDataset, 90 in Svanström, 2,001 ESC-50 metadata rows) and the
checkpoint's MD5 (`a419303e1c88aa1b9d2ac3811563d371`, matching Zenodo).
From the Svanström archive (311 MB) only the 90 audio files are extracted;
the 650 videos are skipped.

---

## 6. How models are evaluated

### 6.1 The three tiers

Each tier asks a harder question:

| Tier | Train on | Test on | Question it answers |
|---|---|---|---|
| 1 | 70% of indoor recordings | 15% held-out indoor recordings | Does it work on new recordings of the same drones, same room? |
| 2 | Bebop drones + 80% of background | Mambo drones + 20% of background | Does it detect *drones*, or only the drone it was trained on? |
| 3 | All indoor recordings (except val) | 90 outdoor recordings | Does it survive a new place, new microphone, new drones? |

Tier 1 alone would make the project look nearly solved; tiers 2 and 3 show
how well it generalises.

### 6.2 The metrics

| Metric | What it tells you | Why it's used |
|---|---|---|
| **PR-AUC** | Ranking quality, focused on the rare drone class (chance = prevalence) | Accuracy is meaningless at 7.8 : 1 imbalance |
| **95% CI** | Range from the grouped bootstrap | Shows whether a difference is real |
| **ROC-AUC** | Ranking quality overall (chance = 0.5) | Standard, for comparison |
| **Recall @ 1% FPR** | Share of drones caught when only 1% of background raises an alarm | The number an operator cares about |
| **Test false alarms** | Count and rate of background wrongly flagged on test | Shows whether the threshold set on validation holds up on test |
| **False alarms by category** | Which sounds fool it (helicopter, chainsaw…) | Where to improve |

---

## 7. Results so far

Source: `reports/baseline_audio.md` and `reports/pretrained_audio.md`.

### 7.1 Per 1 s clip, threshold fixed at 1% validation FPR

| Model | Tier | PR-AUC [95% CI] | Recall | Test false alarms |
|---|---|---|---|---|
| baseline logreg | 1 | 0.966 [0.933, 0.986] | 0.833 | 5 / 1759 |
| baseline hgb | 1 | **0.981** [0.954, 0.998] | **0.935** | 3 / 1759 |
| pretrained logreg | 1 | 0.968 [0.929, 0.992] | 0.878 | 10 / 1759 |
| baseline logreg | 2 | 0.884 [0.844, 0.917] | 0.536 | 28 / 2113 |
| baseline hgb | 2 | 0.891 [0.845, 0.927] | 0.393 | 14 / 2113 |
| pretrained logreg | 2 | **0.929** [0.901, 0.952] | **0.647** | 22 / 2113 |
| baseline logreg | 3 | **0.735** [0.606, 0.840] | 0.073 | 0 / 600 |
| baseline hgb | 3 | 0.702 [0.562, 0.820] | 0.197 | 7 / 600 |
| pretrained logreg | 3 | 0.649 [0.514, 0.760] | **0.330** | 19 / 600 |

Prevalence (PR-AUC of random guessing): tier 1 0.122, tier 2 0.240, tier 3
0.333.

### 7.2 Tier 3 per 10 s outdoor recording (pretrained, ≥ 2 consecutive hits)

| Category | Alarm rate | Alarms / recordings |
|---|---|---|
| drone (detection rate) | 0.50 | 15 / 30 |
| helicopter (false alarms) | 0.07 | 2 / 30 |
| outdoor background (false alarms) | 0.00 | 0 / 30 |

### 7.3 What the numbers mean

- **Tier 1 is close to solved by both**; the baseline is slightly ahead.
- **Tier 2 is where the pretrained model wins.** On a drone it never heard,
  recall rises from 39–54% to 65%. The baseline had partly learned "Bebop"
  rather than "drone"; the pretrained features carry over better.
- **Tier 3 is weak for both.** The baseline ranks outdoor clips reasonably
  (PR-AUC 0.735) but its threshold does not transfer: it catches only 7%. The
  pretrained model keeps 33% recall, at 3% false alarms instead of 1%.
  Indoors-only training data is the bottleneck, and that is what
  augmentation (next step) addresses.
- **Helicopter is the main outdoor confuser**, as expected.

### 7.4 Shortcut probes

| Nuisance variable alone | Separability (0.5 = useless, 1.0 = gives the label away) |
|---|---|
| clip duration | **1.0** |
| clip loudness | **0.79** |

This is why `fix_length` never zero-pads, why the baseline features are
level-invariant, and why the encoder input is RMS-normalised.

---

## 8. Where every output is generated

| Output | Made by | Contents |
|---|---|---|
| `data/raw_droneaudio/`, `data/raw_svanstrom/Audio/`, `data/esc50_meta.csv` | `scripts/download_data.py` | Raw datasets |
| `data/pretrained/MobileNetV1_mAP=0.389.pth`, `class_labels_indices.csv` | `scripts/download_data.py` | Pretrained encoder + AudioSet label names |
| `data/manifest.parquet` (+ `manifest.csv`) | `scripts/build_manifest.py` | One row per clip/window: labels, groups, splits |
| `data/features/X.npy` | `scripts/make_features.py` | Log-mel, (12602, 128, 63) float32, ~390 MB |
| `data/features/index.parquet`, `config.json` | `scripts/make_features.py` | Row index; config fingerprint + manifest hash |
| `data/embeddings/E.npy` | `scripts/make_embeddings.py` | Frozen-encoder embeddings, (12602, 1024) |
| `data/embeddings/tags.npy` | `scripts/make_embeddings.py` | AudioSet tag probabilities, (12602, 527) |
| `data/embeddings/index.parquet`, `config.json` | `scripts/make_embeddings.py` | Row index; encoder name, RMS target, manifest hash |
| `reports/baseline_audio.md` / `.json` | `scripts/train_baseline.py` | Baseline results, tiers 1–3, shortcut probes, false alarms by category |
| `reports/pretrained_audio.md` / `.json` | `scripts/train_pretrained.py` | Pretrained results vs baseline, tier 3 per recording |
| `data/models/pretrained_head.joblib` | `scripts/train_pretrained.py` | Trained head + threshold, read by `detect.py` |
| terminal timeline | `scripts/detect.py` | Per-window scores and verdict for one audio file |
| `reports/figures/feature_examples.png` | *not made by any committed script* | Example spectrograms from an earlier session; gitignored |
| `data/features_legacy/` | an old uncommitted script | Unused; safe to delete (360 MB) |

`data/` is gitignored throughout; everything there can be rebuilt from the
scripts. `reports/*.md` and `*.json` are tracked in git (the
`pretrained_audio.*` files are new and not committed yet).

---

## 9. Tests: what they guard against

`python -m pytest tests -q` runs 36 tests. Tests that need data skip
themselves cleanly when it is missing.

| File | Count | Key checks |
|---|---|---|
| `test_manifest.py` | 13 | No group spans two splits (all tiers); **injected leak is caught** by the validator; tier 2 holds out all Mambo; categories match ESC-50 metadata (0 mismatches over 9,000+); splits are deterministic; short clips excluded but kept; no duplicate (path, window) rows; outdoor corpus is test-only; outdoor windows group by recording |
| `test_audio_features.py` | 12 | Shape matches config; centre-crop and loop work; **clip length is not a feature** (same noise at two lengths gives the same features; this test fails under zero-padding); silence is finite; resampling keeps pitch; window 0 is the first second; cached features match the current config, manifest, and code |
| `test_baseline.py` | 5 | Features unchanged by +17 dB; threshold respects target FPR; operating-point counts; bootstrap resamples groups, not clips; per-category false alarms |
| `test_pretrained.py` | 6 | Checkpoint loads every layer; **embedding unchanged by 0.1× or 3× volume**; silence is finite; different sounds give different embeddings; detection needs consecutive hits; windows cover the file |

The injected-leak test deliberately corrupts a split and asserts the
validator raises an error. A guard that has never been seen to fail could be
silently broken.

---

## 10. Design decisions, and why

| Decision | Why |
|---|---|
| Split by recording (`group_id`), never by clip | Slices of one recording are near-duplicates; a clip-level split puts the same flight in train and test, so the score would reflect memorisation |
| Splits from a hash of the group name | Identical on every machine and run; no seed to get wrong; all slices of a group move together |
| Labels joined from ESC-50 metadata | Filename arithmetic mislabels ~20 clips |
| Loop short clips, never zero-pad | Duration alone separates the classes perfectly (1.0); a silent tail would hand over the label |
| Level-invariant baseline features / RMS-normalised encoder input | Loudness alone separates the classes at 0.79; gain and distance must not be the cue |
| Excluded clips marked, not deleted | The exclusion is visible and countable |
| Three tiers instead of one test split | Tier 1 alone says "solved"; tiers 2–3 show the real generalisation gap |
| Outdoor corpus never used for training or threshold tuning | Otherwise tier 3 would not be an honest cross-dataset test |
| Threshold fixed on validation at 1% FPR | A detector is used at one operating point; tuning it on test would inflate results |
| PR-AUC + recall at fixed FPR, not accuracy | 7.8 : 1 imbalance makes accuracy meaningless |
| Grouped bootstrap CIs | Clip-level resampling overstates confidence |
| Frozen pretrained CNN + linear head instead of training a CNN | Project constraint: simple and working, no big model, no training from scratch. Runs in seconds on a laptop, and AudioSet knowledge transfers to unseen drones |
| MobileNetV1 (4.2 M params) rather than Cnn14 (80 M) or AST (87 M) | Small, fast, 24 MB, still pretrained on 2 M clips |
| Pooled embedding before `fc1` | `fc1` is tuned to AudioSet's tags; the pooled features transfer better (tier 2 recall 0.65 vs 0.41) |
| Strong regularisation (`C=0.01`) | 1024 dims vs 312 drone recordings; less overfitting to the indoor domain |
| Numpy front end with the checkpoint's own mel filterbank | Drops the `torchlibrosa` dependency, which is fragile with current librosa/numpy, and matches the training front end exactly |
| Everything band-limited to 8 kHz | Both corpora look alike to the encoder whatever their native rate |
| Two consecutive positive windows to alarm | Simple time smoothing; cuts one-off false alarms |
| One shared evaluation module | The baseline and pretrained numbers are guaranteed comparable |
| Caches store a config fingerprint + manifest hash | A stale cache is detected by tests instead of silently used |

---

## 11. Known limitations

- **Pretrained results are mildly optimistic.** The embedding layer and `C`
  were chosen from a small grid (3 C values × 3 layers × with/without RMS
  normalisation) while looking at the tier results. The baseline was not
  tuned this way.
- **Indoor-only training data.** All drones in training were recorded
  indoors; outdoor recall is only 33% per window and 50% per recording.
- **Two drone models** (Bebop, Mambo) in training. Svanström doesn't label
  the drone type per file, so outdoor results can't be broken down by drone.
- **1 s windows**, while the proposal mentions 2–4 s. The indoor corpus is
  published in 1 s slices.
- **No noise-robustness testing yet** (voice, traffic, wind, music masking).
- **No calibration, uncertainty, or "unknown drone" output yet.** The score
  is not yet a calibrated probability.
- **Small outdoor test set** (30 recordings per class), hence the wide CIs
  on tier 3.

---

## 12. Progress against the proposal and next steps

| Proposal item | Status |
|---|---|
| Data pipeline, leak-free splits, 3 test tiers | ✅ Done |
| Classical baseline | ✅ Done |
| Deep acoustic model | ✅ Done (frozen pretrained CNN + head) |
| Cross-dataset generalisation (outdoor) | ✅ Measured; this is the main weakness |
| Temporal smoothing (stretch goal) | ✅ Basic: 2 consecutive hits |
| Runnable demo | ✅ `scripts/detect.py` |
| CNN vs CNN+LSTM vs transformer comparison | ⚠️ Partial: baseline vs pretrained CNN only |
| Noise robustness (voice / traffic / wind / music / mixed × low / med / high) | ❌ Next |
| Drone-model classification + per-class confusion matrices | ❌ |
| Calibration + uncertainty (entropy / MC-dropout) | ❌ |
| Open-set "Unknown drone" | ❌ (tier 2 is a ready-made test: Mambo as unknown) |
| Streamlit dashboard + explainability overlay | ❌ (`detect.py` is the starting point) |
| Fusion with RF + ablation | ❌ Needs the RF teammate's outputs |

**Next steps, in order:**

1. **Noise augmentation and the robustness table.** Mix outdoor background,
   wind, speech, traffic and music into training clips at set signal-to-noise
   ratios (e.g. 20 / 10 / 0 dB), using the same frozen encoder and
   retraining only the head. This targets the outdoor weakness and produces
   the proposal's robustness table. Noise sources must be split by group too.
2. **Calibration and uncertainty.** Temperature scaling, a reliability
   measure (ECE), and a "low confidence" flag.
3. **Open-set detection.** Train on Bebop, check that Mambo is flagged as
   "unknown".
4. **Streamlit dashboard.** Replay clips as a live feed: status, confidence,
   AudioSet tags, spectrogram with an explainability overlay.
5. **Fusion contract with the RF teammate.** Agree on a per-window output
   such as `{p_drone, class_probs, uncertainty, is_unknown}` in
   `contracts.py`. Note that the audio and RF datasets were not recorded
   together, so fusion has to be trained on synthetic pairs matched by class.
