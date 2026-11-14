"""Noise corruption for the robustness stress test and for augmentation.

Conditions follow the proposal (voice, traffic, wind, music, environment,
mixed) at three severities, given as signal-to-noise ratios. The sources are
ESC-50 background recordings already in the manifest, so no new data is
needed. Two caveats, stated wherever results are reported:

- ESC-50 has no speech. "voice" means human vocal sounds (laughing,
  coughing, crying baby, sneezing), not conversation.
- ESC-50 has no music. "music" is synthetic: random chords of harmonic tones
  with a kick-drum pulse. It tests tonal masking, not real recordings.

Two conditions add real recordings from outside ESC-50, to close those gaps:

- "speech": read English speech, LibriSpeech dev-clean (Panayotov et al.
  2015, CC BY 4.0), split by speaker.
- "music_real": GTZAN (Tzanetakis & Cook 2002, 10 genres x 100 excerpts,
  research use), split by excerpt.

They are test-only conditions: augmentation keeps the original six
(`AUG_CONDITIONS`), fixed before any result was seen, so these two also test
whether augmentation carries over to noise it never trained on.

The encoder RMS-normalises every clip, so only the SNR matters, not the
absolute level of the mix.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000

CONDITIONS: dict[str, tuple[str, ...]] = {
    "voice": ("laughing", "coughing", "crying_baby", "sneezing"),
    "traffic": ("engine", "car_horn", "train", "siren"),
    "wind": ("wind",),
    "environment": ("rain", "thunderstorm", "chirping_birds", "sea_waves",
                    "crickets", "insects"),
    "music": (),          # synthetic, see synth_music
    "mixed": (),          # one of each of the above, equal power
    "speech": ("speech",),            # LibriSpeech, see load_external
    "music_real": ("music_real",),    # GTZAN, see load_external
}
AUG_CONDITIONS = ("voice", "traffic", "wind", "environment", "music", "mixed")
EXTERNAL = {"speech": "LibriSpeech/dev-clean", "music_real": "genres"}
# 1 s slices per group: 20 speakers x 20 and 500 excerpts x 4 per split
PER_GROUP = {"speech": 20, "music_real": 4}
SEVERITY_SNR_DB = {"low": 20.0, "medium": 10.0, "high": 0.0}
MIN_RMS = 1e-3            # ESC-50 clips are padded with silence; skip those slices


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `noise` scaled so that 10*log10(P_clean / P_noise) == snr_db."""
    if len(noise) != len(clean):
        noise = np.resize(noise, len(clean))
    pc, pn = rms(clean), rms(noise)
    if pc < 1e-8 or pn < 1e-8:
        return clean.astype(np.float32)
    gain = pc / (pn * 10.0 ** (snr_db / 20.0))
    return (clean + gain * noise).astype(np.float32)


def synth_music(rng: np.random.Generator, n: int = SAMPLE_RATE,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """Random triads (root 110-440 Hz, 4 harmonics, plucked decay) changing
    every 0.25-0.5 s, over a 100-140 bpm kick drum."""
    t = np.arange(n) / sr
    y = np.zeros(n)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.25, 0.5) * sr)
        seg = slice(pos, min(pos + length, n))
        ts = t[seg] - t[pos]
        root = 110.0 * 2 ** rng.uniform(0, 2)
        third = 4 if rng.random() < 0.5 else 3           # major or minor
        for semis in (0, third, 7):
            f = root * 2 ** (semis / 12)
            for h in range(1, 5):
                y[seg] += np.sin(2 * np.pi * f * h * ts + rng.uniform(0, 2 * np.pi)) \
                    * np.exp(-3.0 * ts) / h
        pos += length
    beat = 60.0 / rng.uniform(100, 140)
    for b in np.arange(rng.uniform(0, beat), n / sr, beat):
        a = int(b * sr)
        k = np.arange(min(int(0.12 * sr), n - a)) / sr
        y[a:a + len(k)] += 2.0 * np.sin(2 * np.pi * (60 + 90 * np.exp(-30 * k)) * k) \
            * np.exp(-25 * k)
    return (y / (rms(y) + 1e-12) * 0.05).astype(np.float32)


class NoiseBank:
    """1 s noise clips by condition. Build one bank from the split being
    corrupted (test) and a separate one from train for augmentation, so no
    noise recording is shared between them."""

    def __init__(self, clips: list[np.ndarray], categories: list[str],
                 groups: list[str]):
        self.by_cond: dict[str, list[tuple[np.ndarray, str]]] = {}
        for cond, cats in CONDITIONS.items():
            self.by_cond[cond] = [(c, g) for c, cat, g in zip(clips, categories, groups)
                                  if cat in cats and rms(c) >= MIN_RMS]
        for cond in ("voice", "traffic", "wind", "environment"):
            if not self.by_cond[cond]:
                raise ValueError(f"no usable noise clips for {cond!r}")

    def sample(self, cond: str, rng: np.random.Generator,
               avoid_group: str | None = None) -> np.ndarray:
        if cond == "music":
            return synth_music(rng)
        if cond == "mixed":
            parts = [self.sample(c, rng, avoid_group)
                     for c in ("voice", "traffic", "wind", "environment", "music")]
            return np.sum([p / (rms(p) + 1e-12) for p in parts], axis=0).astype(np.float32)
        pool = self.by_cond[cond]
        if not pool:
            raise ValueError(f"no noise clips for {cond!r} in this bank")
        for _ in range(20):
            clip, group = pool[rng.integers(len(pool))]
            if group != avoid_group:
                return clip
        return clip

    def corrupt(self, clean: np.ndarray, cond: str, snr_db: float,
                rng: np.random.Generator, avoid_group: str | None = None) -> np.ndarray:
        return mix(clean, self.sample(cond, rng, avoid_group), snr_db)


def _external_files(root, kind: str) -> list[tuple[str, str]]:
    """(path, group) for one external corpus, sorted. Group = speaker for
    LibriSpeech (dev-clean/<speaker>/<chapter>/<utt>.flac), excerpt for GTZAN
    (genres/<genre>/<genre>.000NN.au)."""
    from pathlib import Path
    base = Path(root) / EXTERNAL[kind]
    if kind == "speech":
        files = sorted(base.glob("*/*/*.flac"))
        return [(str(f), f"libri_{f.parts[-3]}") for f in files]
    files = sorted(list(base.glob("*/*.au")) + list(base.glob("*/*.wav")))
    return [(str(f), f"gtzan_{f.stem}") for f in files]


def load_external(root, kind: str, split: str, per_group: int | None = None,
                  seed: int = 0) -> tuple[list[np.ndarray], list[str], list[str]]:
    """1 s clips at 16 kHz from an external corpus, for one split.

    Groups (speakers / excerpts) are dealt alternately to "test" and "train"
    in sorted order, so no speaker or excerpt is in both. Each group gives at
    most `per_group` slices that are not near-silent."""
    from dronedet.features.audio import load_mono
    per_group = per_group or PER_GROUP[kind]
    files = _external_files(root, kind)
    if not files:
        raise FileNotFoundError(f"no {kind} files under {root}; "
                                "run scripts/download_data.py --noise")
    groups = sorted({g for _, g in files})
    keep = set(groups[0::2] if split == "test" else groups[1::2])
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[str]] = {}
    for f, g in files:
        if g in keep:
            by_group.setdefault(g, []).append(f)
    clips, cats, grps = [], [], []
    for g, fs in by_group.items():
        taken = 0
        for f in rng.permutation(fs):
            try:
                y = load_mono(f, SAMPLE_RATE)
            except Exception:         # GTZAN's jazz.00054 is a known corrupt file
                continue
            for a in range(0, len(y) - SAMPLE_RATE + 1, SAMPLE_RATE):
                c = y[a:a + SAMPLE_RATE]
                if rms(c) >= MIN_RMS:
                    clips.append(c.astype(np.float32))
                    cats.append(kind)
                    grps.append(g)
                    taken += 1
                    if taken >= per_group:
                        break
            if taken >= per_group:
                break
    return clips, cats, grps
