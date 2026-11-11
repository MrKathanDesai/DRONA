"""Noise corruption for the robustness stress test and for augmentation.

Conditions follow the proposal (voice, traffic, wind, music, environment,
mixed) at three severities, given as signal-to-noise ratios. The sources are
ESC-50 background recordings already in the manifest, so no new data is
needed. Two caveats, stated wherever results are reported:

- ESC-50 has no speech. "voice" means human vocal sounds (laughing,
  coughing, crying baby, sneezing), not conversation.
- ESC-50 has no music. "music" is synthetic: random chords of harmonic tones
  with a kick-drum pulse. It tests tonal masking, not real recordings.

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
}
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
        for _ in range(20):
            clip, group = pool[rng.integers(len(pool))]
            if group != avoid_group:
                return clip
        return clip

    def corrupt(self, clean: np.ndarray, cond: str, snr_db: float,
                rng: np.random.Generator, avoid_group: str | None = None) -> np.ndarray:
        return mix(clean, self.sample(cond, rng, avoid_group), snr_db)
