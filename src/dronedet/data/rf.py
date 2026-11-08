"""DroneRF (Allahham et al., Data in Brief 2019): labels, splits, reading, PSD.

Each of the 227 segments is 0.25 s of raw RF amplitude from two receivers, one
on the lower and one on the upper half of the 2.4 GHz band: 10 M samples per
band, 40 MS/s. The paper's prose says 1 M samples; the files and its Table 1
say 10 M. The rate is not stated: it follows from 10.25 s of background across
41 segments.

Every class is ONE continuous recording, so neighbouring segments are the RF
equivalent of the slices in the audio corpus. Splits therefore take whole
blocks of consecutive segments, in time order. See `assign_splits`.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch

SEGMENT_SAMPLES = 10_000_000
BLOCK_SEGMENTS = 3            # 0.75 s of consecutive segments, the leakage unit

# BUI = experiment bit + 2 drone bits + 2 mode bits (paper section 2.5).
DRONES = {"00": "bebop", "01": "ar", "10": "phantom"}
MODES = {"00": "on_connected", "01": "hovering",
         "10": "flying", "11": "flying_video"}

# Table 1 of the paper; a full run must reproduce these.
EXPECT_SEGMENTS = {"bebop": 84, "ar": 81, "phantom": 21, "background": 41}
HELD_OUT_DRONE = "phantom"    # tier 2: never trained on, as Mambo is for audio

ARCHIVE_NAME = re.compile(r"_(\d{5})_([LH])\d?\.rar$")
MEMBER_NAME = re.compile(r"^(\d{5})([LH])_?(\d+)\.csv$")


@dataclass(frozen=True)
class RFConfig:
    sample_rate: int = 40_000_000
    windows_per_segment: int = 10    # 25 ms each
    n_fft: int = 2048                # 19.5 kHz per bin, as in Al-Sa'd et al.

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def window_s(self) -> float:
        return SEGMENT_SAMPLES / self.windows_per_segment / self.sample_rate

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


DEFAULT = RFConfig()


def decode_bui(bui: str) -> dict:
    """'10110' -> drone, ar, flying. Background has no drone or mode."""
    if bui[0] == "0":
        return dict(label="background", drone_model=None, mode=None)
    return dict(label="drone", drone_model=DRONES[bui[1:3]], mode=MODES[bui[3:]])


def archive_key(filename: str) -> str:
    """'RF Data_11000_L1.rar' -> '11000_L1' (one archive is named 'FR Data')."""
    m = ARCHIVE_NAME.search(filename)
    if not m:
        raise ValueError(f"unexpected archive name: {filename}")
    return filename[m.start() + 1:-len(".rar")]


def parse_member(name: str) -> tuple[str, str, int]:
    """'11000L_3.csv' -> ('11000', 'L', 3)."""
    m = MEMBER_NAME.match(name.rsplit("/", 1)[-1])
    if not m:
        raise ValueError(f"unexpected member name: {name}")
    return m[1], m[2], int(m[3])


def iter_archive(path):
    """Yield (member name, samples) without extracting anything to disk.

    The CSVs unpack to about 45 GB. Each is a single comma-separated line, so
    the concatenated stream splits back into members one line at a time.
    """
    names = subprocess.run(["bsdtar", "-tf", str(path)], check=True,
                           capture_output=True, text=True).stdout.split()
    names = [n for n in names if n.lower().endswith(".csv")]
    proc = subprocess.Popen(["bsdtar", "-xOf", str(path)],
                            stdout=subprocess.PIPE)
    try:
        for name in names:
            line = proc.stdout.readline()
            x = np.fromstring(line.decode("ascii"), sep=",", dtype=np.float32)
            if len(x) != SEGMENT_SAMPLES:
                raise ValueError(f"{path}:{name} has {len(x)} samples, "
                                 f"expected {SEGMENT_SAMPLES}")
            yield name, x
    finally:
        proc.stdout.close()
        code = proc.wait()
    if code:
        raise RuntimeError(f"bsdtar failed on {path} (exit {code})")


def segment_psd(x: np.ndarray, cfg: RFConfig = DEFAULT):
    """One segment -> (windows, bins) PSD in dB, plus each window's level.

    Averaging in power keeps short bursts visible. Absolute level is kept, as
    in `features.audio.log_mel`: what to do about it is the model's decision.
    """
    win = x.reshape(cfg.windows_per_segment, -1)
    _, p = welch(win, fs=cfg.sample_rate, window="hann", nperseg=cfg.n_fft,
                 noverlap=0, detrend=False, axis=-1)
    level = 10.0 * np.log10(np.mean(win.astype(np.float64) ** 2, axis=1) + 1e-10)
    return (10.0 * np.log10(p + 1e-10)).astype(np.float32), level.astype(np.float32)


def _held_out(n_blocks: int, frac: float) -> int:
    return max(1, round(n_blocks * frac))


def assign_splits(df: pd.DataFrame, val=0.15, test=0.15, tier2_test=0.2,
                  held_out: str = HELD_OUT_DRONE) -> pd.DataFrame:
    """Add group_id and the split columns to one row per (bui, segment).

    Within each recording the last blocks are test and the ones before them
    val. Time order rather than a hash: with so few blocks per recording, a
    hash can leave a class out of a split, and later data is the honest test.

    tier 1: held-out time, same drones.
    tier 2: the `held_out` drone is test, plus the last background blocks.
    tiers 3 and 4: excluded. DroneRF is a single lab session; RF has no second corpus.
    """
    df = df.copy()
    df["block"] = df.segment // BLOCK_SEGMENTS
    df["group_id"] = [f"rf_{b}_{k:02d}" for b, k in zip(df.bui, df.block)]
    df["split_tier1"] = "train"
    df["split_tier2"] = "train"
    df["split_tier3"] = df["split_tier4"] = "excluded"

    for _, sub in df.groupby("bui"):
        blocks = np.sort(sub.block.unique())
        n_test, n_val = _held_out(len(blocks), test), _held_out(len(blocks), val)
        rank = sub.block.map({b: i for i, b in enumerate(blocks)})
        df.loc[sub.index[rank >= len(blocks) - n_test], "split_tier1"] = "test"
        df.loc[sub.index[(rank >= len(blocks) - n_test - n_val)
                         & (rank < len(blocks) - n_test)], "split_tier1"] = "val"
        if sub.label.iloc[0] == "background":
            n = _held_out(len(blocks), tier2_test)
            df.loc[sub.index[rank >= len(blocks) - n], "split_tier2"] = "test"
    df.loc[df.drone_model == held_out, "split_tier2"] = "test"
    return df
