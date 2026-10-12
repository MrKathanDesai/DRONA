"""Build and validate the manifest — the single source of truth about the data.

Everything downstream reads this file and nothing else. Two ideas carry it:

1.  **group_id is the leakage unit.** DroneAudioDataset cuts every parent
    recording into five one-second slices. Those slices are near-duplicates,
    so a random clip-level split puts the same flight in train and test and
    the reported score measures memorisation. The same applies to the
    negatives, which are five-second ESC-50 clips cut the same way.

2.  **Labels come from metadata, not from filename guessing.** The negatives
    keep ESC-50's naming with a segment digit appended, so you *can* decode
    the category with arithmetic on the last field — but doing so disagrees
    with ESC-50's own metadata on about 0.2% of files. We join on
    (src_file, take) against esc50.csv instead, which is provably correct.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

ESC_NAME = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)\.wav$")


# ---------------------------------------------------------------- group ids
def positive_group(filename: str) -> str:
    """Parent recording for a drone clip.

    Names look like `B_S2_D1_067-bebop_000_.wav` or `mixed_membo_34-membo_2_.wav`;
    the part before the first '-' identifies the recording, and the trailing
    index is the slice number.
    """
    return filename.split("-")[0]


def drone_model(filename: str) -> str | None:
    low = filename.lower()
    if "bebop" in low:
        return "bebop"
    if "membo" in low or "mambo" in low:
        return "membo"
    return None


# ---------------------------------------------------------------- ESC-50 join
def load_esc50(meta_csv: Path) -> dict[tuple[str, str], dict]:
    """Index ESC-50 metadata by (src_file, take) — the pair that survives the
    renaming DroneAudioDataset applied."""
    out = {}
    with open(meta_csv) as fh:
        for r in csv.DictReader(fh):
            out[(r["src_file"], r["take"])] = r
    return out


def negative_info(filename: str, esc: dict) -> tuple[str, str | None]:
    """Return (group_id, category) for a background clip.

    Falls back gracefully for the ~372 synthetic clips (silence, white noise,
    running tap...) that are not from ESC-50 at all.
    """
    m = ESC_NAME.match(filename)
    if not m:
        # synthetic clip: strip the trailing index so repeats group together
        stem = Path(filename).stem
        return f"synth_{re.sub(r'\\d+$', '', stem)}", None
    _, src, take, _ = m.groups()
    row = esc.get((src, take))
    group = f"esc_{src}_{take}"            # all 5 segments of one ESC-50 clip
    return group, (row["category"] if row else None)


# ---------------------------------------------------------------- splitting
def _hash_frac(key: str, salt: str) -> float:
    """Deterministic 0-1 value for a group. Stable across runs and machines —
    unlike anything seeded from a shuffled list."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF


def assign_tier1(group_id: str, val=0.15, test=0.15, salt="tier1") -> str:
    """Held-out groups, same data source. The everyday split."""
    f = _hash_frac(group_id, salt)
    if f < test:
        return "test"
    if f < test + val:
        return "val"
    return "train"


def assign_tier2(row) -> str:
    """Held-out drone MODEL. Train on bebop, test on membo.

    Answers the question that matters operationally: does the detector
    recognise drones, or only the drones it was trained on? Background clips
    are split as in tier 1 so both sides have negatives.
    """
    if row.label == "drone":
        return "test" if row.drone_model == "membo" else "train"
    f = _hash_frac(row.group_id, "tier2")
    return "test" if f < 0.2 else "train"


# ---------------------------------------------------------------- build
MIN_DURATION = 0.3       # seconds; below this a 1 s window is mostly padding


def build(root: Path, esc_csv: Path, source: str = "droneaudio") -> pd.DataFrame:
    esc = load_esc50(esc_csv)
    rows = []
    for label, folder in (("drone", "yes_drone"), ("background", "unknown")):
        for p in sorted((root / folder).glob("*.wav")):
            name = p.name
            if label == "drone":
                group, cat = positive_group(name), None
                model = drone_model(name)
            else:
                group, cat = negative_info(name, esc)
                model = None
            info = sf.info(str(p))
            rows.append(dict(
                path=str(p), label=label, group_id=group, source=source,
                drone_model=model, bg_category=cat,
                sample_rate=info.samplerate, duration=round(info.duration, 4),
            ))
    df = pd.DataFrame(rows)

    # Quality gate. Excluded clips stay in the manifest with a stated reason
    # rather than vanishing, so the counts always reconcile.
    df["exclude_reason"] = ""
    too_short = df.duration < MIN_DURATION
    df.loc[too_short, "exclude_reason"] = f"duration < {MIN_DURATION}s"

    df["split_tier1"] = df.group_id.map(assign_tier1)
    df["split_tier2"] = df.apply(assign_tier2, axis=1)
    df.loc[df.exclude_reason != "", ["split_tier1", "split_tier2"]] = "excluded"
    # tier 3 (held-out dataset) needs a second corpus; the column exists so the
    # protocol is visible, and is filled when another source is added.
    df["split_tier3"] = "unavailable"
    return df


# ---------------------------------------------------------------- validation
class LeakageError(AssertionError):
    pass


def validate(df: pd.DataFrame) -> dict:
    """Fail loudly rather than silently reporting an inflated score.

    Run this at the top of every training script, not just once.
    """
    problems = []

    for tier in ("split_tier1", "split_tier2"):
        overlap = (df[df[tier] != "excluded"]
                     .groupby("group_id")[tier]
                     .nunique()
                     .loc[lambda s: s > 1])
        if len(overlap):
            problems.append(
                f"{tier}: {len(overlap)} group(s) span multiple splits, "
                f"e.g. {list(overlap.index[:3])}")

    if df.path.duplicated().any():
        problems.append(f"{int(df.path.duplicated().sum())} duplicate paths")

    for tier in ("split_tier1", "split_tier2"):
        for sp in df[tier].unique():
            sub = df[df[tier] == sp]
            if sp != "excluded" and sub.label.nunique() < 2:
                problems.append(f"{tier}/{sp} contains only one class")

    if problems:
        raise LeakageError("manifest validation failed:\n  - "
                           + "\n  - ".join(problems))

    return {
        "clips": len(df),
        "usable_clips": int((df.exclude_reason == "").sum()),
        "excluded": int((df.exclude_reason != "").sum()),
        "groups": df.group_id.nunique(),
        "drone_clips": int((df.label == "drone").sum()),
        "background_clips": int((df.label == "background").sum()),
        "imbalance": round((df.label == "background").sum()
                           / max((df.label == "drone").sum(), 1), 2),
    }
