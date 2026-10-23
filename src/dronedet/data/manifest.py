from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

from dronedet.paths import DATA_DIR

ESC_NAME = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)\.wav$")


def positive_group(filename: str) -> str:
    return filename.split("-")[0]


def drone_model(filename: str) -> str | None:
    low = filename.lower()
    if "bebop" in low:
        return "bebop"
    if "membo" in low or "mambo" in low:
        return "membo"
    return None


def load_esc50(meta_csv: Path) -> dict[tuple[str, str], dict]:
    out = {}
    with open(meta_csv) as fh:
        for r in csv.DictReader(fh):
            out[(r["src_file"], r["take"])] = r
    return out


def negative_info(filename: str, esc: dict) -> tuple[str, str | None]:
    m = ESC_NAME.match(filename)
    if not m:
        stem = re.sub(r"\d+$", "", Path(filename).stem)
        return "synth_" + stem, None
    _, src, take, _ = m.groups()
    row = esc.get((src, take))
    group = f"esc_{src}_{take}"
    return group, (row["category"] if row else None)


def _hash_frac(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF


def assign_tier1(group_id: str, val=0.15, test=0.15, salt="tier1") -> str:
    f = _hash_frac(group_id, salt)
    if f < test:
        return "test"
    if f < test + val:
        return "val"
    return "train"


def assign_tier2(row) -> str:
    if row.label == "drone":
        return "test" if row.drone_model == "membo" else "train"
    f = _hash_frac(row.group_id, "tier2")
    return "test" if f < 0.2 else "train"


MIN_DURATION = 0.3


def build(root: Path, esc_csv: Path, source: str = "droneaudio",
          data_dir: Path = DATA_DIR) -> pd.DataFrame:
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
                path=p.relative_to(data_dir).as_posix(), label=label, group_id=group, source=source,
                drone_model=model, bg_category=cat,
                sample_rate=info.samplerate, duration=round(info.duration, 4),
            ))
    df = pd.DataFrame(rows)

    df["exclude_reason"] = ""
    too_short = df.duration < MIN_DURATION
    df.loc[too_short, "exclude_reason"] = f"duration < {MIN_DURATION}s"

    df["split_tier1"] = df.group_id.map(assign_tier1)
    df["split_tier2"] = df.apply(assign_tier2, axis=1)
    df.loc[df.exclude_reason != "", ["split_tier1", "split_tier2"]] = "excluded"
    df["split_tier3"] = "unavailable"
    return df


class LeakageError(AssertionError):
    pass


def validate(df: pd.DataFrame) -> dict:
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

    absolute = df.path.map(lambda p: Path(p).is_absolute())
    if absolute.any():
        problems.append(f"{int(absolute.sum())} absolute path(s); "
                        "paths must be relative to the data directory")

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
