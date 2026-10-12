#!/usr/bin/env python3
"""Build the manifest and print what's in it.

    python scripts/build_manifest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                        # noqa: E402
from dronedet.data import manifest as M                    # noqa: E402

RAW = ROOT / "data" / "raw_droneaudio" / "Binary_Drone_Audio"
ESC = ROOT / "data" / "esc50_meta.csv"
OUT = ROOT / "data" / "manifest.parquet"


def main() -> int:
    print("scanning audio (reads a header per file, takes a moment)...")
    df = M.build(RAW, ESC)
    stats = M.validate(df)                 # raises if anything leaks

    df.to_parquet(OUT, index=False)

    print("\n" + "=" * 64)
    print("  MANIFEST")
    print("=" * 64)
    for k, v in stats.items():
        print(f"  {k:20s} {v}")

    print(f"\n  sample rates      {dict(df.sample_rate.value_counts())}")
    print(f"  duration          {df.duration.min():.3f}–{df.duration.max():.3f} s "
          f"(median {df.duration.median():.3f})")

    print("\n  clips per group:")
    g = df.groupby("group_id").size()
    print(f"    drone groups      {df[df.label=='drone'].group_id.nunique():5d}")
    print(f"    background groups {df[df.label=='background'].group_id.nunique():5d}")
    print(f"    slices per group  min {g.min()}  median {int(g.median())}  max {g.max()}")

    print("\n  drone models:")
    for m, n in df[df.label == "drone"].drone_model.value_counts().items():
        print(f"    {m:12s} {n:5d} clips, "
              f"{df[df.drone_model==m].group_id.nunique()} recordings")

    print("\n  background categories (top 8 of "
          f"{df.bg_category.nunique()}):")
    for c, n in df.bg_category.value_counts().head(8).items():
        print(f"    {c:18s} {n}")
    hard = ["helicopter", "chainsaw", "vacuum_cleaner", "washing_machine",
            "engine", "airplane"]
    present = {c: int((df.bg_category == c).sum()) for c in hard}
    print(f"\n  hard confusers    {present}")
    print(f"  synthetic clips   "
          f"{int(df.group_id.str.startswith('synth_').sum())} "
          f"(silence / noise / taps — no ESC-50 category)")

    for tier in ("split_tier1", "split_tier2"):
        print(f"\n  {tier}:")
        t = (df.groupby([tier, "label"]).size().unstack(fill_value=0))
        for sp in t.index:
            row = t.loc[sp]
            print(f"    {sp:10s} drone {row.get('drone',0):5d}   "
                  f"background {row.get('background',0):6d}")

    print(f"\n  written to {OUT.relative_to(ROOT)}")
    print("  validation PASSED — no group spans a split\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
