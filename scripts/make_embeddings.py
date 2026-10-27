#!/usr/bin/env python3
"""Cache frozen MobileNetV1 embeddings for every usable clip in the manifest.

    python scripts/make_embeddings.py

Writes data/embeddings/{E.npy, tags.npy, index.parquet, config.json}. Row i
of E is row i of index.parquet. Nothing is trained; this is one forward pass
per clip (~1 min on Apple-silicon GPU).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from dronedet.features import audio as A
from dronedet.models import pretrained as P
from dronedet.paths import DATA_DIR, resolve
from make_features import manifest_hash

MANIFEST = DATA_DIR / "manifest.parquet"
OUT = DATA_DIR / "embeddings"


def _load(rel_path: str, start_s: float | None) -> np.ndarray:
    return A.load_clip(resolve(rel_path), start_s)


def main() -> int:
    df = pd.read_parquet(MANIFEST)
    df = df[df.exclude_reason == ""].reset_index(drop=True)
    print(f"  loading {len(df)} clips ...", flush=True)
    with ProcessPoolExecutor() as pool:
        clips = list(pool.map(_load, df.path, df.start_s, chunksize=64))

    enc = P.Encoder()
    print(f"  encoding on {enc.device} ...", flush=True)
    E, T = enc(clips)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "E.npy", E.astype(np.float32))
    np.save(OUT / "tags.npy", T.astype(np.float16))
    df.to_parquet(OUT / "index.parquet", index=False)
    (OUT / "config.json").write_text(json.dumps({
        "encoder": P.CHECKPOINT.name,
        "target_rms": P.TARGET_RMS,
        "manifest_hash": manifest_hash(df),
        "shape": list(E.shape),
    }, indent=2))
    print(f"  E {E.shape}, non-finite {int((~np.isfinite(E)).sum())}")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
