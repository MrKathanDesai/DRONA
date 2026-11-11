#!/usr/bin/env python3
"""Cache log-mel features for every usable clip in the manifest.

    python scripts/make_features.py

Writes data/features/{X.npy, index.parquet, config.json}. Row i of X is row i
of index.parquet. config.json records the feature config and a hash of the
manifest, so a stale cache is detected instead of silently reused.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from dronedet.features import audio as A
from dronedet.paths import DATA_DIR, resolve

MANIFEST = DATA_DIR / "manifest.parquet"
OUT = DATA_DIR / "features"


def manifest_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:12]


def _one(rel_path: str, start_s: float | None) -> np.ndarray:
    return A.clip_features(resolve(rel_path), A.DEFAULT, start_s)


def main() -> int:
    df = pd.read_parquet(MANIFEST)
    df = df[df.exclude_reason == ""].reset_index(drop=True)
    cfg = A.DEFAULT

    # A cache without config.json was made by code we no longer have.
    # Keep it aside rather than overwrite it blind.
    if (OUT / "X.npy").exists() and not (OUT / "config.json").exists():
        legacy = DATA_DIR / "features_legacy"
        print(f"  moving unversioned cache to {legacy}")
        shutil.move(str(OUT), str(legacy))
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"  {len(df)} clips -> ({cfg.n_mels}, {cfg.n_frames}) log-mel ...")
    with ProcessPoolExecutor() as pool:
        feats = list(pool.map(_one, df.path, df.start_s, chunksize=64))
    X = np.stack(feats).astype(np.float32)

    np.save(OUT / "X.npy", X)
    df.to_parquet(OUT / "index.parquet", index=False)
    (OUT / "config.json").write_text(json.dumps({
        "config": asdict(cfg),
        "fingerprint": cfg.fingerprint(),
        "manifest_hash": manifest_hash(df),
        "shape": list(X.shape),
    }, indent=2))

    print(f"  X {X.shape} {X.dtype}, {X.nbytes / 1e6:.0f} MB")
    print(f"  dB range {X.min():.1f} .. {X.max():.1f}, "
          f"non-finite {int((~np.isfinite(X)).sum())}")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
