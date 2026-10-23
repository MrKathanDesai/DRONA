from __future__ import annotations

import os
from pathlib import Path

# Manifest paths are stored relative to this directory, so the manifest
# works on any machine. Override with DRONEDET_DATA to keep data elsewhere.
DATA_DIR = Path(os.environ.get(
    "DRONEDET_DATA", Path(__file__).resolve().parents[2] / "data"))


def resolve(rel_path: str) -> Path:
    return DATA_DIR / rel_path
