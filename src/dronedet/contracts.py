"""Shared types. Every module speaks in these; nothing reaches inside anything else."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Label = Literal["drone", "background"]
Split = Literal["train", "val", "test", "excluded"]


@dataclass(frozen=True)
class Clip:
    """One labelled audio clip, as the manifest describes it.

    `group_id` is the leakage unit — the parent recording a clip was sliced
    from. Clips sharing a group_id are near-duplicates and must never be
    separated across a split. This is the single most important field here.
    """
    path: str
    label: Label
    group_id: str
    source: str                      # which dataset it came from
    drone_model: Optional[str]       # bebop / membo / None
    bg_category: Optional[str]       # ESC-50 category for negatives
    sample_rate: int
    duration: float
