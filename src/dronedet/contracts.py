from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Label = Literal["drone", "background"]
Split = Literal["train", "val", "test", "excluded"]


@dataclass(frozen=True)
class Clip:
    path: str
    label: Label
    group_id: str
    source: str
    drone_model: Optional[str]
    bg_category: Optional[str]
    sample_rate: int
    duration: float
