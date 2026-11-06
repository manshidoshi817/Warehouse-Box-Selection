"""Plain data classes shared by the detector, job store and API (no framework imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float]  # None for the colour detector
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class FrameCount:
    frame_index: int
    time_sec: float
    count: int


@dataclass
class VideoSummary:
    total_frames: int
    frames_processed: int
    fps: float
    width: int
    height: int
    max_count: int
    min_count: int
    avg_count: float
    tracking: bool
    unique_count: Optional[int]
    browser_playable: bool
    per_frame: List[FrameCount] = field(default_factory=list)
