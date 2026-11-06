"""Estimate the room size from ONE photo, using the cartons as a ruler.

Idea (pinhole camera model):
    A carton is `box.height` cm tall in real life. If it appears `h_px` pixels tall,
    its distance from the camera is   Z = f_px * box.height / h_px.
    f_px (focal length in pixels) comes from the camera's horizontal field of view.

From the distances of all detected cartons we get:
    * length  = distance to the farthest detected stack (a CCTV/phone usually looks down the room)
    * width   = width of the scene visible at that far plane: 2 * Z_far * tan(hfov / 2)
    * height  = number of carton layers in the main stack face * carton height

This is a ROUGH estimate (typically within +/-25-40 %). It assumes an upright, roughly level
camera and that the farthest stacks are near the far wall. It is not a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .capacity import BoxSize  # (only change: package-relative import)

BoxXYXY = Tuple[float, float, float, float]

DEFAULT_HFOV_DEG = 70.0    # typical phone main camera / wide CCTV lens
DEFAULT_USABLE = 0.6       # share of floor that holds cartons (rest = aisles); fixed, not asked
MIN_BOXES = 5


@dataclass
class RoomEstimate:
    length_m: float
    width_m: float
    height_m: float
    usable: float
    camera_distance_m: float      # camera -> main stack face
    layers_visible: int
    boxes_used: int
    confidence: str               # "low" | "medium"
    notes: list = field(default_factory=list)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def estimate_room(boxes: Sequence[BoxXYXY], img_w: int, img_h: int, box: BoxSize = BoxSize(),
                  hfov_deg: float = DEFAULT_HFOV_DEG) -> Optional[RoomEstimate]:
    """Return a RoomEstimate, or None when there are too few usable boxes to judge scale."""
    if img_w <= 0 or img_h <= 0:
        return None
    f_px = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    box_h_m = box.height / 100.0

    # 1. keep boxes whose shape looks like an upright carton face (drops merged/split blobs)
    arr = np.array([b for b in boxes], dtype=float).reshape(-1, 4)
    w = arr[:, 2] - arr[:, 0]
    h = arr[:, 3] - arr[:, 1]
    ok = (w > 3) & (h > 3) & (h / np.maximum(w, 1) > 0.35) & (h / np.maximum(w, 1) < 2.5)
    arr, h = arr[ok], h[ok]
    if len(arr) < MIN_BOXES:
        return None

    # 2. distance of every box from the camera, then drop obvious scale outliers
    z = f_px * box_h_m / h
    med = np.median(z)
    keep = (z > med * 0.3) & (z < med * 3.0)
    arr, h, z = arr[keep], h[keep], z[keep]
    if len(arr) < MIN_BOXES:
        return None

    z_main = float(np.median(z))                 # distance to the main stack face
    z_far = float(np.percentile(z, 92))          # farthest stacks ~ far wall (robust to 1-2 outliers)

    # 3. room dimensions
    length = z_far
    width = 2.0 * z_far * math.tan(math.radians(hfov_deg) / 2.0)

    # 4. stack height = layers visible on the main face (boxes at about the same distance)
    face = (z > z_main * 0.8) & (z < z_main * 1.25)
    h_med = float(np.median(h[face])) if face.any() else float(np.median(h))
    ys0, ys1 = arr[face, 1], arr[face, 3]
    # percentiles instead of min/max so one stray box cannot inflate the stack height
    span = float(np.percentile(ys1, 98) - np.percentile(ys0, 2)) if face.any() else h_med
    layers = int(_clamp(round(span / h_med), 1, 30))
    height = layers * box_h_m

    notes = ["Estimated from carton size in the photo (rough: expect roughly +/-30 %)."]
    confidence = "medium" if len(arr) >= 30 else "low"
    if len(arr) < 30:
        notes.append("Few boxes detected, so the scale reference is weak.")
    if z_far / max(z_main, 1e-6) < 1.15:
        notes.append("All boxes look equally far away; room depth may be underestimated.")

    return RoomEstimate(
        length_m=round(_clamp(length, 2, 500), 1),
        width_m=round(_clamp(width, 2, 500), 1),
        height_m=round(_clamp(height, 0.25, 30), 2),
        usable=DEFAULT_USABLE,
        camera_distance_m=round(z_main, 1),
        layers_visible=layers,
        boxes_used=int(len(arr)),
        confidence=confidence,
        notes=notes,
    )
