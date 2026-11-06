"""Photo pipeline: choose an engine, detect, then estimate room size and capacity.

Framework-free on purpose (no FastAPI) so it can be unit tested.
  * engine "classic" -> colour detector from the fullstack project (default for photos)
  * engine "yolo"    -> your trained model
Video never goes through here; it always uses YOLO (see detector.process_video).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from . import config
from .classic.capacity import BoxSize, fill_status, room_capacity
from .classic.room_estimator import DEFAULT_HFOV_DEG, estimate_room
from .classic_detector import ClassicDetector
from .detector import BoxDetector
from .entities import Detection

ENGINES = ("auto", "classic", "yolo")


class EngineUnavailableError(RuntimeError):
    """The requested engine cannot run (missing model or missing libraries)."""


@dataclass
class PhotoParams:
    # YOLO settings
    conf: float = config.PHOTO_CONF
    iou: float = config.PHOTO_IOU
    imgsz: int = config.PHOTO_IMGSZ
    # colour-detector settings
    a_thr: int = 134
    seam: int = 12
    min_dist: int = 7
    # carton size (cm) and camera, used to estimate the room from the photo
    box_h: float = 25.0
    box_w: float = 30.0
    box_b: float = 35.0
    hfov: float = DEFAULT_HFOV_DEG


@dataclass
class PhotoResult:
    engine: str
    detections: List[Detection]
    inference_ms: float
    width: int
    height: int
    room: Optional[dict]
    capacity: Optional[dict]


def resolve_engine(requested: str) -> str:
    if requested not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}")
    if requested == "auto":
        return "yolo" if config.PHOTO_ENGINE == "yolo" else "classic"
    return requested


def analyze_photo(
    image_bgr: np.ndarray,
    engine: str,
    yolo: BoxDetector,
    classic: ClassicDetector,
    params: PhotoParams,
) -> PhotoResult:
    engine = resolve_engine(engine)
    if engine == "yolo":
        if not yolo.loaded:
            raise EngineUnavailableError(yolo.load_error or "The YOLO model is not loaded.")
        detections, ms = yolo.predict_image(image_bgr, params.conf, params.iou, params.imgsz)
    else:
        if not classic.available:
            raise EngineUnavailableError(classic.load_error or "The colour detector is not available.")
        detections, ms = classic.predict_image(image_bgr, params.a_thr, params.seam, params.min_dist)

    height, width = image_bgr.shape[:2]
    size = BoxSize(params.box_h, params.box_w, params.box_b)
    estimate = estimate_room(
        [(d.x1, d.y1, d.x2, d.y2) for d in detections], width, height, size, params.hfov
    )
    room = capacity = None
    if estimate is not None:
        room = asdict(estimate)
        cap = room_capacity(estimate.length_m, estimate.width_m, estimate.height_m, size, estimate.usable)
        status = fill_status(len(detections), cap["capacity"])
        capacity = {
            "floor_area_m2": cap["floor_area_m2"],
            "usable_area_m2": cap["usable_area_m2"],
            "boxes_per_layer": cap["boxes_per_layer"],
            "layers": cap["layers"],
            "capacity": cap["capacity"],
            "detected": status["detected"],
            "fill_pct": round(status["fill_pct"], 2),
            "remaining": status["remaining"],
        }
    return PhotoResult(engine, detections, round(ms, 1), width, height, room, capacity)
