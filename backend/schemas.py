"""Pydantic response models (these define the public API contract)."""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class DetectionOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class ImageParams(BaseModel):
    conf: float
    iou: float
    imgsz: int
    a_thr: int
    seam: int
    min_dist: int


class RoomOut(BaseModel):
    """Room size estimated from the photo (rough: about +/-30%)."""
    length_m: float
    width_m: float
    height_m: float
    usable: float
    camera_distance_m: float
    layers_visible: int
    boxes_used: int
    confidence: str
    notes: List[str]


class CapacityOut(BaseModel):
    floor_area_m2: float
    usable_area_m2: float
    boxes_per_layer: int
    layers: int
    capacity: int
    detected: int
    fill_pct: float
    remaining: int


class ImageResult(BaseModel):
    engine: str
    count: int
    detections: List[DetectionOut]
    image_width: int
    image_height: int
    inference_ms: float
    params: ImageParams
    room: Optional[RoomOut] = None
    capacity: Optional[CapacityOut] = None
    annotated_image_b64: Optional[str] = None


class FrameCountOut(BaseModel):
    frame_index: int
    time_sec: float
    count: int


class VideoResult(BaseModel):
    total_frames: int
    frames_processed: int
    fps: float
    width: int
    height: int
    max_count: int
    min_count: int
    avg_count: float
    tracking: bool
    unique_count: Optional[int] = None
    browser_playable: bool
    per_frame: List[FrameCountOut]
    video_url: str
    csv_url: str


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    error: Optional[str] = None
    result: Optional[VideoResult] = None


class HealthOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str
    classic_available: bool
    photo_engine: str
    detail: Optional[str] = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    classes: List[str]
    size_mb: float
