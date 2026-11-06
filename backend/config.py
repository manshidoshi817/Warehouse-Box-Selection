"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- model -------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best.pt"))

# --- engines -----------------------------------------------------------------
# Photos use the colour detector by default (the logic from box-counter-fullstack).
# Video always uses the YOLO model. Set PHOTO_ENGINE=yolo to send photos to YOLO as well.
PHOTO_ENGINE = os.getenv("PHOTO_ENGINE", "classic")  # classic | yolo

# --- detection defaults (YOLO) -----------------------------------------------
# Photos: dense scenes with many small boxes -> lower confidence, bigger inference size.
PHOTO_CONF = float(os.getenv("PHOTO_CONF", "0.25"))
PHOTO_IOU = float(os.getenv("PHOTO_IOU", "0.5"))
PHOTO_IMGSZ = int(os.getenv("PHOTO_IMGSZ", "960"))
# Video: conveyor belts with a few large boxes.
VIDEO_CONF = float(os.getenv("VIDEO_CONF", "0.5"))
VIDEO_IOU = float(os.getenv("VIDEO_IOU", "0.4"))
VIDEO_IMGSZ = int(os.getenv("VIDEO_IMGSZ", "640"))
IMGSZ_CHOICES = (320, 480, 640, 800, 960, 1280)
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "1000"))

# --- uploads -----------------------------------------------------------------
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "15"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "200"))
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# --- background video jobs ---------------------------------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
VIDEO_WORKERS = int(os.getenv("VIDEO_WORKERS", "1"))

# --- http --------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
