"""Image / video helpers (OpenCV + ffmpeg). No web-framework imports."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .entities import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOX_COLOR = (60, 200, 60)  # BGR green


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR image or raise ValueError."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The uploaded file is not a readable image.")
    return img


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode the annotated image.")
    return buf.tobytes()


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    show_conf: Optional[bool] = None,
    show_ids: bool = False,
) -> np.ndarray:
    """Return a copy of ``image`` with boxes and a count banner drawn on it."""
    out = image.copy()
    h, w = out.shape[:2]
    short = min(h, w)
    thickness = max(1, round(short / 400))
    scale = max(0.4, short / 1100)
    if show_conf is None:
        show_conf = len(detections) <= 60  # avoid label clutter on dense scenes

    for d in detections:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, _BOX_COLOR, thickness)

        parts = []
        if show_ids and d.track_id is not None:
            parts.append(f"#{d.track_id}")
        if show_conf and d.confidence is not None:
            parts.append(f"{d.confidence:.2f}")
        if parts:
            label = " ".join(parts)
            fs = scale * 0.55
            (tw, th), _ = cv2.getTextSize(label, _FONT, fs, 1)
            y = max(p1[1], th + 4)
            cv2.rectangle(out, (p1[0], y - th - 4), (p1[0] + tw + 4, y), _BOX_COLOR, -1)
            cv2.putText(out, label, (p1[0] + 2, y - 3), _FONT, fs, (0, 0, 0), 1, cv2.LINE_AA)

    banner = f"Boxes: {len(detections)}"
    fs = scale * 0.9
    (tw, th), _ = cv2.getTextSize(banner, _FONT, fs, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(out, banner, (10, th + 10), _FONT, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264/yuv420p so browsers can play the result. Returns success."""
    exe = _ffmpeg_exe()
    if not exe:
        return False
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src), "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-movflags", "+faststart", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=3600)
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    except Exception:
        return False
