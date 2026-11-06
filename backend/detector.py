"""YOLO wrapper: single-image prediction and full-video processing.

The Ultralytics import is lazy so the rest of the code base (and the unit tests)
work without it. Pass ``model_factory`` to inject a fake model in tests.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .entities import Detection, FrameCount, VideoSummary
from .utils import draw_detections, to_h264

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but no model is loaded."""


def _np(tensor) -> np.ndarray:
    return tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)


class BoxDetector:
    def __init__(self, model_path: Path, model_factory: Optional[Callable[[Path], object]] = None):
        self.model_path = Path(model_path)
        self._factory = model_factory or self._default_factory
        self._model = None
        self._lock = threading.Lock()  # one Ultralytics predictor must not be shared across threads
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _default_factory(path: Path):
        from ultralytics import YOLO  # lazy import

        return YOLO(str(path))

    def load(self) -> bool:
        if not self.model_path.exists():
            self.load_error = (
                f"Model file not found: {self.model_path}. "
                "Train a model and copy best.pt to models/best.pt (see README)."
            )
            logger.warning(self.load_error)
            return False
        try:
            self._model = self._factory(self.model_path)
            self.load_error = None
            logger.info("Loaded model from %s", self.model_path)
            return True
        except Exception as exc:  # pragma: no cover - depends on the model file
            self._model = None
            self.load_error = f"Could not load model: {exc}"
            logger.exception(self.load_error)
            return False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        return list(names)

    def _require_model(self) -> None:
        if not self.loaded:
            raise ModelNotLoadedError(self.load_error or "Model is not loaded.")

    # ------------------------------------------------------------------ parsing
    def _parse_result(self, result, with_ids: bool = False) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _np(boxes.xyxy)
        conf = _np(boxes.conf)
        cls = _np(boxes.cls).astype(int)
        ids = None
        if with_ids and getattr(boxes, "id", None) is not None:
            ids = _np(boxes.id).astype(int)
        names = getattr(result, "names", None) or {}
        if not isinstance(names, dict):
            names = dict(enumerate(names))

        detections: List[Detection] = []
        for i in range(len(xyxy)):
            detections.append(
                Detection(
                    x1=float(xyxy[i][0]), y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]), y2=float(xyxy[i][3]),
                    confidence=float(conf[i]),
                    class_id=int(cls[i]),
                    class_name=str(names.get(int(cls[i]), "box")),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return detections

    # ------------------------------------------------------------------ images
    def predict_image(
        self, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int
    ) -> Tuple[List[Detection], float]:
        self._require_model()
        start = time.perf_counter()
        with self._lock:
            results = self._model.predict(
                image_bgr, conf=conf, iou=iou, imgsz=imgsz,
                agnostic_nms=True, max_det=config.MAX_DETECTIONS, verbose=False,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._parse_result(results[0]), elapsed_ms

    # ------------------------------------------------------------------ video
    def process_video(
        self,
        in_path: Path,
        out_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
        frame_stride: int = 1,
        track: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> VideoSummary:
        """Annotate a video and collect per-frame box counts.

        ``track=True`` uses ByteTrack so each physical box gets a stable ID, which
        gives a *unique* box count (useful for conveyor belts). Without tracking you
        get the number of boxes visible in each frame.
        """
        self._require_model()
        in_path, out_path = Path(in_path), Path(out_path)
        stride = max(1, int(frame_stride))

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise ValueError("The uploaded file is not a readable video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        raw_path = out_path.with_name(out_path.stem + ".raw.mp4")
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open the video writer.")

        # Private model instance: tracker state must not leak into other requests.
        model = self._factory(self.model_path)

        per_frame: List[FrameCount] = []
        unique_ids = set()
        last: List[Detection] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    kwargs = dict(
                        conf=conf, iou=iou, imgsz=imgsz,
                        agnostic_nms=True, max_det=config.MAX_DETECTIONS, verbose=False,
                    )
                    if track:
                        res = model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)
                    else:
                        res = model.predict(frame, **kwargs)
                    last = self._parse_result(res[0], with_ids=track)
                    for d in last:
                        if d.track_id is not None:
                            unique_ids.add(d.track_id)
                    per_frame.append(FrameCount(idx, round(idx / fps, 3), len(last)))
                writer.write(draw_detections(frame, last, show_ids=track))
                idx += 1
                if progress_cb and (idx % 5 == 0 or idx == total):
                    progress_cb(idx, total)
        finally:
            cap.release()
            writer.release()

        if not per_frame:
            raw_path.unlink(missing_ok=True)
            raise ValueError("The video contains no readable frames.")

        playable = to_h264(raw_path, out_path)
        if playable:
            raw_path.unlink(missing_ok=True)
        else:
            logger.warning("ffmpeg re-encode failed; keeping the mp4v file (may not play in browsers).")
            raw_path.replace(out_path)

        counts = [f.count for f in per_frame]
        return VideoSummary(
            total_frames=idx,
            frames_processed=len(per_frame),
            fps=round(float(fps), 3),
            width=width,
            height=height,
            max_count=max(counts),
            min_count=min(counts),
            avg_count=round(sum(counts) / len(counts), 2),
            tracking=track,
            unique_count=len(unique_ids) if track else None,
            browser_playable=playable,
            per_frame=per_frame,
        )
