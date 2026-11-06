"""Colour + watershed carton finder for photos (logic from box-counter-fullstack).

This wraps ``backend/classic/counter_classic.py`` WITHOUT changing it. The scaling of the
boxes back to the input size is the same as in the original fullstack project.
Works best on orange/red cartons; it is not a trained model.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .entities import Detection

logger = logging.getLogger(__name__)


class ClassicDetector:
    name = "classic"

    def __init__(self) -> None:
        self.load_error: Optional[str] = None
        self._count_boxes = None
        try:
            from .classic.counter_classic import count_boxes

            self._count_boxes = count_boxes
        except Exception as exc:  # scipy / scikit-image missing
            self.load_error = f"The colour detector needs scipy and scikit-image ({exc})."
            logger.warning(self.load_error)

    @property
    def available(self) -> bool:
        return self._count_boxes is not None

    def predict_image(
        self, image_bgr: np.ndarray, a_thr: int = 134, seam: int = 12, min_dist: int = 7
    ) -> Tuple[List[Detection], float]:
        if not self.available:
            raise RuntimeError(self.load_error or "Colour detector is not available.")
        start = time.perf_counter()
        resized, _mask, boxes = self._count_boxes(image_bgr, a_thr=a_thr, seam=seam, min_dist=min_dist)
        # count_boxes works on a copy resized to <=960px wide: scale back to the input size
        scale = image_bgr.shape[1] / resized.shape[1]
        detections = [
            Detection(
                x1=float(x0 * scale), y1=float(y0 * scale),
                x2=float((x1 + 1) * scale), y2=float((y1 + 1) * scale),
                confidence=None, class_id=0, class_name="box",
            )
            for x0, y0, x1, y1 in boxes
        ]
        return detections, (time.perf_counter() - start) * 1000
