"""Tests for the photo pipeline: engine choice, colour detector wrapper, room and capacity."""
import cv2
import numpy as np
import pytest

from backend import config
from backend.classic.counter_classic import count_boxes
from backend.classic_detector import ClassicDetector
from backend.detector import BoxDetector
from backend.photo_service import EngineUnavailableError, PhotoParams, analyze_photo, resolve_engine
from tests.fakes import EightBoxModel, FakeModel


def orange_scene(n=12):
    """Grey background with orange-red 'cartons' laid out in a grid."""
    img = np.full((400, 700, 3), 110, np.uint8)
    for i in range(n):
        x, y = 30 + (i % 6) * 105, 80 + (i // 6) * 140
        cv2.rectangle(img, (x, y), (x + 85, y + 110), (35, 95, 225), -1)  # BGR orange-red
        cv2.line(img, (x, y + 55), (x + 85, y + 55), (20, 60, 150), 3)
    return img


def make_yolo(tmp_path, model):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    d = BoxDetector(weights, model_factory=lambda p: model)
    assert d.load()
    return d


def test_resolve_engine():
    assert resolve_engine("classic") == "classic"
    assert resolve_engine("yolo") == "yolo"
    assert resolve_engine("auto") == ("yolo" if config.PHOTO_ENGINE == "yolo" else "classic")
    with pytest.raises(ValueError):
        resolve_engine("magic")


def test_classic_wrapper_matches_original_function():
    classic = ClassicDetector()
    assert classic.available
    img = orange_scene()
    dets, ms = classic.predict_image(img)
    _, _, original = count_boxes(img)
    assert len(dets) == len(original) and len(dets) > 0
    assert all(d.confidence is None and d.class_name == "box" for d in dets)
    assert all(0 <= d.x1 < d.x2 <= img.shape[1] + 1 for d in dets)


def test_auto_uses_colour_detector_without_any_model(tmp_path):
    yolo = BoxDetector(tmp_path / "missing.pt")  # no model at all
    result = analyze_photo(orange_scene(), "auto", yolo, ClassicDetector(), PhotoParams())
    assert result.engine == "classic" and len(result.detections) > 0


def test_yolo_engine_with_room_and_capacity(tmp_path):
    yolo = make_yolo(tmp_path, EightBoxModel())
    img = np.zeros((300, 600, 3), np.uint8)
    result = analyze_photo(img, "yolo", yolo, ClassicDetector(), PhotoParams())
    assert result.engine == "yolo" and len(result.detections) == 8
    assert result.room is not None and result.room["boxes_used"] == 8
    assert result.capacity is not None
    assert result.capacity["detected"] == 8
    assert 0 <= result.capacity["fill_pct"]


def test_too_few_boxes_gives_no_room(tmp_path):
    yolo = make_yolo(tmp_path, FakeModel())  # only 2 boxes
    result = analyze_photo(np.zeros((200, 200, 3), np.uint8), "yolo", yolo, ClassicDetector(), PhotoParams())
    assert len(result.detections) == 2
    assert result.room is None and result.capacity is None


def test_yolo_engine_without_model_raises(tmp_path):
    yolo = BoxDetector(tmp_path / "missing.pt")
    with pytest.raises(EngineUnavailableError):
        analyze_photo(np.zeros((50, 50, 3), np.uint8), "yolo", yolo, ClassicDetector(), PhotoParams())
