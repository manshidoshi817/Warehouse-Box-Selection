"""Tests for the framework-free core: detector, utils, job store."""
import cv2
import numpy as np
import pytest

from backend.detector import BoxDetector, ModelNotLoadedError
from backend.jobs import JobStore
from backend.utils import decode_image, draw_detections, encode_jpeg
from tests.fakes import EmptyModel, FakeModel


@pytest.fixture
def weights(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"fake")
    return p


@pytest.fixture
def detector(weights):
    d = BoxDetector(weights, model_factory=lambda path: FakeModel())
    assert d.load()
    return d


def make_video(path, frames=20, size=(128, 96)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8))
    writer.release()


def test_missing_model_reports_error(tmp_path):
    d = BoxDetector(tmp_path / "nope.pt", model_factory=lambda p: FakeModel())
    assert d.load() is False
    assert "not found" in d.load_error
    with pytest.raises(ModelNotLoadedError):
        d.predict_image(np.zeros((10, 10, 3), np.uint8), 0.5, 0.4, 640)


def test_predict_image_counts_boxes(detector):
    img = np.zeros((120, 160, 3), np.uint8)
    dets, ms = detector.predict_image(img, 0.5, 0.4, 640)
    assert len(dets) == 2
    assert dets[0].class_name == "box" and dets[0].track_id is None
    assert ms >= 0


def test_predict_image_empty(weights):
    d = BoxDetector(weights, model_factory=lambda p: EmptyModel())
    d.load()
    dets, _ = d.predict_image(np.zeros((50, 50, 3), np.uint8), 0.5, 0.4, 640)
    assert dets == []


def test_class_names(detector):
    assert detector.class_names == ["box"]


def test_draw_and_codec_roundtrip(detector):
    img = np.full((200, 300, 3), 127, np.uint8)
    dets, _ = detector.predict_image(img, 0.5, 0.4, 640)
    annotated = draw_detections(img, dets)
    assert annotated.shape == img.shape
    assert not np.array_equal(annotated, img)  # something was drawn
    decoded = decode_image(encode_jpeg(annotated))
    assert decoded.shape == img.shape


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"this is not an image")


def test_process_video_counts(detector, tmp_path):
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, frames=20)
    progress = []
    summary = detector.process_video(src, out, 0.5, 0.4, 640, frame_stride=1, track=False,
                                     progress_cb=lambda d, t: progress.append((d, t)))
    assert summary.total_frames == 20
    assert summary.frames_processed == 20
    assert summary.max_count == summary.min_count == 2
    assert summary.avg_count == 2.0
    assert summary.unique_count is None
    assert out.exists() and out.stat().st_size > 0
    assert progress and progress[-1][0] == 20
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 19
    cap.release()


def test_process_video_stride_and_tracking(detector, tmp_path):
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, frames=20)
    summary = detector.process_video(src, out, 0.5, 0.4, 640, frame_stride=5, track=True)
    assert summary.frames_processed == 4  # frames 0, 5, 10, 15
    assert summary.total_frames == 20
    assert summary.tracking is True
    assert summary.unique_count == 2  # the fake tracker reuses ids 1 and 2


def test_process_video_rejects_non_video(detector, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(ValueError):
        detector.process_video(bad, tmp_path / "out.mp4", 0.5, 0.4, 640)


def test_job_store_lifecycle(tmp_path):
    store = JobStore(ttl_seconds=0)
    job = store.create()
    assert store.get(job.id).status == "queued"
    store.update(job.id, status="running", progress=0.5)
    assert store.get(job.id).progress == 0.5
    (tmp_path / job.id).mkdir()
    store.update(job.id, status="done")
    assert store.cleanup(tmp_path) == 1  # ttl 0 -> expired immediately
    assert store.get(job.id) is None
    assert not (tmp_path / job.id).exists()
    store.update("missing", status="x")  # must not raise
