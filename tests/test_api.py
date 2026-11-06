"""API tests. Skipped automatically if FastAPI / httpx are not installed."""
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from backend.detector import BoxDetector  # noqa: E402
from backend.main import create_app  # noqa: E402
from tests.fakes import FakeModel  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    detector = BoxDetector(weights, model_factory=lambda p: FakeModel())
    detector.load()
    with TestClient(create_app(detector)) as c:
        yield c


@pytest.fixture
def no_model_client(tmp_path):
    detector = BoxDetector(tmp_path / "missing.pt", model_factory=lambda p: FakeModel())
    with TestClient(create_app(detector)) as c:
        yield c


def jpeg_bytes():
    ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), 100, np.uint8))
    return buf.tobytes()


def video_bytes(tmp_path):
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    for i in range(15):
        writer.write(np.full((96, 128, 3), i * 10, np.uint8))
    writer.release()
    return path.read_bytes()


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True and body["classic_available"] is True
    assert body["status"] == "ok" and body["photo_engine"] in ("classic", "yolo")


def test_health_without_model(no_model_client):
    body = no_model_client.get("/health").json()
    assert body["model_loaded"] is False and "not found" in body["detail"]
    assert body["status"] == "partial"  # photos still work with the colour detector


def test_yolo_photo_without_model_is_503(no_model_client):
    r = no_model_client.post("/predict/image", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")},
                             data={"engine": "yolo"})
    assert r.status_code == 503


def test_photo_works_without_model_using_colour_detector(no_model_client):
    r = no_model_client.post("/predict/image", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["engine"] == "classic"


def test_video_without_model_is_503(no_model_client, tmp_path):
    r = no_model_client.post("/predict/video", files={"file": ("v.mp4", video_bytes(tmp_path), "video/mp4")})
    assert r.status_code == 503


def test_predict_image(client):
    r = client.post("/predict/image", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")},
                    data={"engine": "yolo", "conf": 0.5, "iou": 0.4, "imgsz": 640})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "yolo" and body["count"] == 2 and len(body["detections"]) == 2
    assert body["annotated_image_b64"]
    assert body["room"] is None  # needs at least 5 boxes


def test_predict_image_annotated_returns_jpeg(client):
    r = client.post("/predict/image/annotated", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")},
                    data={"engine": "yolo"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg" and r.headers["x-box-count"] == "2"
    assert r.headers["x-engine"] == "yolo"


def test_rejects_bad_inputs(client):
    assert client.post("/predict/image", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/predict/image", files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 400
    r = client.post("/predict/image", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}, data={"imgsz": 123})
    assert r.status_code == 422
    r = client.post("/predict/image", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}, data={"conf": 5})
    assert r.status_code == 422
    r = client.post("/predict/image", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}, data={"engine": "magic"})
    assert r.status_code == 422


def test_video_job_flow(client, tmp_path):
    r = client.post("/predict/video", files={"file": ("v.mp4", video_bytes(tmp_path), "video/mp4")},
                    data={"track": "true", "frame_stride": 1})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    for _ in range(60):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    assert status["result"]["max_count"] == 2 and status["result"]["unique_count"] == 2

    assert client.get(f"/jobs/{job_id}/video").status_code == 200
    csv_text = client.get(f"/jobs/{job_id}/report.csv").text
    assert csv_text.startswith("frame_index,time_sec,box_count")
    assert client.get("/jobs/doesnotexist").status_code == 404
