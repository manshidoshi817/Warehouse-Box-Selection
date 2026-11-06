# Warehouse Box Counter

Upload a **photo or a video** of a warehouse and get the boxes detected and counted.
A **FastAPI** backend does the work and a **Streamlit** app is the user interface.

Photos and video use **separate logic**, so each can be improved without touching the other:

| Input | Engine | Needs a trained model? |
|---|---|---|
| **Photo** | Colour detector (from the box-counter-fullstack project, unchanged) by default, or YOLO if you choose it | No for the colour detector |
| **Video** | YOLO (`models/best.pt`), optional tracking for a unique box count | Yes |

```
 Streamlit UI  ──HTTP──▶  FastAPI backend ─┬─ photos ─▶ colour detector  (or YOLO)  ─▶ room size + capacity
 (frontend/app.py)        (backend/main.py)└─ videos ─▶ YOLO + ByteTrack (background job with progress)
```

## Features

- **Photos:** one or many at once, annotated image, box count, **estimated room size and capacity**, CSV summary.
- **Video:** background processing with a live progress bar, annotated video, per-frame count chart, **total unique boxes** (tracking) and CSV.
- **Engine choice for photos:** Auto (colour detector), colour detector, or YOLO, selectable in the app.
- **Separate settings** for photos and video (photos default to confidence 0.25 / size 960, video to 0.5 / 640).
- **REST API** with automatic docs at `/docs`.
- **Docker** setup and a **training pipeline** that cleans a Roboflow export and produces `best.pt`.

## Which engine for what

- **Colour detector** (`backend/classic/`): a colour + watershed method. It needs no training and works right away, but it
  only recognises **orange/red cartons**. Plain brown cardboard and very dense stacks are missed or mis-boxed, so treat the
  count as approximate.
- **YOLO** (`models/best.pt`): a trained model. It can handle any box type it was trained on, if you give it enough labeled
  photos from your own cameras.

### One model for everything (later)

The colour detector is not a trained model, so it cannot be merged into `best.pt`. To get a single model that handles both
photos and video, train YOLO on a combined dataset (conveyor frames plus labeled photos from your warehouse cameras), then
set `PHOTO_ENGINE=yolo` (or pick "YOLO model" in the app). The video code does not change.

## 1. Quick start (local)

Python 3.10 to 3.12 recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Put your trained weights at **`models/best.pt`** (see section 4 if you don't have them yet), then:

```bash
run_all.bat                     # Windows: opens the API and the UI
# ./run_all.sh                  # macOS / Linux
```

or start each part yourself:

```bash
python -m uvicorn backend.main:app --port 8000     # API  -> http://localhost:8000/docs
python -m streamlit run frontend/app.py            # UI   -> http://localhost:8501
```

If `best.pt` is missing, **photos still work** (colour detector). Video needs the model: add the file and press **Reload model** in the sidebar.

## 2. Docker

```bash
docker compose up --build
```

UI: http://localhost:8501, API: http://localhost:8000/docs. The `models/` folder is mounted into the container, so put `best.pt` there. The image uses CPU PyTorch; for GPU hosts, remove the CPU wheel line in the `Dockerfile`.

## 3. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status of both engines (`ok`, `partial` or `unavailable`) |
| POST | `/model/reload` | Load `models/best.pt` without restarting |
| GET | `/model/info` | Class names and model file size |
| POST | `/predict/image` | JSON: engine used, count, boxes, room estimate, capacity, annotated image (base64) |
| POST | `/predict/image/annotated` | Annotated JPEG directly (`X-Box-Count` and `X-Engine` headers) |
| POST | `/predict/video` | Queue a video (always YOLO), returns a `job_id` (HTTP 202) |
| GET | `/jobs/{job_id}` | Progress and, when done, the summary |
| GET | `/jobs/{job_id}/video` | Annotated MP4 |
| GET | `/jobs/{job_id}/report.csv` | Per-frame box counts |

**Photo form fields:** `engine` (`auto`, `classic`, `yolo`); YOLO: `conf` (default 0.25), `iou` (0.5), `imgsz` (960);
colour detector: `a_thr` (134), `seam` (12), `min_dist` (7); room estimate: `box_h`, `box_w`, `box_b` (cm, default 25 / 30 / 35) and `hfov` (camera field of view, default 70).
**Video form fields:** `conf` (0.5), `iou` (0.4), `imgsz` (640), `frame_stride` (1 to 30), `track` (`true`/`false`).

```bash
# photo
curl -F "file=@warehouse.jpg" http://localhost:8000/predict/image                 # colour detector
curl -F "file=@warehouse.jpg" -F engine=yolo -F imgsz=960 http://localhost:8000/predict/image   # YOLO

# photo -> annotated JPEG file
curl -F "file=@warehouse.jpg" http://localhost:8000/predict/image/annotated -o result.jpg

# video
curl -F "file=@belt.mp4" -F track=true http://localhost:8000/predict/video
curl http://localhost:8000/jobs/<job_id>
curl http://localhost:8000/jobs/<job_id>/video -o annotated.mp4
```

Python:

```python
import requests
r = requests.post("http://localhost:8000/predict/image",
                  files={"file": open("warehouse.jpg", "rb")}, data={"engine": "auto"})
print(r.json()["count"], r.json()["room"])
```

## 4. Train your own model (`best.pt`)

1. **Get frames** from your videos: `python scripts/extract_frames.py video.mp4 --every 10 --out frames`
2. **Label** them in Roboflow (Object Detection, one class named `box`). Draw a box around every box you want counted, and be consistent. Export as **YOLOv8** (zip).
3. **Train** (use Google Colab with a T4 GPU for speed):

```bash
pip install ultralytics pyyaml
python -m training.train --data my_roboflow_export.zip --epochs 100
```

This script cleans the export first: it keeps only the `box` class, removes tiny noisy boxes, converts polygons to boxes, creates a validation split if the export has none, trains YOLOv8, prints mAP, and **copies the result to `models/best.pt`**. Download that file from Colab and put it in `models/`.

Useful options: `--max-boxes 80` (drop images with an extreme number of labels), `--min-area 0.003` (smallest box kept, as a fraction of the image), `--model yolov8s.pt` (bigger model), `--imgsz 960` (better for small boxes).

### Getting good accuracy

- **Match your camera.** A model only works well on views like the ones it was trained on. Conveyor-belt frames will not teach it dense pallet stacks seen from a ceiling camera. Label images from the cameras you will actually use.
- **More, varied data.** Aim for 100+ labeled images (200 to 500 is better) with different lighting, angles and box arrangements.
- **Label consistently.** Decide the rule up front (for example "every visible carton face") and apply it to every image. Inconsistent labels cause overlapping or missing detections.
- **Check mAP50** printed after training. Below about 0.6 usually means too little or too inconsistent data.

## 5. Configuration

Environment variables (see `.env.example`): `MODEL_PATH`, `PHOTO_ENGINE`, `PHOTO_CONF`, `PHOTO_IOU`, `PHOTO_IMGSZ`, `VIDEO_CONF`, `VIDEO_IOU`, `VIDEO_IMGSZ`, `MAX_IMAGE_MB`, `MAX_VIDEO_MB`, `MAX_DETECTIONS`, `JOB_TTL_SECONDS`, `VIDEO_WORKERS`, `CORS_ORIGINS`, and `API_URL` for the Streamlit app.

## 6. Project layout

```
backend/
  main.py          FastAPI routes, validation, background jobs
  detector.py      YOLO wrapper: image + video inference, tracking
  classic_detector.py  adapter for the colour detector (photos)
  photo_service.py engine choice + room estimate + capacity for photos
  classic/         colour detector, room estimator and capacity maths (from box-counter-fullstack)
  jobs.py          thread-safe job store with expiry
  utils.py         drawing, image codecs, H.264 re-encoding
  schemas.py       response models (API contract)
  config.py        settings from environment variables
frontend/app.py    Streamlit UI
training/          dataset cleaning + training script
scripts/           frame extraction
tests/             unit tests (fake model, no GPU needed)
models/            put best.pt here
```

## 7. Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a fake model, so they need no GPU and no trained weights.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| UI says "Cannot reach the API" | Start the backend, or set `API_URL` |
| "Model file not found" | Place `best.pt` in `models/`, press **Reload model** |
| Many overlapping boxes | Raise confidence, lower the IoU slider, retrain on cleaner labels |
| Boxes missed | Lower confidence, raise inference size, add similar images to training. For plain brown cartons, the colour detector cannot help: use YOLO |
| Photo count is far too low | The colour detector only sees orange/red cartons. Switch the photo engine to YOLO |
| Annotated video won't play | ffmpeg re-encode failed; download the file and open it in a media player |
| Video upload rejected | Limit is 200 MB by default (`MAX_VIDEO_MB`, and `server.maxUploadSize` in `.streamlit/config.toml`) |

## Limitations

- Counts are of **visible** boxes. Boxes hidden behind others cannot be counted.
- Room size and capacity are rough estimates (about +/-30%), computed from carton size in the photo.
- Video jobs are kept in memory and expire after an hour; restarting the API clears them.
- There is no login. Do not expose the API to the internet without adding authentication.
