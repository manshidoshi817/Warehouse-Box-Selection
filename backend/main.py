"""FastAPI application: image and video box detection.

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import base64
import csv
import io
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import __version__, config
from .classic.room_estimator import DEFAULT_HFOV_DEG
from .classic_detector import ClassicDetector
from .detector import BoxDetector, ModelNotLoadedError
from .jobs import Job, JobStore
from .photo_service import EngineUnavailableError, PhotoParams, PhotoResult, analyze_photo, resolve_engine
from .schemas import HealthOut, ImageResult, JobCreated, JobStatus, ModelInfo
from .utils import decode_image, draw_detections, encode_jpeg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("box-counter")

router = APIRouter()

ConfForm = Annotated[float, Form(ge=0.01, le=0.99)]
IouForm = Annotated[float, Form(ge=0.05, le=0.95)]
StrideForm = Annotated[int, Form(ge=1, le=30)]


# --------------------------------------------------------------------------- dependencies
def get_detector(request: Request) -> BoxDetector:
    """The YOLO detector. Does not raise; callers decide whether the model is required."""
    detector: BoxDetector = request.app.state.detector
    if not detector.loaded:
        detector.load()  # lets you drop best.pt into models/ without restarting
    return detector


def require_model(detector: BoxDetector) -> BoxDetector:
    if not detector.loaded:
        raise HTTPException(status_code=503, detail=detector.load_error or "Model is not loaded.")
    return detector


def get_classic(request: Request) -> ClassicDetector:
    return request.app.state.classic


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def get_executor(request: Request) -> ThreadPoolExecutor:
    return request.app.state.executor


# --------------------------------------------------------------------------- helpers
def _check_imgsz(imgsz: int) -> None:
    if imgsz not in config.IMGSZ_CHOICES:
        raise HTTPException(status_code=422, detail=f"imgsz must be one of {list(config.IMGSZ_CHOICES)}")


def _check_extension(filename: Optional[str], allowed: set) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: {sorted(allowed)}",
        )
    return ext


async def _read_limited(upload: UploadFile, max_mb: int) -> bytes:
    limit = max_mb * 1024 * 1024
    buf = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail=f"File too large (limit {max_mb} MB).")
    if not buf:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return bytes(buf)


async def _save_limited(upload: UploadFile, dest: Path, max_mb: int) -> None:
    limit = max_mb * 1024 * 1024
    size = 0
    too_large = False
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                too_large = True
                break
            fh.write(chunk)
    if too_large:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File too large (limit {max_mb} MB).")
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")


def _job_to_status(job: Job) -> dict:
    result = None
    if job.summary is not None:
        result = asdict(job.summary)
        result["video_url"] = f"/jobs/{job.id}/video"
        result["csv_url"] = f"/jobs/{job.id}/report.csv"
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": round(job.progress, 3),
        "message": job.message,
        "error": job.error,
        "result": result,
    }


def _run_video_job(detector, jobs, job_id, in_path, out_path, conf, iou, imgsz, stride, track) -> None:
    """Runs in the worker thread pool."""
    jobs.update(job_id, status="running", message="Processing video")

    def on_progress(done: int, total: int) -> None:
        if total > 0:
            jobs.update(job_id, progress=min(done / total, 0.99), message=f"Frame {done}/{total}")
        else:
            jobs.update(job_id, message=f"Frame {done}")

    try:
        summary = detector.process_video(
            in_path, out_path, conf=conf, iou=iou, imgsz=imgsz,
            frame_stride=stride, track=track, progress_cb=on_progress,
        )
        jobs.update(job_id, status="done", progress=1.0, summary=summary, message="Done")
    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        jobs.update(job_id, status="failed", error=str(exc), message="Failed")
    finally:
        Path(in_path).unlink(missing_ok=True)


@dataclass
class PhotoRequest:
    engine: str
    params: PhotoParams


def photo_form(
    engine: str = Form("auto"),
    conf: ConfForm = config.PHOTO_CONF,
    iou: IouForm = config.PHOTO_IOU,
    imgsz: int = Form(config.PHOTO_IMGSZ),
    a_thr: int = Form(134, ge=100, le=160),
    seam: int = Form(12, ge=2, le=40),
    min_dist: int = Form(7, ge=2, le=30),
    box_h: float = Form(25.0, gt=0, le=500),
    box_w: float = Form(30.0, gt=0, le=500),
    box_b: float = Form(35.0, gt=0, le=500),
    hfov: float = Form(DEFAULT_HFOV_DEG, ge=40, le=120),
) -> PhotoRequest:
    """Form fields shared by the photo endpoints."""
    _check_imgsz(imgsz)
    try:
        resolve_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PhotoRequest(
        engine,
        PhotoParams(conf=conf, iou=iou, imgsz=imgsz, a_thr=a_thr, seam=seam, min_dist=min_dist,
                    box_h=box_h, box_w=box_w, box_b=box_b, hfov=hfov),
    )


async def _analyze(image, req: PhotoRequest, detector: BoxDetector, classic: ClassicDetector) -> PhotoResult:
    try:
        return await run_in_threadpool(analyze_photo, image, req.engine, detector, classic, req.params)
    except (EngineUnavailableError, ModelNotLoadedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _photo_payload(result: PhotoResult, params: PhotoParams) -> dict:
    return {
        "engine": result.engine,
        "count": len(result.detections),
        "detections": [asdict(d) for d in result.detections],
        "image_width": result.width,
        "image_height": result.height,
        "inference_ms": result.inference_ms,
        "params": {"conf": params.conf, "iou": params.iou, "imgsz": params.imgsz,
                   "a_thr": params.a_thr, "seam": params.seam, "min_dist": params.min_dist},
        "room": result.room,
        "capacity": result.capacity,
        "annotated_image_b64": None,
    }


# --------------------------------------------------------------------------- routes: meta
@router.get("/", tags=["meta"])
def root():
    return {"name": "Warehouse Box Counter API", "version": __version__, "docs": "/docs"}


@router.get("/health", response_model=HealthOut, tags=["meta"])
def health(request: Request):
    detector: BoxDetector = request.app.state.detector
    classic: ClassicDetector = request.app.state.classic
    if detector.loaded and classic.available:
        status = "ok"
    elif detector.loaded or classic.available:
        status = "partial"
    else:
        status = "unavailable"
    return HealthOut(
        status=status,
        model_loaded=detector.loaded,
        model_path=str(detector.model_path),
        classic_available=classic.available,
        photo_engine=config.PHOTO_ENGINE,
        detail=None if detector.loaded else detector.load_error,
    )


@router.post("/model/reload", response_model=HealthOut, tags=["meta"])
def reload_model(request: Request):
    request.app.state.detector.load()
    return health(request)


@router.get("/model/info", response_model=ModelInfo, tags=["meta"])
def model_info(detector: BoxDetector = Depends(get_detector)):
    require_model(detector)
    return ModelInfo(
        model_path=str(detector.model_path),
        classes=detector.class_names,
        size_mb=round(detector.model_path.stat().st_size / 1024 / 1024, 2),
    )


# --------------------------------------------------------------------------- routes: images
@router.post("/predict/image", response_model=ImageResult, tags=["prediction"])
async def predict_image(
    file: UploadFile = File(...),
    include_image: bool = Form(True),
    req: PhotoRequest = Depends(photo_form),
    detector: BoxDetector = Depends(get_detector),
    classic: ClassicDetector = Depends(get_classic),
):
    """Detect boxes in one photo.

    engine=auto uses the colour detector (PHOTO_ENGINE=classic, the default); engine=yolo uses
    the trained model. The result also includes an estimated room size and capacity when
    enough boxes are found.
    """
    _check_extension(file.filename, config.ALLOWED_IMAGE_EXT)
    data = await _read_limited(file, config.MAX_IMAGE_MB)
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = await _analyze(image, req, detector, classic)
    payload = _photo_payload(result, req.params)
    if include_image:
        annotated = await run_in_threadpool(draw_detections, image, result.detections)
        payload["annotated_image_b64"] = base64.b64encode(encode_jpeg(annotated)).decode("ascii")
    return payload


@router.post("/predict/image/annotated", tags=["prediction"], response_class=Response)
async def predict_image_annotated(
    file: UploadFile = File(...),
    req: PhotoRequest = Depends(photo_form),
    detector: BoxDetector = Depends(get_detector),
    classic: ClassicDetector = Depends(get_classic),
):
    """Same as /predict/image but returns the annotated JPEG (count in X-Box-Count header)."""
    _check_extension(file.filename, config.ALLOWED_IMAGE_EXT)
    data = await _read_limited(file, config.MAX_IMAGE_MB)
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = await _analyze(image, req, detector, classic)
    annotated = await run_in_threadpool(draw_detections, image, result.detections)
    return Response(
        content=encode_jpeg(annotated),
        media_type="image/jpeg",
        headers={"X-Box-Count": str(len(result.detections)), "X-Engine": result.engine},
    )


# --------------------------------------------------------------------------- routes: video
@router.post("/predict/video", response_model=JobCreated, status_code=202, tags=["prediction"])
async def predict_video(
    file: UploadFile = File(...),
    conf: ConfForm = config.VIDEO_CONF,
    iou: IouForm = config.VIDEO_IOU,
    imgsz: int = Form(config.VIDEO_IMGSZ),
    frame_stride: StrideForm = 1,
    track: bool = Form(False),
    detector: BoxDetector = Depends(get_detector),
    jobs: JobStore = Depends(get_jobs),
    executor: ThreadPoolExecutor = Depends(get_executor),
):
    """Queue a video for processing (always uses the YOLO model).

    Poll GET /jobs/{job_id} for progress and results."""
    require_model(detector)
    _check_imgsz(imgsz)
    ext = _check_extension(file.filename, config.ALLOWED_VIDEO_EXT)

    jobs_dir = config.OUTPUT_DIR / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs.cleanup(jobs_dir)

    job = jobs.create()
    job_dir = jobs_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / f"input{ext}"
    out_path = job_dir / "annotated.mp4"
    try:
        await _save_limited(file, in_path, config.MAX_VIDEO_MB)
    except HTTPException:
        jobs.remove(job.id)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    jobs.update(job.id, input_path=in_path, output_path=out_path)
    executor.submit(
        _run_video_job, detector, jobs, job.id, in_path, out_path,
        conf, iou, imgsz, frame_stride, track,
    )
    return {"job_id": job.id, "status": "queued"}


@router.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
def job_status(job_id: str, jobs: JobStore = Depends(get_jobs)):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
    return _job_to_status(job)


@router.get("/jobs/{job_id}/video", tags=["jobs"])
def job_video(job_id: str, jobs: JobStore = Depends(get_jobs)):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
    if job.status != "done" or job.output_path is None or not Path(job.output_path).exists():
        raise HTTPException(status_code=409, detail="The annotated video is not ready yet.")
    return FileResponse(job.output_path, media_type="video/mp4", filename=f"annotated_{job.id}.mp4")


@router.get("/jobs/{job_id}/report.csv", tags=["jobs"])
def job_report(job_id: str, jobs: JobStore = Depends(get_jobs)):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
    if job.status != "done" or job.summary is None:
        raise HTTPException(status_code=409, detail="The report is not ready yet.")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["frame_index", "time_sec", "box_count"])
    for row in job.summary.per_frame:
        writer.writerow([row.frame_index, row.time_sec, row.count])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="box_counts_{job.id}.csv"'},
    )


# --------------------------------------------------------------------------- app factory
def create_app(detector: Optional[BoxDetector] = None, classic: Optional[ClassicDetector] = None) -> FastAPI:
    """Build the app. Pass a detector to inject a pre-built (or fake) one, e.g. in tests."""
    state_detector = detector or BoxDetector(config.MODEL_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not app.state.detector.loaded:
            app.state.detector.load()
        yield
        app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Warehouse Box Counter API",
        version=__version__,
        description="Count boxes in warehouse photos (colour detector or YOLO) and videos (YOLO).",
        lifespan=lifespan,
    )
    app.state.detector = state_detector
    app.state.classic = classic or ClassicDetector()
    app.state.jobs = JobStore(config.JOB_TTL_SECONDS)
    app.state.executor = ThreadPoolExecutor(max_workers=config.VIDEO_WORKERS)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Box-Count"],
    )
    app.include_router(router)
    return app


app = create_app()
