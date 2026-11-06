"""Thread-safe in-memory store for background video jobs."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

from .entities import VideoSummary


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | failed
    progress: float = 0.0
    message: str = "Queued"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    summary: Optional[VideoSummary] = None


class JobStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.id] = job
        return replace(job)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job else None

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def cleanup(self, jobs_dir: Optional[Path] = None) -> int:
        """Drop finished jobs older than the TTL and delete their files."""
        now = time.time()
        with self._lock:
            expired = [
                j for j in self._jobs.values()
                if j.status in ("done", "failed") and now - j.updated_at > self._ttl
            ]
            for job in expired:
                del self._jobs[job.id]
        for job in expired:
            if jobs_dir is not None:
                shutil.rmtree(Path(jobs_dir) / job.id, ignore_errors=True)
        return len(expired)
