"""
AgriPulse 2.0 — Asynchronous Batch Job Queue & Background Inference Engine
Enables multi-tenant, non-blocking cloud worker execution for heavy field drone and multi-spectral scans.
"""

import asyncio
import base64
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/jobs", tags=["Async Job Queue"])

# In-memory fast job store (synchronized across requests)
JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}
MAX_JOB_HISTORY = 100


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    scan_type: str  # field, leaf
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    progress_pct: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _clean_old_jobs():
    if len(JOB_REGISTRY) > MAX_JOB_HISTORY:
        # Keep most recent
        sorted_keys = sorted(
            JOB_REGISTRY.keys(),
            key=lambda k: JOB_REGISTRY[k].get("created_at_epoch", 0),
        )
        for k in sorted_keys[: len(JOB_REGISTRY) - MAX_JOB_HISTORY]:
            JOB_REGISTRY.pop(k, None)


async def _execute_async_inference(job_id: str, image_bytes: bytes, scan_type: str, request: Request):
    job = JOB_REGISTRY.get(job_id)
    if not job:
        return

    job["status"] = "processing"
    job["progress_pct"] = 25
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid or corrupted image format")

        job["progress_pct"] = 50

        # Retrieve InferenceService from app state
        svc = getattr(request.app.state, "inference_service", None)
        if svc is None:
            raise RuntimeError("InferenceService engine not loaded")

        loop = asyncio.get_running_loop()
        if scan_type == "leaf":
            result = await loop.run_in_executor(None, svc.analyze_leaf, img)
        else:
            result = await loop.run_in_executor(None, svc.analyze_field, img)

        job["progress_pct"] = 100
        job["status"] = "completed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        job["result"] = result

    except Exception as e:
        job["status"] = "failed"
        job["progress_pct"] = 100
        job["error"] = str(e)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)


@router.post("/analyze", response_model=JobStatusResponse)
async def submit_async_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scan_type: str = Form("field"),
):
    """
    Submits a field or leaf scan to the background worker queue. Returns a job_id for polling.
    """
    if scan_type not in ["field", "leaf"]:
        raise HTTPException(status_code=400, detail="scan_type must be either 'field' or 'leaf'")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    JOB_REGISTRY[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "scan_type": scan_type,
        "filename": file.filename,
        "created_at": now_iso,
        "created_at_epoch": time.time(),
        "started_at": None,
        "completed_at": None,
        "duration_ms": None,
        "progress_pct": 5,
        "result": None,
        "error": None,
    }
    _clean_old_jobs()

    # Launch in background
    background_tasks.add_task(_execute_async_inference, job_id, contents, scan_type, request)

    return JOB_REGISTRY[job_id]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll the status of an asynchronous inference job.
    """
    job = JOB_REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get("", response_model=List[JobStatusResponse])
async def list_jobs():
    """
    List all tracked background jobs in reverse chronological order.
    """
    sorted_jobs = sorted(
        JOB_REGISTRY.values(),
        key=lambda j: j.get("created_at_epoch", 0),
        reverse=True,
    )
    return sorted_jobs
