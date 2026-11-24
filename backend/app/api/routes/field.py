"""
Field-level canopy analysis endpoint.
"""

import asyncio
from functools import partial
import json
import time
import uuid

import cv2
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from ...core.farmer_language import translate_to_farmer_language
from ...core.limiter import limiter
from ...core.logger import logger
from ...db.database import get_db
from ...db.models import ScanResult
from ..dependencies import (
    get_inference_service,
    resize_if_needed,
    sanitize_filename,
)

router = APIRouter()


@router.post("/analyze/field")
@limiter.limit("10/minute")
async def analyze_field(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_inference_service),
):
    """
    Field-level canopy analysis — full Spatio-Spectral-Temporal pipeline.
    Max file: 20 MB. Images larger than 4096×4096 are auto-resized.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, WEBP).")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(f"Failed to decode image from file: {file.filename}")
        raise HTTPException(
            422, "Could not decode image. Ensure the file is a valid image."
        )

    # Auto-resize oversized images instead of rejecting them
    try:
        image = resize_if_needed(image)
    except ValueError as e:
        raise HTTPException(422, str(e))

    safe_name = sanitize_filename(file.filename)
    request_id = getattr(request.state, "request_id", "N/A")
    logger.info(
        f"[{request_id}] Field analysis — shape: {image.shape}, file: '{safe_name}'"  # noqa: E501
    )

    try:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, partial(svc.analyze_field, image)
        )
        elapsed = time.perf_counter() - t0

        if "error" in results:
            logger.warning(
                f"[{request_id}] Analysis returned error: {results['error']}"
            )
            raise HTTPException(422, results["error"])

        logger.info(
            f"[{request_id}] Field inference completed in {elapsed:.2f}s — "
            f"disease: '{results['disease_type']}', "
            f"confidence: {results['confidence']:.1f}%, "
            f"severity: {results['average_severity']:.1f}%, "
            f"spread_risk: {results.get('spread_risk', 'N/A')}"
        )

        # Step 4: Simple Field Health Message
        infection_rate = results.get("infection_rate", 0)
        if infection_rate > 30:
            health_msg = f"Watch out! {infection_rate:.0f}% of your field is affected. Focus your treatment in the Red areas."  # noqa: E501
        elif infection_rate > 5:
            health_msg = f"{infection_rate:.0f}% of your field shows signs of stress. Keep a close eye on the Yellow areas."  # noqa: E501
        else:
            health_msg = "Your field looks mostly healthy. Great job!"

        results["farmer_friendly"] = translate_to_farmer_language(
            results["disease_type"], results["confidence"]
        )
        results["farmer_friendly"]["field_health_message"] = health_msg

        scan = ScanResult(
            id=str(uuid.uuid4()),
            scan_type="field",
            filename=safe_name,
            disease_type=results["disease_type"],
            severity=results["average_severity"],
            confidence=results["confidence"],
            spread_risk=results.get("spread_risk"),
            treatment=results["treatment"],
            full_results=json.dumps(
                {
                    k: v
                    for k, v in results.items()
                    if k
                    not in ("disease_distribution", "severity_map", "regions")
                }
            ),
        )
        db.add(scan)
        await db.commit()

        results["scan_id"] = scan.id
        results["disclaimer"] = (
            "AI suggestion only. Always consult a local expert for high-stakes decisions."  # noqa: E501
        )

        return results

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            f"[{request_id}] Unhandled error during field analysis"
        )
        raise HTTPException(500, "Internal inference error.")


@router.post("/analyze/field/stream")
@limiter.limit("10/minute")
async def analyze_field_stream(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_inference_service),
):
    """
    Streaming field-level canopy analysis with real-time SSE progress events.
    Emits events: 'progress', 'complete', or 'error'.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, WEBP).")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(f"Failed to decode image from file: {file.filename}")
        raise HTTPException(
            422, "Could not decode image. Ensure the file is a valid image."
        )

    try:
        image = resize_if_needed(image)
    except ValueError as e:
        raise HTTPException(422, str(e))

    safe_name = sanitize_filename(file.filename)
    request_id = getattr(request.state, "request_id", "N/A")
    logger.info(
        f"[{request_id}] Streaming field analysis — shape: {image.shape}, file: '{safe_name}'"
    )

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sync_progress_callback(event_data: dict[str, Any]):
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", event_data))

    async def run_inference():
        try:
            t0 = time.perf_counter()
            results = await loop.run_in_executor(
                None,
                partial(
                    svc.analyze_field,
                    image,
                    progress_callback=sync_progress_callback,
                ),
            )
            elapsed = time.perf_counter() - t0

            if "error" in results:
                logger.warning(
                    f"[{request_id}] Stream analysis error: {results['error']}"
                )
                await queue.put(("error", {"error": results["error"]}))
                return

            infection_rate = results.get("infection_rate", 0)
            if infection_rate > 30:
                health_msg = f"Watch out! {infection_rate:.0f}% of your field is affected. Focus your treatment in the Red areas."
            elif infection_rate > 5:
                health_msg = f"{infection_rate:.0f}% of your field shows signs of stress. Keep a close eye on the Yellow areas."
            else:
                health_msg = "Your field looks mostly healthy. Great job!"

            results["farmer_friendly"] = translate_to_farmer_language(
                results["disease_type"], results["confidence"]
            )
            results["farmer_friendly"]["field_health_message"] = health_msg

            try:
                scan = ScanResult(
                    id=str(uuid.uuid4()),
                    scan_type="field",
                    filename=safe_name,
                    disease_type=results["disease_type"],
                    severity=results["average_severity"],
                    confidence=results["confidence"],
                    spread_risk=results.get("spread_risk"),
                    treatment=results["treatment"],
                    full_results=json.dumps(
                        {
                            k: v
                            for k, v in results.items()
                            if k not in ("disease_distribution", "severity_map", "regions")
                        }
                    ),
                )
                db.add(scan)
                await db.commit()
                results["scan_id"] = scan.id
            except Exception as dberr:
                logger.warning(f"[{request_id}] DB logging skipped: {dberr}")
                results["scan_id"] = str(uuid.uuid4())

            results["disclaimer"] = (
                "AI suggestion only. Always consult a local expert for high-stakes decisions."
            )
            results["inference_time_s"] = round(elapsed, 2)

            logger.info(
                f"[{request_id}] Streaming field inference complete in {elapsed:.2f}s"
            )
            await queue.put(("complete", results))
        except Exception as e:
            logger.exception(f"[{request_id}] Error during streaming inference")
            await queue.put(("error", {"error": str(e)}))

    asyncio.create_task(run_inference())

    async def event_generator():
        while True:
            try:
                event_type, payload = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                break

            yield {
                "event": event_type,
                "data": json.dumps(payload),
            }

            if event_type in ("complete", "error"):
                break

    return EventSourceResponse(event_generator())


from pydantic import BaseModel, Field as PydanticField
from typing import List, Optional, Dict, Any


class FieldConsultRequest(BaseModel):
    question: str = PydanticField(..., min_length=2, description="Agronomic question from user")
    disease_type: str = PydanticField("Rice Blast", description="Identified field disease")
    average_severity: float = PydanticField(..., description="Average severity across field (0-100)")
    infection_rate: float = PydanticField(..., description="Infection rate percentage (0-100)")
    spread_risk: str = PydanticField("MODERATE", description="Spatial spread risk")
    clusters: Optional[List[Dict[str, Any]]] = PydanticField(default_factory=list, description="List of detected spatial clusters")
    vegetation_coverage: float = PydanticField(85.0, description="Vegetation coverage percentage")
    land_area: float = PydanticField(1.0, description="Land area in specified unit")
    area_unit: str = PydanticField("acre", description="Unit of land area")
    history: Optional[List[Dict[str, str]]] = PydanticField(default_factory=list, description="Conversation history")


@router.post("/analyze/field/consult")
@limiter.limit("30/minute")
async def field_consult(
    request: Request,
    payload: FieldConsultRequest,
):
    """
    Field-Level Spatial Intelligence Decision Engine.
    Grounded strictly in DINOv2 adaptive grid spatial telemetry and ICAR-IRRI protocols.
    """
    from ...services.genai_service import genai_service
    res = await genai_service.consult_field_intelligence(
        question=payload.question,
        disease_type=payload.disease_type,
        average_severity=payload.average_severity,
        infection_rate=payload.infection_rate,
        spread_risk=payload.spread_risk,
        clusters=payload.clusters,
        vegetation_coverage=payload.vegetation_coverage,
        land_area=payload.land_area,
        area_unit=payload.area_unit,
        history=payload.history,
    )
    return res
