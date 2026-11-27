"""
Leaf-level micro-diagnostic endpoint.
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


@router.post("/analyze/leaf")
@limiter.limit("20/minute")
async def analyze_leaf(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_inference_service),
):
    """
    Leaf-level micro-diagnostic — spectral-only inference.
    Max file: 20 MB. Images larger than 4096×4096 are auto-resized.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(422, "Could not decode image.")

    # Auto-resize oversized images instead of rejecting them
    try:
        image = resize_if_needed(image)
    except ValueError as e:
        raise HTTPException(422, str(e))

    safe_name = sanitize_filename(file.filename)
    request_id = getattr(request.state, "request_id", "N/A")
    logger.info(
        f"[{request_id}] Leaf analysis — shape: {image.shape}, file: '{safe_name}'"  # noqa: E501
    )

    try:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, partial(svc.analyze_leaf, image)
        )
        elapsed = time.perf_counter() - t0

        logger.info(
            f"[{request_id}] Leaf inference completed in {elapsed:.2f}s — "
            f"disease: '{results['disease_type']}', "
            f"confidence: {results['confidence']:.1f}%, "
            f"status: {results['status']}"
        )

        # Trust & Safety: Step 10
        if results["confidence"] < 40.0:
            results["farmer_friendly"] = {
                "simple_label": "Not sure",
                "certainty": "Low Certainty",
                "what_is_it": "The image is not clear enough to be certain.",
                "what_to_do": "Please try again with a clear photo in better light, or ask a human expert.",  # noqa: E501
            }
        else:
            results["farmer_friendly"] = translate_to_farmer_language(
                results["disease_type"], results["confidence"]
            )

        scan = ScanResult(
            id=str(uuid.uuid4()),
            scan_type="leaf",
            filename=safe_name,
            disease_type=results["disease_type"],
            severity=results["severity"],
            confidence=results["confidence"],
            spread_risk=None,
            treatment=results["treatment"],
            full_results=json.dumps(
                {k: v for k, v in results.items() if k != "ngvc"}
            ),
        )
        db.add(scan)
        await db.commit()

        results["scan_id"] = scan.id
        # Add a mandatory disclaimer (Step 10)
        results["disclaimer"] = (
            "AI suggestion only. Always consult a local expert for high-stakes decisions."  # noqa: E501
        )

        return results

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            f"[{request_id}] Unhandled error during leaf analysis"
        )
        raise HTTPException(500, "Internal inference error.")


from pydantic import BaseModel, Field

from ...core.agronomy_engine import consult_agronomist


class ConsultRequest(BaseModel):
    disease_type: str = Field(..., description="Identified disease or condition")
    severity: float = Field(..., description="Severity percentage (0-100)")
    question: str = Field(..., description="Agronomic question from user")


@router.post("/analyze/consult")
@limiter.limit("30/minute")
async def agronomy_consult(
    request: Request,
    payload: ConsultRequest,
):
    """
    AI Agronomist Consultation endpoint grounded in ICAR & IRRI agronomic research.
    """
    answer = consult_agronomist(
        disease_type=payload.disease_type,
        severity=payload.severity,
        question=payload.question,
    )
    return {
        "disease_type": payload.disease_type,
        "severity": payload.severity,
        "question": payload.question,
        "answer": answer,
    }

