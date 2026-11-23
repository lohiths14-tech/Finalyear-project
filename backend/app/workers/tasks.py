"""
Background tasks: offline queue sync and async inference.
"""

import cv2
import numpy as np

from app.workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3)
def sync_offline_scan(self, file_data: str, scan_type: str, lang: str = "en"):
    """
    Process a single offline scan from the farmer's queue.
    Retries on failure with exponential backoff.
    """
    import base64

    from app.services.inference_service import InferenceService

    try:
        # Decode base64 image
        header, data = file_data.split(",", 1)
        raw = base64.b64decode(data)
        nparr = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image from base64")

        svc = InferenceService()
        if scan_type == "leaf":
            result = svc.analyze_leaf(image)
        else:
            result = svc.analyze_field(image)

        # Save to DB
        return result

    except Exception as exc:
        raise self.retry(exc=exc, countdown=2**self.request.retries)


@celery_app.task
def retrain_model_trigger(model_version: str):
    """
    Triggered by MLflow webhook or scheduled cron.
    Runs the training pipeline and registers the new model.
    """
    import os
    import subprocess

    result = subprocess.run(
        ["python", "scripts/train_synthetic.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    return {"status": "retrained", "version": model_version}
