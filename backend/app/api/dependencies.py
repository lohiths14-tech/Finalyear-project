"""
Shared FastAPI route utilities.
Centralises repeated helpers that were previously copy-pasted across field.py and leaf.py.
"""  # noqa: E501

import re

from fastapi import Request


def sanitize_filename(name: str) -> str:
    """Strip path-traversal characters; limit to 120 chars."""
    return re.sub(r"[^\w.\-]", "_", name or "unknown")[:120]


def get_inference_service(request: Request):
    """
    FastAPI dependency — returns the singleton InferenceService
    stored on app.state (set during lifespan startup).
    """
    return request.app.state.inference_service


# Maximum allowed image dimensions (prevents OOM on extremely large images)
MAX_IMAGE_DIMENSION = 4096


def resize_if_needed(image):
    """
    If the image exceeds MAX_IMAGE_DIMENSION on either axis, downscale it
    proportionally using area interpolation (best for downscaling).
    Images smaller than the limit are returned unchanged.
    Raises ValueError for None or sub-16px images.
    """
    import cv2

    if image is None:
        raise ValueError("Image is None")
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(
            f"Image dimensions {w}×{h} are too small. Minimum 16×16 required."
        )
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


# Keep old name as alias so existing callers don't break
def validate_image_dimensions(image) -> None:
    """Deprecated: use resize_if_needed(). Raises ValueError if image is invalid."""
    resize_if_needed(image)  # will raise on None / sub-16px
