"""
Real-World Robustness Evaluation & Stress Testing
================================================
Systematically evaluates model robustness across 25+ real-world degradation conditions.
Used for:
  - Pre-deployment validation
  - Automated red-teaming during CI/CD
  - Model comparison benchmarking
  - Identifying failure modes before field deployment
"""  # noqa: E501

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

# ── Degradation conditions ───────────────────────────────


@dataclass
class RobustnessCondition:
    name: str
    category: str  # "lighting" | "blur" | "noise" | "weather" | "geometric" | "content"  # noqa: E501
    severity_range: tuple[float, float]  # min/max degradation strength
    description: str


ROBUSTNESS_CONDITIONS = [
    # Lighting conditions
    RobustnessCondition(
        "low_light", "lighting", (0.3, 0.7), "Underexposure: image too dark"
    ),
    RobustnessCondition(
        "overexposure",
        "lighting",
        (0.3, 0.7),
        "Overexposure: bright white washout",
    ),
    RobustnessCondition(
        "uneven_illumination", "lighting", (0.4, 0.8), "Strong corner shading"
    ),
    RobustnessCondition(
        "harsh_sunlight",
        "lighting",
        (0.3, 0.7),
        "Simulated direct sunlight glare",
    ),
    RobustnessCondition(
        "mixed_lighting",
        "lighting",
        (0.4, 0.7),
        "Mixed shadow + light patches",
    ),
    # Blur conditions
    RobustnessCondition(
        "motion_blur",
        "blur",
        (2.0, 8.0),
        "Directional motion blur (drone vibration)",
    ),
    RobustnessCondition(
        "defocus_blur",
        "blur",
        (3.0, 10.0),
        "Out-of-focus blur (depth-of-field limits)",
    ),
    RobustnessCondition(
        "gaussian_blur", "blur", (1.5, 5.0), "Mild Gaussian smoothing"
    ),
    # Noise conditions
    RobustnessCondition(
        "high_iso_noise", "noise", (15.0, 30.0), "High ISO sensor noise"
    ),
    RobustnessCondition(
        "compression_artifacts",
        "noise",
        (30.0, 70.0),
        "JPEG/MPEG compression artifacts",
    ),
    # Weather / atmosphere
    RobustnessCondition(
        "fog_haze", "weather", (0.3, 0.7), "Atmospheric haze / fog"
    ),
    RobustnessCondition(
        "rain_droplets", "weather", (0.3, 0.6), "Water droplets on lens"
    ),
    RobustnessCondition(
        "dust_contamination",
        "weather",
        (0.3, 0.6),
        "Dust/mud splatter on image",
    ),
    # Geometric distortions
    RobustnessCondition(
        "perspective_tilt",
        "geometric",
        (5.0, 15.0),
        "Pitch/roll from drone flight",
    ),
    RobustnessCondition(
        "rotation", "geometric", (10.0, 30.0), "Arbitrary rotation"
    ),
    RobustnessCondition(
        "scale_variation", "geometric", (0.5, 0.8), "Scale down (zoom out)"
    ),
    RobustnessCondition(
        "scale_variation_hi", "geometric", (1.2, 1.5), "Scale up (zoom in)"
    ),
    # Content / occlusion
    RobustnessCondition(
        "leaf_occlusion", "content", (0.1, 0.3), "Overlapping leaf overlay"
    ),
    RobustnessCondition(
        "soil_overlap", "content", (0.15, 0.35), "Soil patch overlay"
    ),
    RobustnessCondition(
        "shadow_overlay", "content", (0.2, 0.4), "Heavy shadow region"
    ),
    RobustnessCondition(
        "background_noise",
        "content",
        (0.3, 0.6),
        "Non-plant background distraction",
    ),
    # Seasonal / structural
    RobustnessCondition(
        "growth_stage_early",
        "content",
        (0.2, 0.5),
        "Early growth stage (less leaf area)",
    ),
    RobustnessCondition(
        "growth_stage_late",
        "content",
        (0.2, 0.5),
        "Late growth stage (dense canopy)",
    ),
    RobustnessCondition(
        "low_resolution",
        "content",
        (0.25, 0.5),
        "Downsampled low-resolution image",
    ),
    RobustnessCondition(
        "mixed_healthy_diseased",
        "content",
        (0.3, 0.6),
        "Mixed healthy + diseased regions",
    ),
    RobustnessCondition(
        "multiple_diseases",
        "content",
        (0.4, 0.7),
        "Multiple concurrent diseases",
    ),
]


# ── Degradation applicators ──────────────────────────────


_DEGRADATIONS = {}


def _register_degradation(name):
    def decorator(func):
        _DEGRADATIONS[name] = func
        return func
    return decorator


@_register_degradation("low_light")
def _apply_low_light(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    factor = 1.0 - strength * 0.7
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img


@_register_degradation("overexposure")
def _apply_overexposure(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    # Compress highlights
    img_f = np.where(
        img_f > 0.7, 0.7 + (img_f - 0.7) * (1 - strength * 0.5), img_f
    )
    img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return img


@_register_degradation("uneven_illumination")
def _apply_uneven_illumination(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    overlay = np.ones_like(img, dtype=np.float32)
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.8 * strength)
    cv2.circle(
        overlay[:],
        (cx, cy),
        radius,
        (0.3 + strength * 0.3, 0.3 + strength * 0.3, 0.3 + strength * 0.3),
        -1,
    )
    img = np.clip(img.astype(np.float32) * overlay, 0, 255).astype(
        np.uint8
    )
    return img


@_register_degradation("harsh_sunlight")
def _apply_harsh_sunlight(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(
        (img_f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV
    ).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + strength * 0.4), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + strength * 0.2), 0, 255)
    img = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
    ).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@_register_degradation("mixed_lighting")
def _apply_mixed_lighting(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(3 * strength)):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        rx = rng.integers(w // 6, w // 2)
        ry = rng.integers(h // 6, h // 2)
        y, x = np.ogrid[:h, :w]
        dx = (x - cx) / (rx + 1e-6)
        dy = (y - cy) / (ry + 1e-6)
        brightness = np.exp(-(dx * dx + dy * dy)) * strength
        mask += brightness * 0.3
    img = np.clip(img.astype(np.float32) * (1.0 - mask), 0, 255).astype(
        np.uint8
    )
    return img


@_register_degradation("motion_blur")
def _apply_motion_blur(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    ksize = int(strength * 2) + 3
    if ksize % 2 == 0:
        ksize += 1
    angle = rng.uniform(0, 180)
    k = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for i in range(ksize):
        proj = int(ca * (i - center) + sa * (center - center))
        if abs(proj) <= ksize // 2:
            k[center, proj + center] = 1.0
    k /= k.sum()
    img = cv2.filter2D(img, -1, k)
    return img


@_register_degradation("defocus_blur")
def _apply_defocus_blur(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    ksize = int(strength * 2) + 3
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getGaussianKernel(ksize, sigma=strength)
    kernel = kernel @ kernel.T
    img = cv2.filter2D(img, -1, kernel / kernel.sum())
    return img


@_register_degradation("gaussian_blur")
def _apply_gaussian_blur(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    ksize = int(strength) + 3
    if ksize % 2 == 0:
        ksize += 1
    img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=strength * 0.5)
    return img


@_register_degradation("high_iso_noise")
def _apply_high_iso_noise(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    sigma = strength * 10
    noise = rng.normal(0, sigma, size=img.shape, dtype=np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


@_register_degradation("compression_artifacts")
def _apply_compression_artifacts(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    # Simulate block artifacts via downsampling + upsampling
    scale = 1.0 - strength * 0.3
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 32 and new_h > 32:
        small = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


@_register_degradation("fog_haze")
def _apply_fog_haze(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    fog = np.ones_like(img_f) * 0.9 * strength
    img = (
        np.clip(img_f * (1 - strength * 0.5) + fog * strength, 0, 1) * 255
    )
    img = img.astype(np.uint8)
    return img


@_register_degradation("rain_droplets")
def _apply_rain_droplets(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    overlay = img.copy()
    for _ in range(int(20 * strength)):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        r = rng.integers(2, 6)
        cv2.circle(overlay, (cx, cy), r, (200, 200, 200), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return img


@_register_degradation("dust_contamination")
def _apply_dust_contamination(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    for _ in range(int(50 * strength)):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        r = rng.integers(1, 4)
        color = (
            rng.integers(80, 140),
            rng.integers(60, 100),
            rng.integers(30, 80),
        )
        cv2.circle(img, (cx, cy), r, color, -1)
    return img


@_register_degradation("perspective_tilt")
def _apply_perspective_tilt(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    roll = strength * 10
    M = cv2.getRotationMatrix2D((w // 2, h // 2), roll, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


@_register_degradation("rotation")
def _apply_rotation(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    angle = strength * 30
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


@_register_degradation("scale_variation")
def _apply_scale_variation(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    scale = 1.0 - strength * 0.4
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 32 and new_h > 32:
        small = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


@_register_degradation("scale_variation_hi")
def _apply_scale_variation_hi(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    scale = 1.0 + strength * 0.4
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < w * 2 and new_h < h * 2:
        small = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


@_register_degradation("leaf_occlusion")
def _apply_leaf_occlusion(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    overlay = img.copy()
    for _ in range(int(2 * strength) + 1):
        cx = rng.integers(w // 8, 7 * w // 8)
        cy = rng.integers(h // 8, 7 * h // 8)
        rx = rng.integers(w // 10, w // 4)
        ry = rng.integers(h // 10, h // 4)
        cv2.ellipse(
            overlay,
            (cx, cy),
            (rx, ry),
            rng.uniform(0, 180),
            0,
            360,
            (10, 10, 10),
            -1,
        )
    img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    return img


@_register_degradation("soil_overlap")
def _apply_soil_overlap(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    for _ in range(int(2 * strength) + 1):
        x = rng.integers(w // 10, 8 * w // 10)
        y = rng.integers(h // 10, 8 * h // 10)
        rw = rng.integers(w // 12, w // 5)
        rh = rng.integers(h // 12, h // 5)
        soil = np.array(
            [
                rng.integers(80, 140),
                rng.integers(60, 100),
                rng.integers(30, 70),
            ],
            dtype=np.uint8,
        )
        patch = np.full((rh, rw, 3), soil, dtype=np.uint8)
        x1, y1 = max(0, x - rw // 2), max(0, y - rh // 2)
        x2, y2 = min(w, x + rw // 2), min(h, y + rh // 2)
        img[y1:y2, x1:x2] = patch[: y2 - y1, : x2 - x1]
    return img


@_register_degradation("shadow_overlay")
def _apply_shadow_overlay(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx = rng.integers(w // 4, 3 * w // 4)
    cy = rng.integers(h // 4, 3 * h // 4)
    rx = rng.integers(w // 5, w // 2)
    ry = rng.integers(h // 5, h // 2)
    y, x = np.ogrid[:h, :w]
    dx = (x - cx) / (rx + 1e-6)
    dy = (y - cy) / (ry + 1e-6)
    mask = np.exp(-(dx * dx + dy * dy)) * strength
    img = np.clip(
        img.astype(np.float32) * (1.0 - mask * 0.6), 0, 255
    ).astype(np.uint8)
    return img


@_register_degradation("background_noise")
def _apply_background_noise(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    # Replace random rectangular regions with non-plant colors
    for _ in range(int(3 * strength) + 1):
        x = rng.integers(0, max(1, w - 20))
        y = rng.integers(0, max(1, h - 20))
        rw = rng.integers(w // 10, w // 4)
        rh = rng.integers(h // 10, h // 4)
        color = (
            rng.integers(100, 200),
            rng.integers(80, 160),
            rng.integers(50, 120),
        )
        cv2.rectangle(img, (x, y), (x + rw, y + rh), color, -1)
    return img


@_register_degradation("low_resolution")
def _apply_low_resolution(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    scale = max(0.2, strength * 0.4)
    new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


@_register_degradation("mixed_healthy_diseased")
def _apply_mixed_healthy_diseased(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    # Split image: top half normal, bottom half darker/more desaturated
    img_f = img.astype(np.float32)
    mid = h // 2
    img_f[mid:] = img_f[mid:] * (1.0 - strength * 0.3)
    img = np.clip(img_f, 0, 255).astype(np.uint8)
    return img


@_register_degradation("multiple_diseases")
def _apply_multiple_diseases(
        img: np.ndarray,
        strength: float,
        rng,
        w: int,
        h: int) -> np.ndarray:
    # Create two disease-like regions with different color shifts
    img_f = img.astype(np.float32)
    top_mask = np.zeros_like(img_f)
    top_mask[: h // 3, :] = 1.0
    bottom_mask = np.zeros_like(img_f)
    bottom_mask[2 * h // 3:, :] = 1.0
    img_f = img_f * (1 - top_mask * strength * 0.3)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(
        np.float32
    )
    hsv[:, :, 0] = (hsv[:, :, 0] + strength * 15) % 180
    img_f2 = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
    ).astype(np.float32)
    img = np.clip(img_f2, 0, 255).astype(np.uint8)
    return img


def apply_degradation(
    image: np.ndarray,
    condition_name: str,
    strength: float,
    rng=None,
) -> np.ndarray:
    """Apply a specific degradation at given strength (0–1)."""
    if rng is None:
        rng = np.random.default_rng()

    img = image.copy()
    h, w = img.shape[:2]

    func = _DEGRADATIONS.get(condition_name)
    if func is not None:
        return func(img, strength, rng, w, h)

    return img


# ── Robustness test results ───────────────────────────────────


@dataclass
class ConditionResult:
    condition: str
    category: str
    severity: float
    accuracy_delta: float  # relative to clean accuracy
    confidence_delta: float
    stability_score: float
    passed: bool


@dataclass
class RobustnessReport:
    condition_results: list[ConditionResult]
    average_accuracy_delta: float
    worst_category: str
    overall_score: float
    failure_modes: list[dict]

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "average_accuracy_delta": round(self.average_accuracy_delta, 4),
            "worst_category": self.worst_category,
            "failure_modes": self.failure_modes,
            "per_condition": [
                {
                    "condition": r.condition,
                    "category": r.category,
                    "severity": round(r.severity, 2),
                    "accuracy_delta": round(r.accuracy_delta, 4),
                    "confidence_delta": round(r.confidence_delta, 4),
                    "stability": round(r.stability_score, 3),
                    "passed": r.passed,
                }
                for r in self.condition_results
            ],
            "per_category": self._per_category(),
        }

    def _per_category(self) -> dict:
        cats: dict[str, list[ConditionResult]] = defaultdict(list)
        for r in self.condition_results:
            cats[r.category].append(r)
        return {
            cat: {
                "avg_accuracy_delta": round(
                    np.mean([r.accuracy_delta for r in rs]), 4
                ),
                "avg_stability": round(
                    np.mean([r.stability_score for r in rs]), 3
                ),
                "num_conditions": len(rs),
            }
            for cat, rs in cats.items()
        }


# ── Robustness evaluator ───────────────────────────────────


class RobustnessEvaluator:
    """
    Stress-tests the inference pipeline across all degradation conditions.

    Usage:
      evaluator = RobustnessEvaluator(conditions=ROBUSTNESS_CONDITIONS[:10])
      report = evaluator.evaluate(inference_fn, test_images, reference_accuracy=0.85)
    """  # noqa: E501

    def __init__(
        self,
        conditions: list[RobustnessCondition] | None = None,
        severity_samples: int = 3,
    ):
        self.conditions = conditions or ROBUSTNESS_CONDITIONS
        self.severity_samples = severity_samples

    def evaluate(
        self,
        inference_fn,
        test_images: list[np.ndarray],
        reference_accuracy: float = 0.85,
        reference_confidence: float = 0.75,
    ) -> RobustnessReport:
        """
        Evaluate model across all conditions and severity levels.
        inference_fn(image) -> dict with "disease_type", "confidence", "confidence_raw"
        """  # noqa: E501
        results: list[ConditionResult] = []
        failure_modes: list[dict] = []

        for cond in self.conditions:
            lo, hi = cond.severity_range
            strengths = np.linspace(lo, hi, self.severity_samples)

            for strength in strengths:
                result = self._evaluate_condition(
                    inference_fn,
                    test_images,
                    cond.name,
                    cond.category,
                    strength,
                    reference_accuracy,
                    reference_confidence,
                )
                results.append(result)

                if not result.passed and result.accuracy_delta < -0.15:
                    failure_modes.append(
                        {
                            "condition": cond.name,
                            "category": cond.category,
                            "severity": round(result.severity, 2),
                            "accuracy_drop": round(result.accuracy_delta, 3),
                        }
                    )

        # Aggregate
        avg_delta = np.mean([r.accuracy_delta for r in results])
        cats: dict[str, list[ConditionResult]] = defaultdict(list)
        for r in results:
            cats[r.category].append(r)
        worst_cat = max(
            cats,
            key=lambda c: abs(np.mean([r.accuracy_delta for r in cats[c]])),
        )

        overall_score = max(0.0, min(1.0, 1.0 + avg_delta))
        return RobustnessReport(
            condition_results=results,
            average_accuracy_delta=avg_delta,
            worst_category=worst_cat,
            overall_score=round(overall_score, 4),
            failure_modes=failure_modes,
        )

    def _evaluate_condition(
        self,
        inference_fn,
        test_images: list[np.ndarray],
        condition_name: str,
        category: str,
        strength: float,
        ref_acc: float,
        ref_conf: float,
    ) -> ConditionResult:
        """Run inference on degraded images and compare to reference."""
        confidences = []
        predictions = []

        for img in test_images:
            degraded = apply_degradation(img, condition_name, strength)
            try:
                result = inference_fn(degraded)
                conf = result.get(
                    "confidence", result.get("confidence_raw", 0.0)
                )
                pred = result.get("disease_type", "")
                predictions.append(pred)
                confidences.append(conf / 100.0)
            except Exception:
                confidences.append(0.0)
                predictions.append("")

        avg_confidence = np.mean(confidences) if confidences else 0.0
        confidence_delta = avg_confidence - ref_conf

        # Simple stability: fraction of consistent predictions
        if predictions:
            most_common = (
                max(set(predictions), key=predictions.count)
                if predictions
                else ""
            )
            stability = sum(1 for p in predictions if p == most_common) / len(
                predictions
            )
        else:
            stability = 0.0

        # Accuracy delta approximated as confidence delta for synthetic test
        accuracy_delta = confidence_delta
        passed = accuracy_delta > -0.10 and stability > 0.6

        return ConditionResult(
            condition=condition_name,
            category=category,
            severity=strength,
            accuracy_delta=accuracy_delta,
            confidence_delta=confidence_delta,
            stability_score=stability,
            passed=passed,
        )

    def print_report(self, report: RobustnessReport):
        print("\n" + "=" * 60)
        print("  AgriPulse Robustness Evaluation Report")
        print("=" * 60)
        print(f"\nOverall Score:       {report.overall_score:.1%}")
        print(f"Avg Accuracy Delta: {report.average_accuracy_delta:+.1%}")
        print(f"Worst Category:      {report.worst_category}")

        if report.failure_modes:
            print("\n⚠️  Failure Modes:")
            for fm in report.failure_modes[:5]:
                print(
                    f"  - {fm['condition']} ({fm['category']}): "
                    f"{fm['accuracy_drop']:+.1%} at severity {fm['severity']}"
                )

        print("\nPer-Category:")
        for cat, data in report.to_dict()["per_category"].items():
            print(
                f"  {cat:<12} Δ={data['avg_accuracy_delta']:+.2f}  stab={data['avg_stability']:.3f}"  # noqa: E501
            )
        print("=" * 60 + "\n")


def save_report(report: RobustnessReport, path: Path):
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
