"""
Real-World Image Quality Assessment
===================================
Assesses input image quality before inference to provide actionable feedback.
Rejects low-quality images early to prevent unreliable predictions.

Scores computed:
  - Blur score (Laplacian variance)
  - Brightness score (illumination validity)
  - Noise score (sensor noise estimation)
  - Contrast score (dynamic range)
  - Resolution score (pixel density)
  - Shadow score (shadow ratio)
  - Coverage score (vegetation area fraction)

Combined into a single Image Quality Index (IQI) with per-factor breakdown.

Used by the inference pipeline to:
  1. Decide whether to proceed or request retake
  2. Adapt preprocessing strength based on detected conditions
  3. Inform confidence calibration about input quality
  4. Provide farmer with specific improvement suggestions
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityScores:
    """Individual quality component scores, all in [0, 1] where 1 is best."""

    blur: float
    brightness: float
    noise: float
    contrast: float
    resolution: float
    shadow_ratio: float
    vegetation_coverage: float

    @property
    def overall(self) -> float:
        w = {
            "blur": 0.20,
            "brightness": 0.10,
            "noise": 0.10,
            "contrast": 0.15,
            "resolution": 0.10,
            "shadow_ratio": 0.15,
            "vegetation_coverage": 0.20,
        }
        return sum(getattr(self, k) * v for k, v in w.items())

    @property
    def is_acceptable(self) -> bool:
        from ..core.config import settings

        iqi_min = float(
            settings.thresholds.get("quality", {}).get(
                "iqi_min_acceptable", 0.55
            )
        )
        return self.overall >= iqi_min

    def to_dict(self) -> dict:
        return {
            "blur": round(self.blur, 4),
            "brightness": round(self.brightness, 4),
            "noise": round(self.noise, 4),
            "contrast": round(self.contrast, 4),
            "resolution": round(self.resolution, 4),
            "shadow_ratio": round(self.shadow_ratio, 4),
            "vegetation_coverage": round(self.vegetation_coverage, 4),
            "overall": round(self.overall, 4),
            "acceptable": self.is_acceptable,
        }


class ImageQualityAssessor:
    """
    Assess input image quality across 7 dimensions.
    Used as a pre-inference gate and quality feedback source for farmers.
    """

    MIN_DIMENSION = 64  # minimum pixels on smallest axis
    IDEAL_PIXELS = 256**2  # ~640×640 for leaf, ~1024×1024 for field
    MAX_SHADOW_RATIO = 0.45  # above this is problematic
    BRIGHTNESS_LO = 30  # below this is too dark
    BRIGHTNESS_HI = 220  # above this is overexposed
    NOISE_SIGMA_MAX = 25.0  # sigma above this is too noisy

    def __init__(self, field_mode: bool = False):
        """
        field_mode: True for drone/aerial images (expect larger images,
                    lower vegetation density, more shadow variance).
        """
        self.field_mode = field_mode

    def assess(self, image: np.ndarray) -> QualityScores:
        """Full quality assessment. Returns per-factor scores + overall IQI."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        blur = self._blur_score(gray)
        brightness = self._brightness_score(gray)
        noise = self._noise_score(gray)
        contrast = self._contrast_score(gray)
        resolution = self._resolution_score(h, w)
        shadow_ratio = self._shadow_ratio_score(gray)
        veg_coverage = self._vegetation_coverage_score(image)

        return QualityScores(
            blur=blur,
            brightness=brightness,
            noise=noise,
            contrast=contrast,
            resolution=resolution,
            shadow_ratio=shadow_ratio,
            vegetation_coverage=veg_coverage,
        )

    def _blur_score(self, gray: np.ndarray) -> float:
        """
        Laplacian variance — higher variance means sharper edges.
        Also checks for motion blur using directional variance ratio.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # Calibrate: expected range for sharp images is ~100–5000
        score = min(1.0, laplacian_var / 500.0)

        # Motion blur detection: compare horizontal vs vertical gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        var_x, var_y = sobelx.var(), sobely.var()
        if var_x > 1e-8 and var_y > 1e-8:
            ratio = min(var_x, var_y) / max(var_x, var_y)
            # Strong directional bias suggests motion blur
            motion_penalty = max(0.0, (0.5 - ratio) * 0.3)
            score = max(0.0, score - motion_penalty)

        return float(score)

    def _brightness_score(self, gray: np.ndarray) -> float:
        """Check for under/over-exposure. Optimal around mean 90–140."""
        mean_brightness = gray.mean()

        if mean_brightness < self.BRIGHTNESS_LO:
            # Too dark: penalize linearly below threshold
            return float(mean_brightness / self.BRIGHTNESS_LO) * 0.5
        elif mean_brightness > self.BRIGHTNESS_HI:
            # Overexposed: penalize above threshold
            excess = (mean_brightness - self.BRIGHTNESS_HI) / (
                255 - self.BRIGHTNESS_HI
            )
            return max(0.0, 1.0 - excess * 0.5)
        else:
            # Good range — normalize to 0–1
            lo, hi = self.BRIGHTNESS_LO, self.BRIGHTNESS_HI
            normalized = (mean_brightness - lo) / (hi - lo)
            # Parabolic peak at midpoint for optimal score
            score = 2 * normalized * (1 - normalized)
            return float(max(0.0, min(1.0, score + 0.3)))

    def _noise_score(self, gray: np.ndarray) -> float:
        """
        Estimate sensor noise via median absolute deviation (MAD) of high-frequency residuals.
        Noise sigma is estimated as MAD × 1.4826 (Gaussian scaling factor).
        """  # noqa: E501
        # High-frequency residual via median filter difference
        median_filtered = cv2.medianBlur(gray, 3)
        residual = gray.astype(np.float32) - median_filtered.astype(np.float32)
        mad = np.median(np.abs(residual))
        sigma = mad * 1.4826

        score = max(0.0, 1.0 - sigma / self.NOISE_SIGMA_MAX)
        return float(score)

    def _contrast_score(self, gray: np.ndarray) -> float:
        """
        Michelson contrast: (max - min) / (max + min).
        Also consider local contrast via CLAHE histogram spread.
        """
        pixel_min, pixel_max = float(gray.min()), float(gray.max())
        if pixel_max + pixel_min < 1.0:
            return 0.0

        michelson = (pixel_max - pixel_min) / (pixel_max + pixel_min + 1e-6)
        michelson_score = min(1.0, michelson / 0.6)

        # Local contrast: standard deviation of a 16-bin histogram
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        hist_entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        hist_score = min(1.0, hist_entropy / 2.5)

        score = 0.5 * michelson_score + 0.5 * hist_score
        return float(max(0.0, min(1.0, score)))

    def _resolution_score(self, h: int, w: int) -> float:
        """Pixel density relative to ideal reference resolution."""
        min_axis = min(h, w)
        if min_axis < self.MIN_DIMENSION:
            return 0.1
        area = h * w
        score = min(1.0, area / self.IDEAL_PIXELS)
        return float(score)

    def _shadow_ratio_score(self, gray: np.ndarray) -> float:
        """
        Fraction of pixels below 40 (dark) — high shadow ratio degrades analysis.
        """  # noqa: E501
        total = gray.size
        dark_pixels = np.sum(gray < 40)
        ratio = dark_pixels / total
        # Optimal is ~15–30% shadow (natural canopy shade), bad above 45%
        if ratio <= 0.15:
            score = (
                0.7 + ratio * 2
            )  # slight penalty for too little shadow (unnatural)
        elif ratio <= 0.30:
            score = 1.0  # natural range
        elif ratio <= self.MAX_SHADOW_RATIO:
            penalty = (ratio - 0.30) / (self.MAX_SHADOW_RATIO - 0.30)
            score = 1.0 - penalty * 0.5
        else:
            score = 0.5 * (1.0 - ratio)
        return float(max(0.0, min(1.0, score)))

    def _vegetation_coverage_score(self, image: np.ndarray) -> float:
        """
        ExG-based vegetation fraction. Low coverage means mostly soil/background.
        """  # noqa: E501
        img_f = image.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 2], img_f[:, :, 1], img_f[:, :, 0]
        exg = 2.0 * g - r - b
        exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, veg_mask = cv2.threshold(
            exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        coverage = float(veg_mask.sum()) / veg_mask.size

        if self.field_mode:
            # Fields naturally have more soil exposure
            return float(min(1.0, coverage / 0.15))
        else:
            # Leaf images should be mostly vegetation
            return float(min(1.0, coverage / 0.40))


# ── Farmer feedback generator ─────────────────────────────────


class FarmerFeedbackGenerator:
    """
    Translates quality scores and inference confidence into
    actionable, farmer-friendly guidance messages.
    """

    @staticmethod
    def generate_quality_feedback(scores: QualityScores) -> dict:
        from ..core.config import settings

        qt = settings.thresholds.get("quality", {})

        issues = []
        suggestions = []

        if scores.blur < float(qt.get("blur_max", 0.4)):
            issues.append("blurry")
            suggestions.append(
                "Hold the camera steady or ensure the leaf is in focus."
            )
        if scores.brightness < float(qt.get("brightness_min", 0.5)):
            issues.append("poor lighting")
            suggestions.append(
                "Capture in good natural light — avoid harsh sunlight or deep shade."  # noqa: E501
            )
        if scores.noise < float(qt.get("noise_max", 0.5)):
            issues.append("grainy")
            suggestions.append(
                "Improve lighting to reduce digital noise in the image."
            )
        if scores.contrast < float(qt.get("contrast_min", 0.4)):
            issues.append("low contrast")
            suggestions.append(
                "Ensure the leaf has clear distinction from the background."
            )
        if scores.resolution < float(qt.get("resolution_min", 0.3)):
            issues.append("low resolution")
            suggestions.append(
                "Move closer to the leaf and use the highest camera resolution."  # noqa: E501
            )
        if scores.shadow_ratio < float(qt.get("shadow_ratio_min", 0.4)):
            issues.append("excessive shadows")
            suggestions.append(
                "Avoid casting shadows on the leaf during capture."
            )
        if scores.vegetation_coverage < float(
            qt.get("vegetation_coverage_min", 0.3)
        ):
            issues.append("not enough vegetation")
            suggestions.append("Make sure the leaf fills most of the frame.")

        quality_label = (
            "excellent"
            if scores.overall >= 0.85
            else "good"
            if scores.overall >= 0.70
            else "fair"
            if scores.overall >= 0.55
            else "poor"
        )

        return {
            "quality_label": quality_label,
            "iqi_score": round(scores.overall, 3),
            "issues": issues,
            "suggestions": suggestions,
            "proceed_to_inference": scores.is_acceptable,
        }

    @staticmethod
    def generate_confidence_feedback(
        confidence: float,
        low_confidence_threshold: float = 40.0,
    ) -> dict:
        if confidence >= 80:
            level = "high"
            tone = "positive"
            message = "The analysis shows high confidence."
        elif confidence >= 60:
            level = "moderate"
            tone = "informational"
            message = "The analysis is fairly confident but results should be verified."  # noqa: E501
        elif confidence >= 40:
            level = "low"
            tone = "caution"
            message = "Confidence is low. Consider capturing a clearer image for accurate diagnosis."  # noqa: E501
        else:
            level = "uncertain"
            tone = "warning"
            message = "The model cannot reliably analyze this image. Please try again with better conditions."  # noqa: E501

        return {
            "confidence": round(confidence, 2),
            "level": level,
            "tone": tone,
            "message": message,
        }

    @staticmethod
    def combine_feedback(
        quality: dict,
        inference_confidence: float,
        disease_label: str,
    ) -> dict:
        qual_fb = FarmerFeedbackGenerator.generate_quality_feedback(
            QualityScores(
                **{
                    k: v
                    for k, v in quality.items()
                    if k not in ("overall", "acceptable")
                }
            )
        )
        conf_fb = FarmerFeedbackGenerator.generate_confidence_feedback(
            inference_confidence
        )

        # Overall reliability
        reliability_score = quality["overall"] * (inference_confidence / 100.0)
        reliability_label = (
            "very reliable"
            if reliability_score >= 0.6
            else "reliable"
            if reliability_score >= 0.35
            else "uncertain"
            if reliability_score >= 0.15
            else "not reliable"
        )

        # Build combined recommendation
        if disease_label == "Unknown" or conf_fb["level"] in (
            "uncertain",
            "low",
        ):
            recommendation = "Please capture a new image with better conditions for accurate diagnosis."  # noqa: E501
        elif quality["overall"] < 0.55:
            recommendation = "Image quality is low. Consider retaking with the suggestions provided."  # noqa: E501
        else:
            recommendation = "Results are ready. Monitor the crop and apply treatment as recommended."  # noqa: E501

        return {
            "quality": qual_fb,
            "confidence": conf_fb,
            "reliability": {
                "score": round(reliability_score, 3),
                "label": reliability_label,
            },
            "recommendation": recommendation,
            "disease_label": disease_label,
        }
