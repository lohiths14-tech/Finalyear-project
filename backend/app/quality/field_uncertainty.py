"""
Uncertainty-Aware Field Prediction System
==========================================
Provides spatial uncertainty quantification for plot-level disease intelligence.
Generates confidence-weighted heatmaps, spatial variance maps, and anomaly region
prioritization for large-scale field analysis.

Components:
  SpatialUncertaintyEstimator  — Per-region uncertainty from feature variance
  BayesianTileAggregator      — Bayesian posterior over overlapping tiles
  AnomalyRegionPrioritizer    — Ranks regions by infection likelihood vs uncertainty
  ConfidenceHeatmapGenerator  — Blends severity + uncertainty into explainable heatmaps
  FieldReliabilityScorer      — Overall field-level reliability assessment
"""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

# ── Uncertainty data structures ────────────────────────────────────────


@dataclass
class RegionUncertainty:
    """Uncertainty estimate for a single field region."""

    region_id: int
    x: int
    y: int
    w: int
    h: int
    mean_severity: float
    std_severity: float
    confidence: float
    is_anomalous: bool
    anomaly_score: float  # higher = more likely an unusual/unknown condition
    recommendation: str  # "normal" | "inspect" | "retake" | "expert_review"


@dataclass
class FieldReliabilityAssessment:
    """Overall field-level reliability."""

    reliability_score: float  # 0–1
    coverage_fraction: float  # fraction of field with reliable predictions
    dominant_confidence: float  # field-average confidence
    uncertainty_hotspots: int  # number of high-uncertainty regions
    anomaly_regions: int  # number of flagged anomaly regions
    overall_label: (
        str  # "highly reliable" | "reliable" | "caution" | "unreliable"
    )
    retake_suggested: bool


# ── Spatial uncertainty estimator ──────────────────────────────────


class SpatialUncertaintyEstimator:
    """
    Estimates per-region uncertainty using feature variance across
    overlapping analysis passes and Monte Carlo dropout.

    Methods:
      - feature_variance: variance of per-tile feature embeddings
      - severity_std: standard deviation of severity estimates
      - bayesian_tile: Bayesian posterior combining tile-level evidence
    """

    @staticmethod
    def estimate_from_tiles(
        severity_values: np.ndarray,
        confidence_values: np.ndarray,
        tile_positions: np.ndarray,
        uncertainty_threshold: float = 0.15,
    ) -> list[RegionUncertainty]:
        """
        Compute uncertainty per tile from severity + confidence variance.

        Args:
          severity_values: (N,) array of per-tile severity scores
          confidence_values: (N,) array of per-tile confidence scores
          tile_positions: (N, 4) array of (x, y, w, h) per tile
          uncertainty_threshold: std above which a region is marked uncertain
        """
        if len(severity_values) == 0:
            return []

        # Compute per-region uncertainty via local spatial neighborhood
        uncertainties: list[RegionUncertainty] = []

        for i in range(len(severity_values)):
            x, y, w, h = tile_positions[i]
            sev = float(severity_values[i])
            conf = float(confidence_values[i])

            # Neighbors: tiles within 2× tile size
            neighbor_dists = np.sqrt(
                (tile_positions[:, 0] - x) ** 2
                + (tile_positions[:, 1] - y) ** 2
            )
            neighbor_mask = neighbor_dists < (w + h) / 2
            neighbor_mask[i] = False  # Exclude self

            if neighbor_mask.sum() >= 2:
                neighbor_sev = severity_values[neighbor_mask]
                sev_std = float(np.std(neighbor_sev))
                neighbor_conf = confidence_values[neighbor_mask]
                conf_std = float(np.std(neighbor_conf))
            else:
                sev_std = 0.0
                conf_std = 0.0

            # Combined uncertainty score
            uncertainty_score = sev_std / 100.0 + (1.0 - conf_std)

            # Anomaly score: how much this region's severity deviates from
            # neighbors
            if neighbor_mask.sum() >= 1:
                neighbor_mean = float(np.mean(severity_values[neighbor_mask]))
                anomaly = abs(sev - neighbor_mean) / max(
                    neighbor_mean + 1e-6, 50.0
                )
            else:
                anomaly = 0.0

            # Recommendations
            if uncertainty_score > 0.5:
                if anomaly > 0.4:
                    rec = "expert_review"
                else:
                    rec = "retake"
            elif uncertainty_score > 0.3:
                if anomaly > 0.3:
                    rec = "inspect"
                else:
                    rec = "normal"
            else:
                rec = "normal"

            uncertainties.append(
                RegionUncertainty(
                    region_id=i,
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    mean_severity=round(sev, 2),
                    std_severity=round(sev_std, 2),
                    confidence=round(conf, 3),
                    is_anomalous=anomaly > 0.4,
                    anomaly_score=round(anomaly, 3),
                    recommendation=rec,
                )
            )

        return uncertainties

    @staticmethod
    def compute_spatial_variance_map(
        severity_map: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Compute per-pixel spatial variance of severity predictions.
        High-variance pixels indicate uncertain boundary regions.
        """
        # Local standard deviation via sliding window
        var_raw = (
            uniform_filter(
                (severity_map**2).astype(np.float64), size=kernel_size
            )
            - uniform_filter(severity_map.astype(np.float64), size=kernel_size)
            ** 2
        )
        var_map = np.sqrt(np.clip(var_raw, 0, None)).astype(np.float32)
        return var_map.astype(np.float32)

    @staticmethod
    def compute_confidence_uncertainty_map(
        confidence_map: np.ndarray,
    ) -> np.ndarray:
        """
        Convert confidence map to uncertainty: 1 - confidence.
        """
        return (1.0 - confidence_map / 100.0).astype(np.float32)


# ── Bayesian tile aggregator ─────────────────────────────────────


class BayesianTileAggregator:
    """
    Combines overlapping tile predictions using Bayesian posterior estimation.
    Accounts for tile confidence as observation precision (inverse variance).

    posterior_mean = sum(w_i * prediction_i) / sum(w_i)
    where w_i = confidence_i / variance_i (precision-weighted)
    """

    @staticmethod
    def aggregate_overlapping_tiles(
        tile_predictions: list[dict],
        tile_confidences: list[float],
        image_shape: tuple[int, int],
        sigma_prior: float = 20.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate overlapping tile predictions into a unified field map.

        Returns: (severity_map, confidence_map, count_map)
        """
        h, w = image_shape[:2]
        weighted_severity = np.zeros((h, w), dtype=np.float64)
        weighted_confidence = np.zeros((h, w), dtype=np.float64)
        count_map = np.zeros((h, w), dtype=np.float64)

        for pred, conf in zip(tile_predictions, tile_confidences):
            if not pred:
                continue

            # Gaussian weight based on confidence
            conf_scale = max(conf, 1e-8)
            weight = conf_scale / (sigma_prior**2 + conf_scale**2)

            # Blend contribution using distance-from-center Gaussian
            x, y = pred.get("cx", w // 2), pred.get("cy", h // 2)
            rw, rh = pred.get("w", w // 4), pred.get("h", h // 4)
            sigma_x, sigma_y = max(rw // 3, 5), max(rh // 3, 5)

            y0 = max(0, y - rh // 2)
            y1 = min(h, y + rh // 2)
            x0 = max(0, x - rw // 2)
            x1 = min(w, x + rw // 2)
            if y1 <= y0 or x1 <= x0:
                continue

            _sub_h, _sub_w = y1 - y0, x1 - x0  # noqa: F841
            ys = np.arange(y0, y1)
            xs = np.arange(x0, x1)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            gauss_w = np.exp(
                -(
                    (xx - x) ** 2 / (2 * sigma_x**2)
                    + (yy - y) ** 2 / (2 * sigma_y**2)
                )
            )

            sev_value = pred.get("severity", 0.0)
            weighted_severity[y0:y1, x0:x1] += sev_value * gauss_w * weight
            weighted_confidence[y0:y1, x0:x1] += conf * gauss_w * weight
            count_map[y0:y1, x0:x1] += gauss_w * weight

        # Normalize
        safe_count = np.where(count_map > 1e-8, count_map, 1.0)
        severity_map = (weighted_severity / safe_count).astype(np.float32)
        confidence_map = (weighted_confidence / safe_count).astype(np.float32)

        return severity_map, confidence_map, count_map.astype(np.float32)


# ── Anomaly region prioritization ──────────────────────────────────────


class AnomalyRegionPrioritizer:
    """
    Ranks field regions by infection likelihood while accounting for uncertainty.
    Prioritizes regions that are:
      1. High-severity AND low uncertainty → strong infection signal
      2. High-severity AND high uncertainty → needs expert review
      3. Low-severity AND low uncertainty → confidently healthy
      4. Low-severity AND high uncertainty → retake suggested

    Used to prioritize human review and alert farmers to urgent areas.
    """  # noqa: E501

    SEVERITY_HIGH = 35.0  # above this is "high severity"
    SEVERITY_MEDIUM = 20.0

    def __init__(self):
        self.priority_weights = {
            "critical": 1.0,
            "urgent": 0.75,
            "review": 0.5,
            "monitor": 0.25,
            "normal": 0.0,
        }

    def prioritize(
        self,
        regions: list[RegionUncertainty],
    ) -> list[dict]:
        """
        Rank regions by priority for human review and action.
        Returns sorted list of region dicts with priority labels.
        """
        scored: list[dict] = []

        for r in regions:
            if r.is_anomalous and r.mean_severity >= self.SEVERITY_MEDIUM:
                if r.std_severity < 10:
                    priority = "critical"
                else:
                    priority = "urgent"
            elif r.is_anomalous or r.std_severity > 15:
                priority = "review"
            elif (
                r.mean_severity >= self.SEVERITY_MEDIUM and not r.is_anomalous
            ):
                priority = "monitor"
            else:
                priority = "normal"

            score = self.priority_weights[priority] * (
                r.mean_severity / 100.0 + (1.0 - r.std_severity / 100.0)
            )

            scored.append(
                {
                    "region_id": r.region_id,
                    "x": r.x,
                    "y": r.y,
                    "w": r.w,
                    "h": r.h,
                    "severity": r.mean_severity,
                    "uncertainty": r.std_severity,
                    "anomaly_score": r.anomaly_score,
                    "confidence": r.confidence,
                    "priority": priority,
                    "priority_score": round(score, 3),
                    "recommendation": r.recommendation,
                    "x_percent": round(r.x / (r.x + r.w + 1e-6) * 100, 2),
                    "y_percent": round(r.y / (r.y + r.h + 1e-6) * 100, 2),
                }
            )

        return sorted(scored, key=lambda x: -x["priority_score"])

    def get_urgent_regions(
        self,
        prioritized: list[dict],
        max_regions: int = 10,
    ) -> list[dict]:
        """Extract the top urgent regions for alert generation."""
        urgent = [
            r for r in prioritized if r["priority"] in ("critical", "urgent")
        ]
        return urgent[:max_regions]


# ── Confidence heatmap generator ─────────────────────────────────────────────


class ConfidenceHeatmapGenerator:
    """
    Generates explainable heatmaps that blend:
      - Severity (primary disease signal)
      - Uncertainty (where the model is unsure)
      - Confidence (how much the model trusts the prediction)

    Creates overlays for farmer visualization with clear legend encoding.
    """

    HEATMAP_COLORMAPS = {
        "severity": cv2.COLORMAP_JET,
        "uncertainty": cv2.COLORMAP_PLASMA,
        "combined": cv2.COLORMAP_JET,
    }

    @staticmethod
    def generate_severity_heatmap(
        severity_map: np.ndarray,
        overlay_image: np.ndarray | None = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Generate a colored severity heatmap."""
        h, w = severity_map.shape[:2]

        # Normalize to [0, 255]
        normalized = cv2.normalize(
            severity_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, colormap)

        if overlay_image is not None:
            overlay = (
                cv2.resize(overlay_image, (w, h))
                if overlay_image.shape[:2][::-1] != (w, h)
                else overlay_image
            )
            result = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
        else:
            result = colored

        return result

    @staticmethod
    def generate_uncertainty_heatmap(
        uncertainty_map: np.ndarray,
        colormap: int = cv2.COLORMAP_PLASMA,
    ) -> np.ndarray:
        """Generate uncertainty heatmap (high uncertainty = warm colors)."""
        normalized = cv2.normalize(
            uncertainty_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        return cv2.applyColorMap(normalized, colormap)

    @staticmethod
    def generate_combined_heatmap(
        severity_map: np.ndarray,
        uncertainty_map: np.ndarray,
        confidence_map: np.ndarray,
        overlay_image: np.ndarray | None = None,
        uncertainty_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Generate a multi-signal heatmap where:
          - High severity → warm colors
          - High uncertainty → bright overlay on warm pixels
          - Low confidence → desaturated output
        """
        h, w = severity_map.shape[:2]

        # Normalize each component
        sev_norm = cv2.normalize(
            severity_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        unc_clean = np.nan_to_num(
            uncertainty_map, nan=0.0, posinf=0.0, neginf=0.0
        )
        unc_norm = cv2.normalize(
            unc_clean, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        conf_norm = cv2.normalize(
            confidence_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Colorize severity
        sev_color = cv2.applyColorMap(sev_norm, cv2.COLORMAP_JET)

        # Uncertainty overlay: white flashes on uncertain regions
        unc_heat = cv2.applyColorMap(unc_norm, cv2.COLORMAP_HOT)
        uncertain_mask = (unc_norm > 150).astype(np.float32)[:, :, np.newaxis]
        combined = (
            sev_color * (1 - uncertain_mask * uncertainty_weight)
            + unc_heat * uncertain_mask * uncertainty_weight
        ).astype(np.uint8)

        # Desaturate based on confidence
        conf_factor = conf_norm[:, :, np.newaxis].astype(np.float32) / 255.0
        result = (combined * (0.3 + 0.7 * conf_factor)).astype(np.uint8)

        if overlay_image is not None:
            ov = (
                cv2.resize(overlay_image, (w, h))
                if overlay_image.shape[:2][::-1] != (w, h)
                else overlay_image
            )
            result = cv2.addWeighted(ov, 0.6, result, 0.4, 0)

        return result

    @staticmethod
    def generate_anomaly_overlay(
        regions: list[RegionUncertainty],
        image_shape: tuple[int, int],
        severity_map: np.ndarray,
    ) -> np.ndarray:
        """
        Draw priority-colored boxes around anomaly regions.
        critical=Red, urgent=Orange, review=Yellow, monitor=Blue
        """
        h, w = image_shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        color_map = {
            "critical": (0, 0, 255),  # Red
            "urgent": (0, 127, 255),  # Orange
            "review": (0, 255, 255),  # Yellow
            "monitor": (255, 165, 0),  # Blue
            "normal": (0, 255, 0),  # Green
        }

        for r in regions:
            color = color_map.get(r.recommendation, (0, 255, 0))
            thickness = 2 if r.recommendation in ("critical", "urgent") else 1
            cv2.rectangle(
                overlay, (r.x, r.y), (r.x + r.w, r.y + r.h), color, thickness
            )

            label = f"{r.mean_severity:.0f}±{r.std_severity:.0f}"
            if r.is_anomalous:
                label += " ANOMALY"
            cv2.putText(
                overlay,
                label,
                (r.x, max(r.y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return overlay

    @staticmethod
    def encode_heatmap_to_base64(
        heatmap: np.ndarray,
        format: str = ".png",
    ) -> str:
        """Encode heatmap to base64 for frontend transport."""
        import base64

        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
        _, buf = cv2.imencode(format, heatmap, encode_param)
        return base64.b64encode(buf).decode("utf-8")


# ── Field reliability scorer ──────────────────────────────────────────


class FieldReliabilityScorer:
    """
    Assesses overall field-level reliability for a complete field scan.

    Considers:
      - Coverage: fraction of field with vegetation
      - Dominant confidence: average confidence across all regions
      - Uncertainty hotspots: regions where model is unsure
      - Anomaly regions: regions flagged as unusual
      - Confidence variance: stability of predictions across field
    """

    @staticmethod
    def assess(
        regions: list[RegionUncertainty],
        coverage: float,
        image_shape: tuple[int, int],
    ) -> FieldReliabilityAssessment:
        """Compute overall field reliability."""
        if not regions:
            return FieldReliabilityAssessment(
                reliability_score=0.0,
                coverage_fraction=coverage,
                dominant_confidence=0.0,
                uncertainty_hotspots=0,
                anomaly_regions=0,
                overall_label="unreliable",
                retake_suggested=True,
            )

        confidences = [r.confidence for r in regions]
        uncertainties = [r.std_severity for r in regions]
        anomalies = [r.is_anomalous for r in regions]
        high_uncertainty = [r for r in regions if r.std_severity > 15]
        expert_reviews = [
            r for r in regions if r.recommendation == "expert_review"
        ]

        avg_conf = float(np.mean(confidences))
        _std_conf = float(np.std(confidences))  # noqa: F841
        avg_unc = float(np.mean(uncertainties))
        cov_frac = coverage / 100.0
        anomaly_frac = sum(anomalies) / len(regions)

        # Reliability score components
        conf_component = min(1.0, avg_conf / 0.75)
        unc_component = max(0.0, 1.0 - avg_unc / 50.0)
        anomaly_component = max(0.0, 1.0 - anomaly_frac * 3.0)
        cov_component = min(1.0, cov_frac / 0.30)

        reliability = (
            0.30 * conf_component
            + 0.25 * unc_component
            + 0.25 * anomaly_component
            + 0.20 * cov_component
        )

        if reliability >= 0.75 and avg_conf >= 0.70:
            label = "highly reliable"
            retake = False
        elif reliability >= 0.55:
            label = "reliable"
            retake = False
        elif reliability >= 0.35:
            label = "caution"
            retake = len(expert_reviews) > 0
        else:
            label = "unreliable"
            retake = True

        return FieldReliabilityAssessment(
            reliability_score=round(reliability, 3),
            coverage_fraction=round(cov_frac, 3),
            dominant_confidence=round(avg_conf, 3),
            uncertainty_hotspots=len(high_uncertainty),
            anomaly_regions=sum(anomalies),
            overall_label=label,
            retake_suggested=retake,
        )

    @staticmethod
    def generate_field_summary(
        assessment: FieldReliabilityAssessment,
        prioritized_regions: list[dict],
        coverage: float,
    ) -> dict:
        """Generate a farmer-friendly field summary."""
        urgent = [
            r
            for r in prioritized_regions
            if r["priority"] in ("critical", "urgent")
        ]
        review = [r for r in prioritized_regions if r["priority"] == "review"]

        if assessment.overall_label == "highly reliable":
            status_message = f"Analysis is highly reliable. {len(urgent)} urgent area(s) detected."  # noqa: E501
        elif assessment.overall_label == "reliable":
            status_message = f"Analysis is reliable. {len(urgent)} urgent and {len(review)} areas need inspection."  # noqa: E501
        elif assessment.overall_label == "caution":
            status_message = f"Some areas show uncertainty. {len(review)} region(s) should be reviewed by an expert."  # noqa: E501
        else:
            status_message = "Analysis confidence is low. Consider rescanning with better conditions."  # noqa: E501

        return {
            "status": assessment.overall_label,
            "reliability_score": assessment.reliability_score,
            "status_message": status_message,
            "coverage_percent": round(coverage, 2),
            "uncertainty_hotspots": assessment.uncertainty_hotspots,
            "anomaly_regions": assessment.anomaly_regions,
            "urgent_areas": len(urgent),
            "review_areas": len(review),
            "retake_suggested": assessment.retake_suggested,
            "top_urgent": urgent[:3],
            "top_review": review[:3],
        }
