"""
Field Visualization — Satellite Map Overlays, Stress Zones & Geospatial UI
===========================================================================
Lightweight visualization renderer for field-level agricultural maps.
Renders satellite map overlays, NDVI health zones, disease hotspots,
field boundaries, and temporal crop-health trends.

This module is OPTIONAL — all visual outputs gracefully degrade
when geospatial data is unavailable. The core visualization always works.

Renderers:
  NDVIVisualizer       — Color-coded NDVI overlay with health zone legend
  FieldMapRenderer     — Field boundary + satellite base + stress zones
  DiseaseHotspotLayer   — Geospatial disease hotspot overlay
  TemporalTracker      — Temporal crop health trend chart
  SatelliteOverlayComposer — Composite map with all layers
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ── Visualization data structures ──────────────────────────────────


@dataclass
class HealthZone:
    """A classified health zone within a field."""

    label: str  # "very_healthy" | "healthy" | "stressed" | "bare"
    ndvi_range: tuple[float, float]
    color_rgb: tuple[int, int, int]
    pixel_count: int
    percentage: float
    severity: str  # "none" | "low" | "medium" | "high"


NDVI_HEALTH_LEGEND = {
    "very_healthy": HealthZone(
        "Very Healthy", (0.7, 1.0), (0, 180, 0), 0, 0.0, "none"
    ),
    "healthy": HealthZone(
        "Healthy", (0.5, 0.7), (100, 230, 0), 0, 0.0, "none"
    ),
    "stressed": HealthZone(
        "Stressed", (0.2, 0.5), (0, 100, 255), 0, 0.0, "medium"
    ),
    "bare": HealthZone(
        "Bare/Soil", (-1.0, 0.2), (139, 90, 43), 0, 0.0, "high"
    ),
}


@dataclass
class DiseaseHotspot:
    """A geospatial disease hotspot at a GPS location."""

    lat: float
    lon: float
    severity: float
    disease: str
    confidence: float
    area_sqm: float
    trend: str  # "spreading" | "stable" | "shrinking"


@dataclass
class VisualizationResult:
    """Container for all visualization outputs."""

    ndvi_overlay_base64: Optional[str] = None
    field_map_base64: Optional[str] = None
    hotspot_map_base64: Optional[str] = None
    health_summary: Optional[dict] = None
    legend: Optional[dict] = None
    geojson: Optional[dict] = None
    temporal_chart: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "ndvi_overlay": self.ndvi_overlay_base64,
            "field_map": self.field_map_base64,
            "hotspot_map": self.hotspot_map_base64,
            "health_summary": self.health_summary,
            "legend": self.legend,
            "geojson": self.geojson,
            "temporal_chart": self.temporal_chart,
        }


# ── NDVI Visualizer ────────────────────────────────────────


class NDVIVisualizer:
    """
    Renders NDVI grid as a color-coded health overlay.
    Uses standard NDVI color mapping: brown → yellow → green → dark green.
    """

    @staticmethod
    def render_ndvi_grid(
        ndvi_grid: np.ndarray,
        width: int = 512,
        height: int = 512,
        colormap: int = cv2.COLORMAP_JET,
        add_legend: bool = True,
    ) -> np.ndarray:
        """
        Render NDVI grid as a colored heatmap.

        NDVI value → color mapping:
          -1.0 → deep red (bare/water)
           0.0 → yellow (sparse vegetation)
          +0.5 → light green (moderate)
          +1.0 → dark green (dense vegetation)
        """
        h_grid, w_grid = ndvi_grid.shape[:2]

        # Resize grid to target canvas size
        ndvi_resized = cv2.resize(
            ndvi_grid, (width, height), interpolation=cv2.INTER_LINEAR
        )

        # Map [-1, 1] to [0, 255] for colormap
        ndvi_u8 = (
            ((ndvi_resized + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        )

        # Apply colormap
        colored = cv2.applyColorMap(ndvi_u8, colormap)

        if add_legend:
            colored = NDVIVisualizer._add_legend(colored)

        return colored

    @staticmethod
    def _add_legend(canvas: np.ndarray) -> np.ndarray:
        """Add NDVI color bar legend to the canvas."""
        h, w = canvas.shape[:2]
        legend_w = min(40, w // 8)
        legend_x = w - legend_w - 5

        legend = canvas.copy()
        for v in range(256):
            color = cv2.applyColorMap(
                np.array([[[v]]], dtype=np.uint8), cv2.COLORMAP_JET
            )[0, 0]
            y = int(v / 256 * h)
            cv2.line(
                legend,
                (legend_x, y),
                (legend_x + legend_w, y),
                color.tolist(),
                1,
            )

        # Legend labels
        labels = ["+1.0\nDense", "+0.5\nModerate", "0\nSparse", "-1\nBare"]
        for i, label in enumerate(labels):
            y = int(i * h / 4) + h // 8
            cv2.putText(
                legend,
                label,
                (legend_x - 50, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

        return legend

    @staticmethod
    def encode_to_base64(image: np.ndarray, format: str = ".png") -> str:
        """Encode image to base64 string for frontend transport."""
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
        _, buf = cv2.imencode(format, image, encode_param)
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def compute_health_summary(ndvi_grid: np.ndarray) -> dict:
        """Compute per-zone health statistics from NDVI grid."""
        total = ndvi_grid.size
        results: dict[str, float] = {}
        for name, zone in NDVI_HEALTH_LEGEND.items():
            lo, hi = zone.ndvi_range
            count = float(
                ((ndvi_grid > lo) & (ndvi_grid <= hi)).sum()
                if lo > -1
                else (ndvi_grid <= hi).sum()
            )
            pct = round(count / total * 100, 1)
            results[f"{name}_pct"] = pct
            results[f"{name}_px"] = int(count)

        # Dominant zone
        dominant = max(results, key=results.get)
        results["dominant_zone"] = dominant

        # Mean NDVI
        results["mean_ndvi"] = round(float(np.mean(ndvi_grid)), 3)
        results["std_ndvi"] = round(float(np.std(ndvi_grid)), 3)

        return results


# ── Field Map Renderer ───────────────────────────────────


class FieldMapRenderer:
    """
    Renders field boundary on a base map with health zones overlay.
    When satellite base is unavailable, renders a clean vector-style field outline.
    """  # noqa: E501

    @staticmethod
    def render_field_map(
        field_geojson: dict,
        ndvi_overlay: Optional[np.ndarray] = None,
        base_map: Optional[np.ndarray] = None,
        width: int = 512,
        height: int = 512,
        stress_zones: Optional[list[dict]] = None,
        disease_hotspots: Optional[list[DiseaseHotspot]] = None,
    ) -> np.ndarray:
        """
        Render field map with optional NDVI overlay, stress zones, and hotspots.
        """  # noqa: E501
        # Create canvas from base map or NDVI overlay
        if base_map is not None:
            canvas = cv2.resize(base_map, (width, height))
        elif ndvi_overlay is not None:
            canvas = cv2.resize(ndvi_overlay, (width, height))
        else:
            canvas = np.full(
                (height, width, 3), (240, 240, 240), dtype=np.uint8
            )

        # Parse GeoJSON boundary and draw field outline
        coords = field_geojson.get("geometry", {}).get("coordinates", [[]])[0]
        if coords:
            # Convert lat/lon to pixel coordinates
            points = FieldMapRenderer._coords_to_points(coords, width, height)
            pts = np.array(points, dtype=np.int32)

            # Draw field boundary
            cv2.polylines(
                canvas, [pts], isClosed=True, color=(0, 200, 0), thickness=2
            )
            cv2.fillPoly(
                canvas, [pts], color=(0, 200, 0), lineType=cv2.LINE_AA
            )
            alpha = 0.25
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 200, 0))
            canvas = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)

            # Draw boundary
            cv2.polylines(
                canvas, [pts], isClosed=True, color=(0, 130, 0), thickness=2
            )

        # Draw stress zones
        if stress_zones:
            for zone in stress_zones:
                zone_pts = FieldMapRenderer._coords_to_points(
                    zone.get("coords", []), width, height
                )
                if len(zone_pts) >= 3:
                    zpts = np.array(zone_pts, dtype=np.int32)
                    severity = zone.get("severity", "low")
                    color = {
                        "low": (0, 180, 255),
                        "medium": (0, 100, 255),
                        "high": (0, 0, 255),
                    }.get(severity, (0, 180, 255))
                    cv2.fillPoly(canvas, [zpts], color=color)
                    cv2.polylines(
                        canvas, [zpts], isClosed=True, color=color, thickness=1
                    )

        # Draw disease hotspots
        if disease_hotspots:
            for hs in disease_hotspots:
                px, py = FieldMapRenderer._gps_to_pixel(
                    hs.lat, hs.lon, coords, width, height
                )
                severity = hs.severity
                radius = int(5 + severity * 15)
                color = (
                    (0, 0, 255)
                    if severity > 0.6
                    else (0, 165, 255)
                    if severity > 0.3
                    else (0, 255, 255)
                )
                cv2.circle(canvas, (px, py), radius, color, -1)
                cv2.circle(canvas, (px, py), radius, (255, 255, 255), 2)

        # Add field name label
        props = field_geojson.get("properties", {})
        field_name = props.get("name", "Field")
        area_ha = props.get("area_ha", 0)
        label = f"{field_name} ({area_ha} ha)"
        cv2.putText(
            canvas,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        return canvas

    @staticmethod
    def _coords_to_points(
        coords: list, width: int, height: int
    ) -> list[tuple[int, int]]:
        """Convert [lon, lat] GeoJSON coordinates to pixel points."""
        if not coords:
            return []

        lats = [c[1] for c in coords]
        lons = [c[0] for c in coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        points = []
        for c in coords:
            x = int(
                (c[0] - min_lon) / (max_lon - min_lon + 1e-8) * (width - 20)
                + 10
            )
            y = int(
                (c[1] - min_lat) / (max_lat - min_lat + 1e-8) * (height - 20)
                + 10
            )
            points.append(
                (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
            )
        return points

    @staticmethod
    def _gps_to_pixel(
        lat: float, lon: float, boundary_coords: list, width: int, height: int
    ) -> tuple[int, int]:
        """Project a GPS point to pixel coordinates within field boundary."""
        if not boundary_coords:
            return width // 2, height // 2
        lats = [c[1] for c in boundary_coords]
        lons = [c[0] for c in boundary_coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        x = int(
            (lon - min_lon) / (max_lon - min_lon + 1e-8) * (width - 20) + 10
        )
        y = int(
            (lat - min_lat) / (max_lat - min_lat + 1e-8) * (height - 20) + 10
        )
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))


# ── Disease Hotspot Layer ─────────────────────────────────────


class DiseaseHotspotLayer:
    """
    Renders disease hotspots on the field map.
    Hotspots are ranked by severity and displayed with trend indicators.
    """

    @staticmethod
    def render_hotspot_overlay(
        hotspots: list[DiseaseHotspot],
        field_geojson: dict,
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """Render disease hotspot overlay on blank canvas."""
        canvas = np.full((height, width, 3), (0, 0, 0), dtype=np.uint8)

        coords = field_geojson.get("geometry", {}).get("coordinates", [[]])[0]
        if not coords:
            coords = [[0, 0]]

        for i, hs in enumerate(
            sorted(hotspots, key=lambda h: -h.severity)[:20]
        ):
            px, py = FieldMapRenderer._gps_to_pixel(
                hs.lat, hs.lon, coords, width, height
            )

            # Size based on area
            radius = int(np.sqrt(hs.area_sqm / np.pi) / 5) + 5
            radius = min(radius, 50)

            # Color by severity
            if hs.severity > 0.7:
                color = (0, 0, 255)  # Red
            elif hs.severity > 0.4:
                color = (0, 127, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow

            # Trend arrow
            if hs.trend == "spreading":
                arrow = "↗"
            elif hs.trend == "shrinking":
                arrow = "↘"
            else:
                arrow = "→"

            label = f"#{i + 1} {hs.disease[:12]} {arrow}"
            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                label,
                (px + radius + 5, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

        return canvas

    @staticmethod
    def hotspots_to_geojson(hotspots: list[DiseaseHotspot]) -> dict:
        """Export hotspots as GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [h.lon, h.lat],
                    },
                    "properties": {
                        "disease": h.disease,
                        "severity": round(h.severity, 2),
                        "confidence": round(h.confidence, 2),
                        "area_sqm": round(h.area_sqm, 1),
                        "trend": h.trend,
                    },
                }
                for h in hotspots
            ],
        }


# ── Temporal Tracker ───────────────────────────────────────


class TemporalTracker:
    """
    Builds temporal crop health trend data from historical NDVI/disease data.
    Generates time-series data for the frontend chart.
    """

    @staticmethod
    def build_ndvi_trend(
        ndvi_history: list[float],
        dates: list[str],
        predicted_days: int = 5,
    ) -> dict:
        """
        Build NDVI trend chart from historical observations.

        Args:
          ndvi_history: list of mean NDVI values per date
          dates: corresponding ISO date strings
          predicted_days: number of future days to forecast
        """
        if not ndvi_history:
            return {}

        # Compute trend slope
        x = np.arange(len(ndvi_history))
        slope = float(np.polyfit(x, ndvi_history, 1)[0]) if len(x) > 1 else 0.0
        trend_direction = (
            "improving"
            if slope > 0.001
            else "declining"
            if slope < -0.001
            else "stable"
        )

        # Forecast future NDVI values
        last_ndvi = ndvi_history[-1]
        forecast = []
        for d in range(1, predicted_days + 1):
            forecasted = last_ndvi + slope * (len(ndvi_history) - 1 + d)
            forecasted = max(-1.0, min(1.0, forecasted))
            forecast.append(round(forecasted, 3))

        # Actual data points
        actual = [round(v, 3) for v in ndvi_history]

        return {
            "actual": [
                {"date": dates[i], "ndvi": actual[i]}
                for i in range(len(actual))
            ],
            "forecast": [
                {"date": f"+{d}d", "ndvi": forecast[d - 1]}
                for d in range(1, predicted_days + 1)
            ],
            "trend": trend_direction,
            "slope": round(slope, 5),
            "current_ndvi": round(last_ndvi, 3),
            "ndvi_delta": round(last_ndvi - ndvi_history[0], 3)
            if ndvi_history
            else 0.0,
        }

    @staticmethod
    def build_disease_trend(
        severity_history: list[float],
        dates: list[str],
    ) -> dict:
        """Build disease severity trend from historical observations."""
        if not severity_history:
            return {}

        x = np.arange(len(severity_history))
        slope = (
            float(np.polyfit(x, severity_history, 1)[0]) if len(x) > 1 else 0.0
        )

        trend = (
            "spreading"
            if slope > 0.5
            else "shrinking"
            if slope < -0.5
            else "stable"
        )

        return {
            "data": [
                {"date": dates[i], "severity": round(severity_history[i], 2)}
                for i in range(len(severity_history))
            ],
            "trend": trend,
            "slope": round(slope, 3),
            "current_severity": round(severity_history[-1], 2),
            "peak_severity": round(max(severity_history), 2),
            "peak_date": dates[severity_history.index(max(severity_history))]
            if severity_history
            else None,
        }


# ── Composite composer ─────────────────────────────────────────


class SatelliteOverlayComposer:
    """
    Main orchestrator for field visualization.
    Composes all layers into a single VisualizationResult.
    """

    @staticmethod
    def compose(
        field_geojson: Optional[dict] = None,
        ndvi_grid: Optional[np.ndarray] = None,
        ndvi_metadata: Optional[dict] = None,
        health_zones: Optional[dict] = None,
        stress_zones: Optional[list[dict]] = None,
        disease_hotspots: Optional[list[DiseaseHotspot]] = None,
        temporal_data: Optional[dict] = None,
        width: int = 512,
        height: int = 512,
    ) -> VisualizationResult:
        """
        Compose all geospatial visualization layers into a result object.
        All arguments are optional — degrades gracefully when data is missing.
        """
        result = VisualizationResult()

        # GeoJSON
        if field_geojson:
            result.geojson = field_geojson

        # NDVI overlay
        if ndvi_grid is not None:
            ndvi_colored = NDVIVisualizer.render_ndvi_grid(
                ndvi_grid, width, height
            )
            result.ndvi_overlay_base64 = NDVIVisualizer.encode_to_base64(
                ndvi_colored
            )
            result.health_summary = NDVIVisualizer.compute_health_summary(
                ndvi_grid
            )

        # Field map
        if field_geojson:
            field_map = FieldMapRenderer.render_field_map(
                field_geojson,
                ndvi_overlay=None,
                stress_zones=stress_zones,
                disease_hotspots=disease_hotspots,
                width=width,
                height=height,
            )
            result.field_map_base64 = NDVIVisualizer.encode_to_base64(
                field_map
            )

        # Hotspot overlay
        if disease_hotspots and field_geojson:
            hotspot_overlay = DiseaseHotspotLayer.render_hotspot_overlay(
                disease_hotspots,
                field_geojson,
                width,
                height,
            )
            result.hotspot_map_base64 = NDVIVisualizer.encode_to_base64(
                hotspot_overlay
            )

        # Legend
        result.legend = {
            zone.label: {
                "color": list(zone.color_rgb),
                "ndvi_range": list(zone.ndvi_range),
                "severity": zone.severity,
            }
            for zone in NDVI_HEALTH_LEGEND.values()
        }

        # Temporal chart
        if temporal_data:
            result.temporal_chart = temporal_data

        return result
