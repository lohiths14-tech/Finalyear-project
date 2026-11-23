"""
Geospatial Core — Farm GPS, Field Geometry & Spatial Context
===========================================================
Lightweight geospatial utilities for plot-level agricultural intelligence.
Provides GPS location handling, field boundary geometry, and spatial context
without requiring heavy GIS dependencies.

This module is OPTIONAL — all features degrade gracefully when satellite/
geospatial data is unavailable. The core AI pipeline never depends on it.

Features:
  - GPS coordinate handling (WGS84 / lat-lon)
  - Field boundary geometry from GPS points or satellite polygons
  - Acre/hectare area computation
  - Distance between field points
  - Coordinate transformations
  - Field centroid and bounding box
  - Crop zone segmentation based on coordinates
  - Geospatial metadata enrichment
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

# ── Coordinate system ──────────────────────────────────────────


@dataclass
class GPSCoordinate:
    """A geographic coordinate in WGS84 (EPSG:4326)."""

    latitude: float
    longitude: float
    altitude_m: Optional[float] = None
    accuracy_m: Optional[float] = None
    timestamp: Optional[str] = None

    def is_valid(self) -> bool:
        return -90 <= self.latitude <= 90 and -180 <= self.longitude <= 180

    def distance_to(self, other: "GPSCoordinate") -> float:
        """Haversine distance in meters between two GPS points."""
        return haversine_distance(
            self.latitude,
            self.longitude,
            other.latitude,
            other.longitude,
        )

    def to_tuple(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)

    def to_dict(self) -> dict:
        return {
            "lat": round(self.latitude, 6),
            "lon": round(self.longitude, 6),
            "alt": self.altitude_m,
        }


@dataclass
class FieldGeometry:
    """A field boundary defined by GPS coordinates."""

    boundary: list[GPSCoordinate]
    name: Optional[str] = None
    crop_type: Optional[str] = None
    planting_date: Optional[str] = None

    @property
    def centroid(self) -> GPSCoordinate:
        """Geometric centroid of the field boundary."""
        lats = [p.latitude for p in self.boundary]
        lons = [p.longitude for p in self.boundary]
        return GPSCoordinate(
            latitude=float(np.mean(lats)),
            longitude=float(np.mean(lons)),
        )

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Returns (min_lat, min_lon, max_lat, max_lon)."""
        lats = [p.latitude for p in self.boundary]
        lons = [p.longitude for p in self.boundary]
        return (
            float(np.min(lats)),
            float(np.min(lons)),
            float(np.max(lats)),
            float(np.max(lons)),
        )

    @property
    def approximate_area_sqm(self) -> float:
        """Approximate field area in square meters using Shoelace formula + scaling."""  # noqa: E501
        if len(self.boundary) < 3:
            return 0.0
        # Project to meters at centroid latitude
        ref_lat = self.centroid.latitude
        scale = meters_per_degree_lat(ref_lat)

        # Compute area using shoelace formula on projected coordinates
        projected: list[tuple[float, float]] = []
        for p in self.boundary:
            x = p.longitude * meters_per_degree_lon(ref_lat)
            y = p.latitude * scale
            projected.append((x, y))

        area_sqm = polygon_area(projected)
        return abs(area_sqm)

    @property
    def area_acres(self) -> float:
        return self.approximate_area_sqm / 4046.8564224

    @property
    def area_hectares(self) -> float:
        return self.approximate_area_sqm / 10000.0

    def to_geojson(self) -> dict:
        """Export as GeoJSON Feature."""
        coords = [[p.longitude, p.latitude] for p in self.boundary]
        coords.append(coords[0])  # Close polygon
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "name": self.name or "Unnamed Field",
                "crop_type": self.crop_type,
                "area_ha": round(self.area_hectares, 2),
                "area_acres": round(self.area_acres, 2),
            },
        }

    @classmethod
    def from_bbox(
        cls,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        name: Optional[str] = None,
    ) -> "FieldGeometry":
        """Create a rectangular field from bounding box coordinates."""
        corners = [
            GPSCoordinate(min_lat, min_lon),
            GPSCoordinate(min_lat, max_lon),
            GPSCoordinate(max_lat, max_lon),
            GPSCoordinate(max_lat, min_lon),
        ]
        return cls(boundary=corners, name=name)

    @classmethod
    def from_csv(
        cls, rows: list[dict], lat_col: str = "lat", lon_col: str = "lon"
    ) -> "FieldGeometry":
        """Create field from CSV rows containing lat/lon columns."""
        coords = [
            GPSCoordinate(
                latitude=float(r[lat_col]),
                longitude=float(r[lon_col]),
                altitude_m=float(r.get("alt", 0) or 0),
            )
            for r in rows
            if lat_col in r and lon_col in r
        ]
        return cls(boundary=coords)


# ── Haversine / geodesic utilities ─────────────────────────────────────


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Haversine distance in meters between two WGS84 points.
    Accurate to ~0.5% for distances up to a few thousand km.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + (
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def meters_per_degree_lat(lat: float) -> float:
    """Meters per degree of latitude at a given latitude."""
    return (
        111132.92
        - 559.82 * math.cos(2 * math.radians(lat))
        + 1.175 * math.cos(4 * math.radians(lat))
    )


def meters_per_degree_lon(lat: float) -> float:
    """Meters per degree of longitude at a given latitude."""
    return 111412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(
        3 * math.radians(lat)
    )


def polygon_area(projected: list[tuple[float, float]]) -> float:
    """
    Shoelace formula for polygon area given projected (x, y) coordinates.
    Positive = counter-clockwise, negative = clockwise.
    """
    n = len(projected)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += projected[i][0] * projected[j][1]
        area -= projected[j][0] * projected[i][1]
    return area / 2.0


def buffer_geometry(
    geometry: FieldGeometry,
    buffer_m: float,
) -> FieldGeometry:
    """
    Expand field boundary by buffer_m meters (approximate, ignores curvature).
    """
    ref_lat = geometry.centroid.latitude
    scale_lon = meters_per_degree_lon(ref_lat)
    scale_lat = meters_per_degree_lat(ref_lat)

    buffered_boundary: list[GPSCoordinate] = []
    for p in geometry.boundary:
        # Simple outward offset — approximate for small buffers
        px = p.longitude * scale_lon
        py = p.latitude * scale_lat
        angle = math.atan2(
            p.latitude - geometry.centroid.latitude,
            p.longitude - geometry.centroid.longitude,
        )
        new_x = px + buffer_m * math.cos(angle)
        new_y = py + buffer_m * math.sin(angle)
        buffered_boundary.append(
            GPSCoordinate(
                latitude=new_y / scale_lat,
                longitude=new_x / scale_lon,
            )
        )

    return FieldGeometry(
        boundary=buffered_boundary,
        name=geometry.name,
        crop_type=geometry.crop_type,
    )


# ── NDVI from coordinates (simulated from survey data) ───────────────────


@dataclass
class NDVIBand:
    """An NDVI observation at a specific GPS coordinate and timestamp."""

    lat: float
    lon: float
    ndvi: float  # -1 to +1
    confidence: float  # 0-1
    source: str  # "sentinel2" | "landsat" | "modis" | "simulated"
    timestamp: str  # ISO timestamp


class NDVICloudProvider:
    """
    Lightweight NDVI data provider that attempts real satellite sources.

    Sources tried (in order):
      1. Static map tile API (e.g., Google Static Maps / Mapbox)
      2. Sentinel Hub (EOBrowser API)

    The system NEVER requires satellite imagery for core disease detection.
    NDVI overlays are purely informational for large-field visualization.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: dict[str, NDVIBand] = {}
        self._available = False

    def get_ndvi_at(
        self,
        lat: float,
        lon: float,
        radius_m: float = 100,
        date: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[NDVIBand]:
        """
        Get NDVI value at a coordinate.
        Returns None if no data source is available.
        """
        cache_key = f"{lat:.4f},{lon:.4f},{date}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Attempt satellite API (non-blocking, graceful fallback)
        ndvi_data = self._fetch_satellite_ndvi(lat, lon, radius_m, date)
        if ndvi_data is None:
            return None

        self._cache[cache_key] = ndvi_data
        return ndvi_data

    def get_field_ndvi_grid(
        self,
        geometry: FieldGeometry,
        resolution_m: float = 30,
        date: Optional[str] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generate an NDVI grid over the field area.

        Returns: (ndvi_grid, metadata)
        The grid is a 2D numpy array of NDVI values at resolution_m spacing.

        Grid size is determined by field bounding box.
        Values are NDVI in [-1, 1] range.
        """
        min_lat, min_lon, max_lat, max_lon = geometry.bounding_box

        # Grid dimensions based on resolution
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        ref_lat = (max_lat + min_lat) / 2
        scale_lat = meters_per_degree_lat(ref_lat)
        scale_lon = meters_per_degree_lon(ref_lat)

        rows = max(1, int(lat_range * scale_lat / resolution_m))
        cols = max(1, int(lon_range * scale_lon / resolution_m))
        rows = min(rows, 100)  # Cap at 100×100 for memory
        cols = min(cols, 100)

        lat_step = lat_range / rows if rows > 1 else 0
        lon_step = lon_range / cols if cols > 1 else 0

        ndvi_grid = np.zeros((rows, cols), dtype=np.float32)
        confidences = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                lat_val = min_lat + (r + 0.5) * lat_step
                lon_val = min_lon + (c + 0.5) * lon_step
                ndvi_point = self.get_ndvi_at(lat_val, lon_val, date=date)
                if ndvi_point:
                    ndvi_grid[r, c] = ndvi_point.ndvi
                    confidences[r, c] = ndvi_point.confidence
                else:
                    raise Exception(
                        "Satellite NDVI data unavailable. Real API required."
                    )

        metadata = {
            "rows": rows,
            "cols": cols,
            "resolution_m": resolution_m,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "ref_lat": ref_lat,
            "source": "satellite" if self._available else "none",
            "date": date,
            "coverage": float((confidences > 0.5).sum()) / (rows * cols),
        }

        return ndvi_grid, metadata

    def _fetch_satellite_ndvi(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        date: Optional[str],
    ) -> Optional[NDVIBand]:
        """Attempt to fetch NDVI from satellite API. Returns None on failure."""  # noqa: E501
        # Non-blocking placeholder — real implementation would call
        # Sentinel Hub, Google Earth Engine, or Mapbox Satellite API
        # based on available API key.
        self._available = False
        return None


# ── Geospatial metadata builder ────────────────────────────────────


@dataclass
class GeospatialContext:
    """Enriched geospatial metadata for a field scan."""

    field_geometry: Optional[FieldGeometry] = None
    ndvi_grid: Optional[np.ndarray] = None
    ndvi_metadata: Optional[dict] = None
    ndvi_health_zones: Optional[dict] = None
    field_center: Optional[GPSCoordinate] = None
    nearest_weather_station_km: Optional[float] = None
    elevation_m: Optional[float] = None
    soil_type: Optional[str] = None
    satellite_available: bool = False
    data_source: str = "none"  # "none" | "sentinel2" | "landsat"

    @property
    def is_available(self) -> bool:
        return self.field_geometry is not None

    def to_dict(self) -> dict:
        return {
            "available": self.is_available,
            "data_source": self.data_source,
            "field_center": self.field_center.to_dict()
            if self.field_center
            else None,
            "area_hectares": round(self.field_geometry.area_hectares, 2)
            if self.field_geometry
            else None,
            "area_acres": round(self.field_geometry.area_acres, 2)
            if self.field_geometry
            else None,
            "ndvi_source": self.ndvi_metadata.get("source")
            if self.ndvi_metadata
            else None,
            "ndvi_coverage": self.ndvi_metadata.get("coverage")
            if self.ndvi_metadata
            else None,
            "health_zones": self.ndvi_health_zones,
            "elevation_m": self.elevation_m,
            "soil_type": self.soil_type,
            "satellite_available": self.satellite_available,
        }


class GeospatialContextBuilder:
    """
    Builds geospatial context from optional GPS/crop data.
    This is purely enhancement — the core pipeline works without it.
    """

    def __init__(self, ndvi_provider: Optional[NDVICloudProvider] = None):
        self.ndvi_provider = ndvi_provider or NDVICloudProvider()

    def build(
        self,
        boundary_coords: Optional[list[GPSCoordinate]] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        crop_type: Optional[str] = None,
        field_name: Optional[str] = None,
        enable_ndvi: bool = True,
        ndvi_resolution_m: float = 30,
        date: Optional[str] = None,
    ) -> GeospatialContext:
        """
        Build geospatial context from optional inputs.
        All arguments are optional — returns empty context if none provided.
        """
        if not boundary_coords and (center_lat is None or center_lon is None):
            return GeospatialContext()

        # Build field geometry
        if boundary_coords and len(boundary_coords) >= 3:
            geometry = FieldGeometry(
                boundary=boundary_coords,
                name=field_name,
                crop_type=crop_type,
            )
            center = geometry.centroid
        elif center_lat is not None and center_lon is not None:
            center = GPSCoordinate(center_lat, center_lon)
            # Create a 1-hectare default square around center
            half_deg = math.sqrt(10000) / meters_per_degree_lat(center_lat) / 2
            geometry = FieldGeometry.from_bbox(
                center_lat - half_deg,
                center_lon - half_deg,
                center_lat + half_deg,
                center_lon + half_deg,
                name=field_name,
            )
            geometry.crop_type = crop_type
        else:
            return GeospatialContext()

        # NDVI grid (optional, non-blocking)
        ndvi_grid = None
        ndvi_meta = None
        health_zones = None
        ndvi_available = False

        if enable_ndvi:
            try:
                ndvi_grid, ndvi_meta = self.ndvi_provider.get_field_ndvi_grid(
                    geometry,
                    resolution_m=ndvi_resolution_m,
                    date=date,
                )
                health_zones = self._compute_health_zones(ndvi_grid)
                ndvi_available = self.ndvi_provider._available
            except Exception:
                pass

        return GeospatialContext(
            field_geometry=geometry,
            ndvi_grid=ndvi_grid,
            ndvi_metadata=ndvi_meta,
            ndvi_health_zones=health_zones,
            field_center=center,
            satellite_available=ndvi_available,
            data_source="sentinel2" if ndvi_available else "none",
        )

    @staticmethod
    def _compute_health_zones(ndvi_grid: np.ndarray) -> dict:
        """
        Classify NDVI grid into health zones:
          Very Healthy: NDVI > 0.7
          Healthy: 0.5 < NDVI ≤ 0.7
          Stressed: 0.2 < NDVI ≤ 0.5
          Bare/Soil: NDVI ≤ 0.2
        """
        zones = {
            "very_healthy_pct": 0.0,
            "healthy_pct": 0.0,
            "stressed_pct": 0.0,
            "bare_pct": 0.0,
            "total_px": 0,
        }
        total = ndvi_grid.size
        zones["total_px"] = total
        zones["very_healthy_pct"] = round(
            float((ndvi_grid > 0.7).sum()) / total * 100, 1
        )
        zones["healthy_pct"] = round(
            float(((ndvi_grid > 0.5) & (ndvi_grid <= 0.7)).sum())
            / total
            * 100,
            1,
        )
        zones["stressed_pct"] = round(
            float(((ndvi_grid > 0.2) & (ndvi_grid <= 0.5)).sum())
            / total
            * 100,
            1,
        )
        zones["bare_pct"] = round(
            float((ndvi_grid <= 0.2).sum()) / total * 100, 1
        )
        return zones
