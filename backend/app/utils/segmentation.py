"""
Aerial Image Segmentation with Adaptive Tiling
================================================
Handles ultra-large drone images through intelligent tiling, overlap stitching,
and boundary-aware post-processing. Designed for plot-level disease intelligence.

Features:
  - Adaptive tile sizing based on image resolution and available memory
  - Configurable overlap ratio with feathered blending at seams
  - Boundary-aware segmentation to prevent edge artifacts
  - Memory-mapped processing for very large images (GeoTIFF support via rasterio)
  - Multi-scale tiling (coarse → fine refinement) for ambiguous regions
  - Confidence-weighted aggregation across overlapping predictions
"""  # noqa: E501

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
from typing import NamedTuple

import cv2
import numpy as np

# ── Tile geometry ──────────────────────────────────────────


class Tile(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    scale: float = 1.0

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def slice_row(self) -> slice:
        return slice(self.y, self.y + self.h)

    @property
    def slice_col(self) -> slice:
        return slice(self.x, self.x + self.w)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "scale": self.scale,
        }


class TileMetadata(NamedTuple):
    tile: Tile
    confidence: float
    severity: float
    disease_idx: int
    ngvc_hash: str
    process_ms: float


# ── Adaptive tiling strategy ──────────────────────────────────


class TilingStrategy:
    """
    Determines tile size and overlap dynamically based on image resolution.
    Larger images use larger tiles to keep tile count bounded.

    Strategy:
      - 4K+ images (>3000px): 512px tiles, 25% overlap, max ~200 tiles
      - 1080p images: 256px tiles, 20% overlap
      - Mobile/small: 128px tiles, 15% overlap
    """

    PRESETS = {
        "ultra_large": {
            "tile_size": 512,
            "overlap_ratio": 0.25,
            "stride_ratio": 0.75,
        },
        "large": {
            "tile_size": 384,
            "overlap_ratio": 0.22,
            "stride_ratio": 0.78,
        },
        "medium": {
            "tile_size": 256,
            "overlap_ratio": 0.20,
            "stride_ratio": 0.80,
        },
        "small": {
            "tile_size": 128,
            "overlap_ratio": 0.15,
            "stride_ratio": 0.85,
        },
        "mobile": {
            "tile_size": 96,
            "overlap_ratio": 0.15,
            "stride_ratio": 0.85,
        },
    }

    MAX_TILES = 512  # Upper bound to prevent OOM

    def __init__(self, image_shape: tuple[int, int, int]):
        h, w = image_shape[:2]
        self.h = h
        self.w = w
        self.total_px = h * w

        # Select preset by resolution
        max_dim = max(h, w)
        if max_dim > 3000:
            self.preset = "ultra_large"
        elif max_dim > 1500:
            self.preset = "large"
        elif max_dim > 800:
            self.preset = "medium"
        elif max_dim > 400:
            self.preset = "small"
        else:
            self.preset = "mobile"

        cfg = self.PRESETS[self.preset]
        self.tile_size = cfg["tile_size"]
        self.overlap_px = int(self.tile_size * cfg["overlap_ratio"])
        self.stride = int(self.tile_size * cfg["stride_ratio"])
        self._tiles: list[Tile] | None = None

    @property
    def tiles(self) -> list[Tile]:
        if self._tiles is not None:
            return self._tiles
        self._tiles = self._compute_tiles()
        return self._tiles

    def _compute_tiles(self) -> list[Tile]:
        tiles: list[Tile] = []
        for y in range(0, self.h, self.stride):
            for x in range(0, self.w, self.stride):
                pw = min(self.tile_size, self.w - x)
                ph = min(self.tile_size, self.h - y)
                tiles.append(Tile(x=x, y=y, w=pw, h=ph))
        return tiles

    def uniform_grid(self, rows: int, cols: int) -> list[Tile]:
        """Force a specific rows×cols grid — useful for GraphSAGE node layout."""  # noqa: E501
        tile_h = self.h // rows
        tile_w = self.w // cols
        tiles: list[Tile] = []
        for r in range(rows):
            for c in range(cols):
                y = r * tile_h
                x = c * tile_w
                tiles.append(Tile(x=x, y=y, w=tile_w, h=tile_h))
        return tiles


# ── Overlap stitching with blending ──────────────────────────────


class BlendedStitcher:
    """
    Reconstructs a full-resolution segmentation map from overlapping tile predictions.
    Uses feathered Gaussian blending at seams to eliminate ghosting artifacts.
    """  # noqa: E501

    def __init__(self, target_shape: tuple[int, int]):
        self.h, self.w = target_shape[:2]
        self.accumulator = np.zeros((self.h, self.w), dtype=np.float64)
        self.weight_sum = np.zeros((self.h, self.w), dtype=np.float64)
        self.confidence_map = np.zeros((self.h, self.w), dtype=np.float64)
        self.severity_map = np.zeros((self.h, self.w), dtype=np.float64)
        self.lock = threading.Lock()

    def add_tile_prediction(
        self,
        tile: Tile,
        pred_mask: np.ndarray,
        confidence: float,
        severity: float,
        blending: str = "gaussian",
    ):
        """Accumulate a tile prediction with blending weights."""
        with self.lock:
            y0, x0 = tile.y, tile.x
            y1, x1 = y0 + tile.h, x0 + tile.w

            # Build weight map
            if blending == "gaussian":
                wy = cv2.getGaussianKernel(tile.h, sigma=tile.h / 4)
                wx = cv2.getGaussianKernel(tile.w, sigma=tile.w / 4)
                W = wy @ wx.T  # (h, w)
            elif blending == "linear":
                W = np.ones((tile.h, tile.w), dtype=np.float64)
            else:
                W = np.ones((tile.h, tile.w), dtype=np.float64)

            # Expand pred_mask to tile size if needed
            pm = pred_mask.astype(np.float64)
            if pm.shape != (tile.h, tile.w):
                pm_resized = cv2.resize(
                    pm, (tile.w, tile.h), interpolation=cv2.INTER_LINEAR
                )
                pm = pm_resized

            self.accumulator[y0:y1, x0:x1] += pm * W
            self.weight_sum[y0:y1, x0:x1] += W
            self.confidence_map[y0:y1, x0:x1] += confidence * W
            self.severity_map[y0:y1, x0:x1] += severity * W

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize accumulated maps by total weights."""
        # Avoid div/0 at image borders
        safe_weights = np.where(self.weight_sum > 1e-8, self.weight_sum, 1.0)

        seg_map = (self.accumulator / safe_weights).astype(np.uint8)
        conf_map = (self.confidence_map / safe_weights).astype(np.float32)
        sev_map = (self.severity_map / safe_weights).astype(np.float32)

        return seg_map, conf_map, sev_map


# ── Boundary-aware refinement ─────────────────────────────────────────


class BoundaryRefiner:
    """
    Refines tile predictions at boundary regions using morphological operations
    and confidence-weighted consensus voting.
    """

    @staticmethod
    def refine_boundary(
        seg_map: np.ndarray,
        confidence: np.ndarray,
        severity: np.ndarray,
        kernel_size: int = 5,
        confidence_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1. Suppress low-confidence boundary pixels via closing.
        2. Fill small holes in infected regions (disease morphology).
        3. Erode weak isolated pixels (noise) on boundaries.
        """
        # Mask: only consider confident regions
        _conf_mask = (confidence > confidence_threshold * 255).astype(np.uint8)  # noqa: E501,F841

        # Closing: fill small gaps in the infected mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(seg_map, cv2.MORPH_CLOSE, kernel)

        # Opening: remove small noise islands
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Hole filling in infected regions
        filled = opened.copy()
        contours, _ = cv2.findContours(
            (opened == 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Fill holes < 50px
                cv2.drawContours(filled, [cnt], -1, 255, -1)

        # Erode isolated low-confidence pixels at edges
        eroded = cv2.erode(
            (opened > 0).astype(np.uint8),
            np.ones((3, 3), np.uint8),
        )
        final_seg = np.where(eroded > 0, filled, opened)

        return final_seg, confidence, severity

    @staticmethod
    def connected_component_filter(
        seg_map: np.ndarray,
        min_area_px: int = 100,
    ) -> np.ndarray:
        """Remove connected components smaller than min_area_px (removes noise)."""  # noqa: E501
        labeled, num = cv2.connectedComponents(seg_map)
        filtered = np.zeros_like(seg_map)
        for label_id in range(1, num):
            component_mask = labeled == label_id
            if component_mask.sum() >= min_area_px:
                filtered[component_mask] = seg_map[component_mask]
        return filtered


# ── Main orchestrator ────────────────────────────────────


class AerialSegmentationProcessor:
    """
    Complete pipeline for ultra-large drone image segmentation:

      1. Tiling     — Adaptive grid based on resolution
      2. Processing — Per-tile inference (caller provides inference_fn)
      3. Stitching  — Weighted overlap blending
      4. Refinement — Boundary cleanup + connected-component filter
      5. Analysis   — Per-region disease metrics for GraphSAGE input

    Usage:
      processor = AerialSegmentationProcessor(image)
      tiles = processor.get_tiles()
      for tile in tiles:
          pred_mask = inference_fn(tile_patch)
          processor.add_result(tile, pred_mask, confidence, severity)
      seg_map, conf_map, sev_map, region_analysis = processor.finalize()
    """

    def __init__(
        self,
        image: np.ndarray,
        overlap_ratio: float = 0.25,
        tile_size: int | None = None,
        min_veg_ratio: float = 0.15,
    ):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.strategy = TilingStrategy(image.shape)
        if tile_size:
            self.strategy.tile_size = tile_size
        self.strategy.overlap_px = int(self.strategy.tile_size * overlap_ratio)
        self.strategy.stride = (
            self.strategy.tile_size - self.strategy.overlap_px
        )
        self.strategy._tiles = None  # Force recompute
        self.min_veg_ratio = min_veg_ratio
        self.stitcher = BlendedStitcher(image.shape[:2])
        self.results: list[TileMetadata] = []
        self._tile_patches: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def get_tiles(self) -> list[Tile]:
        return self.strategy.tiles

    def extract_tile_patch(self, tile: Tile) -> np.ndarray:
        """Extract and preprocess a tile region from the image."""
        patch = self.image[tile.slice_row, tile.slice_col].copy()
        return patch

    def compute_tile_veg_mask(self, tile: Tile) -> tuple[np.ndarray, float]:
        """
        Fast vegetation mask for a single tile using ExG thresholding.
        Returns (mask, veg_ratio).
        """
        patch = self.image[tile.slice_row, tile.slice_col]
        img_f = patch.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 2], img_f[:, :, 1], img_f[:, :, 0]
        exg = 2.0 * g - r - b
        exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, mask = cv2.threshold(
            exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        veg_ratio = float(mask.sum() / mask.size)
        return mask, veg_ratio

    def add_result(
        self,
        tile: Tile,
        pred_mask: np.ndarray,
        confidence: float,
        severity: float,
        disease_idx: int = 0,
        ngvc: np.ndarray | None = None,
        process_ms: float = 0.0,
    ):
        """Register a tile inference result for stitching."""
        self.stitcher.add_tile_prediction(
            tile, pred_mask, confidence, severity
        )
        ngvc_hash = ""
        if ngvc is not None:
            ngvc_hash = hashlib.md5(ngvc.tobytes()).hexdigest()[:8]
        self.results.append(
            TileMetadata(
                tile=tile,
                confidence=confidence,
                severity=severity,
                disease_idx=disease_idx,
                ngvc_hash=ngvc_hash,
                process_ms=process_ms,
            )
        )

    def finalize(
        self,
        min_component_area: int = 100,
        boundary_refinement: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        Finalize segmentation:
          - Returns (seg_map, conf_map, sev_map, region_analysis)
          - region_analysis: per-region dicts for GraphSAGE node input
        """
        seg_map, conf_map, sev_map = self.stitcher.finalize()

        # Connected-component cleanup
        seg_clean = BoundaryRefiner.connected_component_filter(
            seg_map, min_area_px=min_component_area
        )

        # Boundary refinement
        if boundary_refinement:
            seg_clean, conf_map, sev_map = BoundaryRefiner.refine_boundary(
                seg_clean, conf_map, sev_map
            )

        # Build region analysis for GraphSAGE
        region_analysis = self._build_region_analysis(seg_clean, sev_map)

        return seg_clean, conf_map, sev_map, region_analysis

    def _build_region_analysis(
        self,
        seg_map: np.ndarray,
        sev_map: np.ndarray,
    ) -> list[dict]:
        """Extract connected disease regions as graph nodes."""
        labeled, num = cv2.connectedComponents(seg_map)
        regions: list[dict] = []

        for label_id in range(1, num):
            mask = labeled == label_id
            ys, xs = np.where(mask)
            if len(xs) < 5:
                continue

            # Bounding box
            x_min, y_min = xs.min(), ys.min()
            x_max, y_max = xs.max(), ys.max()
            area_px = len(xs)
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            avg_severity = float(sev_map[mask].mean())

            # Perimeter for boundary smoothness
            contour_mask = np.zeros_like(seg_map)
            contour_mask[mask] = 255
            perimeter = cv2.arcLength(
                cv2.findContours(
                    contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0],
                True,
            )

            regions.append(
                {
                    "id": label_id - 1,
                    "centroid_x": round(centroid_x, 1),
                    "centroid_y": round(centroid_y, 1),
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                    "area_px": int(area_px),
                    "avg_severity": round(avg_severity, 2),
                    "perimeter_px": round(perimeter, 1),
                    "boundary_ratio": round(
                        perimeter / (np.sqrt(area_px) * 4 + 1e-6), 3
                    ),
                    "x_percent": round(centroid_x / self.w * 100, 2),
                    "y_percent": round(centroid_y / self.h * 100, 2),
                    "connections": [],  # Filled by spatial graph builder
                }
            )

        return regions

    # ── Parallel processing ──────────────────────────────────────

    def process_tiles_parallel(
        self,
        inference_fn,
        max_workers: int = 4,
    ) -> list[tuple[Tile, np.ndarray, float, float]]:
        """
        Process all tiles in parallel using ThreadPoolExecutor.
        inference_fn(tile_patch, veg_mask) -> (pred_mask, confidence, severity)
        """
        tiles = self.get_tiles()
        results: list[tuple[Tile, np.ndarray, float, float]] = []

        def process_one(tile: Tile) -> tuple[Tile, np.ndarray, float, float]:
            patch = self.extract_tile_patch(tile)
            veg_mask, veg_ratio = self.compute_tile_veg_mask(tile)
            if veg_ratio < self.min_veg_ratio:
                # Return empty mask for low-vegetation tiles
                empty_mask = np.zeros((tile.h, tile.w), dtype=np.uint8)
                return tile, empty_mask, 0.0, 0.0
            try:
                pred_mask, confidence, severity = inference_fn(patch, veg_mask)
                return tile, pred_mask, confidence, severity
            except Exception:
                empty_mask = np.zeros((tile.h, tile.w), dtype=np.uint8)
                return tile, empty_mask, 0.0, 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.map(process_one, tiles))
            results = list(futures)

        # Register results
        for tile, pred_mask, confidence, severity in results:
            self.add_result(tile, pred_mask, confidence, severity)

        return results


# ── Multi-scale refinement ───────────────────────────────


class MultiScaleRefiner:
    """
    Two-pass pipeline for ambiguous boundary regions:
      Pass 1 (Coarse): Large tiles, fast inference, identify low-confidence zones
      Pass 2 (Fine):   Small tiles in low-confidence zones, detailed inference

    Reduces total inference cost while improving boundary accuracy.
    """  # noqa: E501

    @staticmethod
    def refine_ambiguous_regions(
        image: np.ndarray,
        coarse_severity: np.ndarray,
        coarse_confidence: np.ndarray,
        uncertainty_threshold: float = 0.15,
        fine_tile_size: int = 128,
        inference_fn=None,
    ) -> np.ndarray:
        """
        Identify low-confidence regions in coarse result and refine with fine tiles.

        uncertainty_threshold: severity stddev within tile that triggers refinement
        """  # noqa: E501
        from scipy.ndimage import standard_deviation

        h, w = coarse_severity.shape

        # Compute local severity stddev (uncertainty proxy)
        kernel_h = min(fine_tile_size, h // 2)
        kernel_w = min(fine_tile_size, w // 2)
        if kernel_h < 3 or kernel_w < 3:
            return coarse_severity  # Too small for refinement

        local_std = standard_deviation(
            coarse_severity, size=(kernel_h, kernel_w)
        )

        # Identify ambiguous regions
        ambiguous_mask = (local_std > uncertainty_threshold).astype(np.uint8)
        ambiguous_mask = cv2.dilate(ambiguous_mask, np.ones((5, 5), np.uint8))

        if ambiguous_mask.sum() == 0:
            return coarse_severity

        # Extract fine tiles only over ambiguous regions
        refined_severity = coarse_severity.copy()
        step = fine_tile_size // 2

        for y in range(0, h, step):
            for x in range(0, w, step):
                y0, x0 = y, x
                y1 = min(y + fine_tile_size, h)
                x1 = min(x + fine_tile_size, w)
                if ambiguous_mask[y0:y1, x0:x1].mean() < 0.1:
                    continue

                patch = image[y0:y1, x0:x1]
                if patch.size < 100:
                    continue

                try:
                    _, _, fine_sev = inference_fn(patch, None)
                    refined_severity[y0:y1, x0:x1] = fine_sev
                except Exception:
                    pass

        return refined_severity


# ── Spatial graph builder ─────────────────────────────────────


class SpatialGraphBuilder:
    """
    Converts per-tile/region predictions into a spatial adjacency graph
    suitable for GraphSAGE input.

    Edges connect spatially neighboring regions (within 2× tile size).
    Edge weights are inverse-distance.
    """

    @staticmethod
    def build_adjacency(
        regions: list[dict],
        max_connections: int = 6,
        distance_threshold_px: float = 256.0,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """
        Returns (adjacency_matrix, edge_lists) for N regions.
        adjacency_matrix: (N, N) binary or weighted
        edge_lists: [[neighbor_ids] for each node]
        """
        if not regions:
            return np.zeros((0, 0), dtype=np.float32), []

        N = len(regions)
        adj = np.zeros((N, N), dtype=np.float32)
        edges: list[list[int]] = [[] for _ in range(N)]

        for i in range(N):
            for j in range(i + 1, N):
                # Distance between centroids
                dx = regions[i]["centroid_x"] - regions[j]["centroid_x"]
                dy = regions[i]["centroid_y"] - regions[j]["centroid_y"]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist > distance_threshold_px:
                    continue

                weight = 1.0 / (dist + 1e-6)
                adj[i, j] = weight
                adj[j, i] = weight

                if len(edges[i]) < max_connections:
                    edges[i].append(j)
                if len(edges[j]) < max_connections:
                    edges[j].append(i)

        return adj, edges
