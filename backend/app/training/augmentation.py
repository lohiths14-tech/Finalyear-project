"""
Agricultural Image Augmentation Pipeline
======================================
Domain-specific augmentation for aerial and leaf-level crop imagery.
Goes beyond generic rotation/brightness/noise to include field-specific
transforms: blur, occlusion, motion blur, camera distortions, illumination
variations, perspective changes, and real field conditions.

Features:
  - FieldConditionAugmenter: composite field-condition simulation
  - BlurAugmenter: Gaussian, motion, defocus, lens-aberration blur
  - OcclusionAugmenter: leaf overlap, soil patches, shadow occlusions
  - IlluminationAugmenter: sun glare, canopy shade, uneven lighting
  - PerspectiveAugmenter: viewpoint variation, pitch/roll from drone flight
  - ColorAugmenter: disease-induced discoloration, nutrient deficiency shifts
  - GeometricAugmenter: affine transforms for arbitrary orientations
  - CompositeAugmenter: randomly chains multiple augmenters
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import cv2
import numpy as np

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# ── Base ───────────────────────────────────────


@dataclass
class AugmentationConfig:
    """Probability and intensity bounds for each augmentation type."""

    blur_prob: float = 0.4
    blur_strength_range: tuple[float, float] = (1.5, 4.0)
    occlusion_prob: float = 0.3
    occlusion_area_range: tuple[float, float] = (0.05, 0.25)
    illumination_prob: float = 0.5
    illumination_strength: float = 0.3
    perspective_prob: float = 0.25
    perspective_degrees: float = 8.0
    color_shift_prob: float = 0.35
    color_shift_strength: float = 0.2
    geometric_prob: float = 0.3
    noise_prob: float = 0.4
    noise_sigma: float = 0.02


class Augmenter:
    """Base augmenter interface."""

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        raise NotImplementedError


# ── Field condition composite ────────────────────────────────────


class FieldConditionAugmenter(Augmenter):
    """
    Simulates realistic field conditions by composing multiple degradations:

    - Low-resolution downsampling → bilinear upscale (motion blur proxy)
    - Uneven illumination (canopy shade patterns)
    - Color cast (soil reflection)
    - Shadow overlay (cloud cover)
    - JPEG-like compression artifacts
    """

    def __init__(self, intensity: float = 1.0):
        self.intensity = intensity

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        img = image.astype(np.float32)
        h, w = img.shape[:2]

        # 1. Resolution degradation (simulate low-quality drone sensor)
        if rng.random() < 0.4 * self.intensity:
            scale = rng.uniform(0.6, 0.9)
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 50 and new_w > 50:
                small = cv2.resize(
                    img, (new_w, new_h), interpolation=cv2.INTER_BILINEAR
                )
                img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Uneven illumination (canopy shade)
        if rng.random() < 0.5 * self.intensity:
            shade_mask = self._generate_canopy_shade(h, w, rng)
            img = img * (1.0 - shade_mask * 0.3 * self.intensity)

        # 3. Color cast (soil reflection)
        if rng.random() < 0.3 * self.intensity:
            cast_strength = rng.uniform(0.05, 0.15) * self.intensity
            img[:, :, 0] = np.clip(
                img[:, :, 0] * (1.0 + cast_strength), 0, 255
            )  # Blue cast

        # 4. Shadow overlay
        if rng.random() < 0.25 * self.intensity:
            shadow_strength = rng.uniform(0.1, 0.3) * self.intensity
            shadow_mask = self._generate_shadow_overlay(h, w, rng)
            img = img * (1.0 - shadow_mask * shadow_strength)

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _generate_canopy_shade(h: int, w: int, rng) -> np.ndarray:
        _size = max(32, min(h, w) // 4)  # noqa: F841
        kernel = rng.integers(3, 7)
        base = rng.uniform(0.1, 0.4, (h, w)).astype(np.float32)
        blurred = cv2.GaussianBlur(base, (kernel, kernel), 0)
        return blurred

    @staticmethod
    def _generate_shadow_overlay(h: int, w: int, rng) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        # Random shadow region (ellipse)
        cx, cy = (
            rng.integers(w // 4, 3 * w // 4),
            rng.integers(h // 4, 3 * h // 4),
        )
        rx, ry = rng.integers(w // 6, w // 3), rng.integers(h // 6, h // 3)
        for y in range(h):
            for x in range(w):
                dx = (x - cx) / (rx + 1e-6)
                dy = (y - cy) / (ry + 1e-6)
                d = dx * dx + dy * dy
                if d < 1.0:
                    mask[y, x] = float(1.0 - d)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask


# ── Blur augmenters ────────────────────────────────────────


class MotionBlurAugmenter(Augmenter):
    """
    Motion blur simulating drone flight velocity and direction changes.
    """

    def __init__(
        self, kernel_range: tuple[int, int] = (5, 15), prob: float = 0.5
    ):
        self.kernel_range = kernel_range
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        kernel_size = rng.integers(*self.kernel_range)
        if kernel_size % 2 == 0:
            kernel_size += 1
        angle = rng.uniform(0, 180)
        kernel = self._motion_kernel(kernel_size, angle)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
        k = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i in range(size):
            for j in range(size):
                x = int(cos_a * (i - center) + sin_a * (j - center))
                if abs(x) <= size // 2:
                    k[center, x + center] = 1.0
        k = k / k.sum()
        return k


class DefocusBlurAugmenter(Augmenter):
    """
    Defocus blur simulating depth-of-field limits in close-range drone captures.
    """  # noqa: E501

    def __init__(
        self, radius_range: tuple[int, int] = (3, 12), prob: float = 0.4
    ):
        self.radius_range = radius_range
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        radius = rng.integers(*self.radius_range)
        if radius % 2 == 0:
            radius += 1
        kernel = cv2.getGaussianKernel(radius * 2 + 1, sigma=radius / 3.0)
        kernel = kernel @ kernel.T
        return cv2.filter2D(image, -1, kernel / kernel.sum())


class LensAberrationAugmenter(Augmenter):
    """
    Chromatic aberration (RGB channel offset) simulating cheap lens optics.
    """

    def __init__(
        self, offset_range: tuple[int, int] = (2, 6), prob: float = 0.3
    ):
        self.offset_range = offset_range
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        img_f = image.astype(np.float32)
        offset = rng.integers(*self.offset_range)
        h, w = img_f.shape[:2]
        result = np.zeros_like(img_f)

        for c in range(3):
            src = img_f[:, :, c]
            if c == 0:
                shifted = np.pad(src, ((offset, 0), (0, 0)), mode="edge")[
                    offset:, :
                ]
            elif c == 2:
                shifted = np.pad(src, ((0, offset), (0, 0)), mode="edge")[
                    :-offset, :
                ]
            else:
                shifted = src
            result[:, :, c] = (
                shifted[:, :w] if shifted.shape[1] > w else shifted
            )

        return np.clip(result, 0, 255).astype(np.uint8)


# ── Occlusion augmenters ───────────────────────────────────


class LeafOcclusionAugmenter(Augmenter):
    """
    Simulates overlapping leaf occlusion common in dense canopy imagery.
    Draws elliptical shadow patches representing occluding leaves.
    """

    def __init__(self, prob: float = 0.35, max_occlusions: int = 3):
        self.prob = prob
        self.max_occlusions = max_occlusions

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        img = image.copy()
        h, w = img.shape[:2]
        num = rng.integers(1, self.max_occlusions + 1)

        for _ in range(num):
            cx = rng.integers(w // 10, 9 * w // 10)
            cy = rng.integers(h // 10, 9 * h // 10)
            rx = rng.integers(w // 12, w // 6)
            ry = rng.integers(h // 12, h // 6)
            angle = rng.uniform(0, 180)
            alpha = rng.uniform(0.3, 0.6)

            overlay = img.copy()
            cv2.ellipse(
                overlay, (cx, cy), (rx, ry), angle, 0, 360, (10, 10, 10), -1
            )
            img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        return img


class SoilPatchAugmenter(Augmenter):
    """
    Simulates soil patches (exposed earth) appearing under canopy gaps.
    """

    def __init__(self, prob: float = 0.3, max_patches: int = 2):
        self.prob = prob
        self.max_patches = max_patches

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        img = image.copy()
        h, w = img.shape[:2]
        num = rng.integers(1, self.max_patches + 1)

        for _ in range(num):
            x = rng.integers(w // 10, 8 * w // 10)
            y = rng.integers(h // 10, 8 * h // 10)
            rw = rng.integers(w // 15, w // 6)
            rh = rng.integers(h // 15, h // 6)

            soil_color = np.array(
                [
                    rng.integers(80, 140),
                    rng.integers(60, 100),
                    rng.integers(30, 70),
                ],
                dtype=np.uint8,
            )
            patch = np.full((rh, rw, 3), soil_color, dtype=np.uint8)
            patch = cv2.GaussianBlur(patch, (5, 5), 0)

            x1, y1 = max(0, x - rw // 2), max(0, y - rh // 2)
            x2, y2 = min(w, x + rw // 2), min(h, y + rh // 2)
            img[y1:y2, x1:x2] = patch[: y2 - y1, : x2 - x1]

        return img


# ── Illumination augmenters ───────────────────────────────────


class SunGlareAugmenter(Augmenter):
    """
    Simulates direct sun glare on sensor — bright white flare streaks.
    """

    def __init__(self, prob: float = 0.25):
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        img = image.astype(np.float32)
        h, w = img.shape[:2]
        cx, cy = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, h // 4)
        radius = rng.integers(w // 8, w // 4)

        overlay = np.zeros_like(img)
        for r in range(radius, 0, -1):
            alpha = rng.uniform(0.05, 0.2)
            cv2.circle(overlay, (cx, cy), r, (255, 255, 255), -1)
            img = np.clip(img * (1 - alpha) + overlay * alpha, 0, 255)

        # Lens flare streaks
        for _ in range(rng.integers(2, 6)):
            angle = rng.uniform(0, 360)
            length = rng.integers(w // 10, w // 4)
            ex = int(cx + length * math.cos(math.radians(angle)))
            ey = int(cy + length * math.sin(math.radians(angle)))
            cv2.line(overlay, (cx, cy), (ex, ey), (200, 200, 200), 2)
        img = np.clip(img + overlay * 0.3, 0, 255)
        return img.astype(np.uint8)


class UnevenIlluminationAugmenter(Augmenter):
    """
    Simulates uneven field illumination — bright patches surrounded by shade.
    """

    def __init__(self, prob: float = 0.45, strength: float = 0.35):
        self.prob = prob
        self.strength = strength

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        img_f = image.astype(np.float32) / 255.0
        h, w = img_f.shape[:2]

        # Generate illumination field using random Gaussian blobs
        for _ in range(rng.integers(2, 5)):
            cx = rng.integers(0, w)
            cy = rng.integers(0, h)
            rx = rng.integers(w // 8, w // 2)
            ry = rng.integers(h // 8, h // 2)
            strength = rng.uniform(-self.strength, self.strength)

            y, x = np.ogrid[:h, :w]
            dx = (x - cx) / (rx + 1e-6)
            dy = (y - cy) / (ry + 1e-6)
            brightness = np.exp(-(dx * dx + dy * dy)) * strength
            img_f = img_f * (1.0 + brightness)
        return np.clip(img_f * 255, 0, 255).astype(np.uint8)


# ── Geometric & perspective augmenters ───────────────────────────────


class PerspectiveAugmenter(Augmenter):
    """
    Simulates drone pitch/roll attitude changes by applying perspective transforms.
    """  # noqa: E501

    def __init__(self, max_degrees: float = 8.0, prob: float = 0.3):
        self.max_degrees = max_degrees
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        h, w = image.shape[:2]
        roll = rng.uniform(-self.max_degrees, self.max_degrees)
        pitch = rng.uniform(-self.max_degrees, self.max_degrees)

        # Build perspective transform matrix
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        _cx, _cy = w / 2, h / 2  # noqa: F841
        scale_x = 1.0 + rng.uniform(-0.05, 0.05)
        scale_y = 1.0 + rng.uniform(-0.05, 0.05)

        roll_rad = math.radians(roll)
        _roll_cos, _roll_sin = math.cos(roll_rad), math.sin(roll_rad)  # noqa: E501,F841
        dst_pts = np.float32(
            [
                [
                    w * 0.0 * scale_x - w * 0.05 + roll * 5,
                    h * 0.0 * scale_y + pitch * 3,
                ],
                [
                    w * 1.0 * scale_x + w * 0.05 + roll * 3,
                    h * 0.0 * scale_y - pitch * 3,
                ],
                [
                    w * 0.0 * scale_x - w * 0.05 - roll * 3,
                    h * 1.0 * scale_y - pitch * 3,
                ],
                [
                    w * 1.0 * scale_x + w * 0.05 - roll * 5,
                    h * 1.0 * scale_y + pitch * 3,
                ],
            ]
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(
            image, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )


class GeometricAugmenter(Augmenter):
    """
    Affine transforms: rotation, scaling, horizontal/vertical flipping.
    """

    def __init__(
        self,
        max_rotation: float = 15.0,
        scale_range: tuple[float, float] = (0.8, 1.2),
        prob: float = 0.3,
    ):
        self.max_rotation = max_rotation
        self.scale_range = scale_range
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        angle = rng.uniform(-self.max_rotation, self.max_rotation)
        scale = rng.uniform(*self.scale_range)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )


# ── Color augmentation ─────────────────────────────────────────


class ColorShiftAugmenter(Augmenter):
    """
    Simulates disease-induced discoloration and nutrient deficiency color shifts.
    """  # noqa: E501

    def __init__(self, strength: float = 0.2, prob: float = 0.4):
        self.strength = strength
        self.prob = prob

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image

        img_f = image.astype(np.float32)
        hsv = cv2.cvtColor(img_f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(
            np.float32
        )

        # Hue shift: disease discoloration
        hue_shift = rng.uniform(-20, 20)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # Saturation shift: nutrient deficiency
        sat_shift = rng.uniform(-self.strength * 50, self.strength * 30)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)

        # Value shift: water stress
        val_shift = rng.uniform(-self.strength * 40, self.strength * 40)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 255)

        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(
            np.float32
        )
        return np.clip(shifted, 0, 255).astype(np.uint8)


# ── Albumentations Wrapper ────────────────────────────────────


class AlbumentationsWrapper(Augmenter):
    """
    Wraps modern albumentations transforms for realistic agricultural augmentations
    including RandomErasing (CoarseDropout) to simulate insect bites.
    """  # noqa: E501

    def __init__(self):
        if not ALBUMENTATIONS_AVAILABLE:
            self.transform = None
            return

        self.transform = A.Compose(
            [
                A.MotionBlur(p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5), src_radius=100, p=0.1
                ),
                A.ISONoise(p=0.2),
                A.RandomFog(p=0.1),
                A.RandomRain(p=0.05),
                A.ImageCompression(quality_lower=60, quality_upper=90, p=0.15),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.15,
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.15),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    fill_value=0,
                    p=0.25,
                ),  # Random Erasing (insect bites)
            ]
        )

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if self.transform is None:
            return image
        return self.transform(image=image)["image"]


# ── Composite augmenter ────────────────────────────────────


class CompositeAugmenter:
    """
    Randomly chains multiple augmenters with configurable probabilities.
    """

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()
        self.augmenters: list[tuple[Augmenter, float]] = [
            (MotionBlurAugmenter(prob=0.6), self.config.blur_prob),
            (DefocusBlurAugmenter(prob=0.5), self.config.blur_prob * 0.8),
            (LensAberrationAugmenter(prob=0.4), self.config.blur_prob * 0.5),
            (LeafOcclusionAugmenter(prob=0.5), self.config.occlusion_prob),
            (SoilPatchAugmenter(prob=0.5), self.config.occlusion_prob * 0.6),
            (SunGlareAugmenter(prob=0.3), self.config.illumination_prob * 0.4),
            (
                UnevenIlluminationAugmenter(prob=0.6),
                self.config.illumination_prob,
            ),
            (PerspectiveAugmenter(prob=0.5), self.config.perspective_prob),
            (GeometricAugmenter(prob=0.5), self.config.geometric_prob),
            (ColorShiftAugmenter(prob=0.6), self.config.color_shift_prob),
            (FieldConditionAugmenter(intensity=0.8), 0.5),
            (
                AlbumentationsWrapper(),
                1.0,
            ),  # Albumentations has its own internal probabilities
        ]

    def __call__(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        result = image.copy()
        random.shuffle(self.augmenters)

        for augmenter, prob in self.augmenters:
            if rng.random() < prob:
                result = augmenter(result, rng)

        return result


# ── NGVC-level augmentation ───────────────────────────────────


class NGVCNoiseAugmenter:
    """
    Augmentation that operates directly on NGVC histograms.
    Applies Gaussian noise, peak shifting, and histogram smoothing.
    """

    def __init__(
        self,
        noise_sigma: float = 0.015,
        peak_shift_prob: float = 0.3,
        smoothing_prob: float = 0.2,
    ):
        self.noise_sigma = noise_sigma
        self.peak_shift_prob = peak_shift_prob
        self.smoothing_prob = smoothing_prob

    def __call__(
        self, ngvc: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        x = ngvc.copy()

        # Gaussian noise
        x = x + rng.normal(0, self.noise_sigma, size=x.shape)
        x = np.clip(x, 0.0, None)

        # Peak shift (simulate slight color variation)
        if rng.random() < self.peak_shift_prob:
            shift = rng.integers(-5, 6)
            if shift != 0:
                x = np.roll(x, shift)
                x[:shift] = 0.0 if shift > 0 else 0.0
                x[-shift:] = 0.0 if shift < 0 else 0.0

        # Smoothing (simulate blur-like histogram smearing)
        if rng.random() < self.smoothing_prob:
            from scipy.ndimage import uniform_filter1d

            x = uniform_filter1d(x, size=3)

        # Re-normalize
        x = np.clip(x, 0.0, None)
        total = x.sum()
        if total > 0:
            x = x / total
        return x.astype(np.float32)
