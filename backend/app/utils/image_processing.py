import cv2
import numpy as np


class AgriImageProcessor:
    @staticmethod
    def remove_soil_and_shadow(image: np.ndarray):
        """
        Removes soil and shadows using ExG (Excess Green) index and Otsu thresholding.
        """  # noqa: E501
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image provided")

        img_float = image.astype(np.float32) / 255.0
        r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

        # Excess Green Index (ExG) = 2g - r - b
        exg = 2 * g - r - b

        # Normalize ExG to 0-255
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Otsu thresholding to segment vegetation
        _, mask = cv2.threshold(
            exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological operations to clean noise
        kernel = np.ones((5, 5), np.uint8)  # FIX: was np.one (NameError)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(image, image, mask=mask)
        return result, mask

    @staticmethod
    def rgb_to_pseudo_hsi(image: np.ndarray):
        """
        Reconstructs pseudo-HSI (Hue, Saturation, Intensity) from RGB.
        Robust to lighting variations.
        """
        img_float = image.astype(np.float32) / 255.0
        r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

        intensity = np.mean(img_float, axis=2)

        min_val = np.min(img_float, axis=2)
        saturation = 1 - (3 / (r + g + b + 1e-6)) * min_val

        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
        hue = np.arccos(np.clip(num / den, -1.0, 1.0))

        hue[b > g] = 2 * np.pi - hue[b > g]
        hue = hue / (2 * np.pi)

        return np.stack([hue, saturation, intensity], axis=2)

    @staticmethod
    def extract_ngvc(image_region: np.ndarray, bins: int = 128):
        """
        Extracts Normalized Gray Value Curve (NGVC).
        Represents the intensity distribution — a key texture feature for disease detection.
        """  # noqa: E501
        if (
            image_region is None
            or image_region.size == 0
            or len(image_region.shape) < 2
            or image_region.shape[0] == 0
            or image_region.shape[1] == 0
        ):
            return np.zeros(bins, dtype=np.float32)

        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region

        pixels = gray[gray > 0].flatten()
        if len(pixels) == 0:
            return np.zeros(bins)

        hist, _ = np.histogram(
            pixels, bins=bins, range=(0, 255), density=False
        )

        # Normalize to sum to 1.0 (proper probability distribution)
        total = hist.sum()
        ngvc = hist / (total + 1e-6)
        return ngvc.astype(np.float32)
