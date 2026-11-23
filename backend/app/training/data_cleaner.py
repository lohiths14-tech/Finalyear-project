import hashlib
import logging
from typing import List, Optional, Set

from PIL import Image

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data deduplication, label harmonization, and corrupt image filtering
    for the AgriPulse unified dataset pipeline.
    """  # noqa: E501

    def __init__(
        self, target_classes: List[str], allow_pseudo_severity: bool = False
    ):
        self.target_classes = target_classes
        self.allow_pseudo_severity = allow_pseudo_severity
        self._seen_hashes: Set[str] = set()

        # Simple disease mappings from common datasets (PlantVillage,
        # PaddyDoctor, PlantDoc)
        # mapped to our DISEASE_CLASSES.
        self._label_map = {
            "healthy": "Healthy",
            "normal": "Healthy",
            "rice blast": "Rice Blast (Magnaporthe oryzae)",
            "blast": "Rice Blast (Magnaporthe oryzae)",
            "rice_blast": "Rice Blast (Magnaporthe oryzae)",
            "brown spot": "Brown Spot (Bipolaris oryzae)",
            "brown_spot": "Brown Spot (Bipolaris oryzae)",
            "rice brown spot": "Brown Spot (Bipolaris oryzae)",
            "leaf scald": "Leaf Scald (Gerlachia oryzae)",
            "rice leaf scald": "Leaf Scald (Gerlachia oryzae)",
            "bacterial blight": "Bacterial Blight (Xanthomonas oryzicola)",
            "bacterial_leaf_blight": "Bacterial Blight (Xanthomonas oryzicola)",  # noqa: E501
            "hispa": "Unknown",
            "tungro": "Unknown",
            "dead heart": "Unknown",
        }

    def harmonize_label(self, raw_label: str) -> str:
        """
        Maps a dataset-specific label to the standardized disease classes.
        Returns "Unknown" if it cannot be mapped or should be dropped.
        """
        lower_label = raw_label.lower().strip()

        if lower_label in self._label_map:
            mapped = self._label_map[lower_label]
            if mapped in self.target_classes:
                return mapped

        # Fuzzy match for existing target classes
        for target in self.target_classes:
            if lower_label in target.lower():
                return target

        return "Unknown"

    def is_corrupt(self, image_path: str) -> bool:
        """
        Attempts to open and verify an image. Returns True if corrupt.
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # verify that it is, in fact, an image

            # verify() doesn't catch all decoding errors, so we attempt to load
            with Image.open(image_path) as img:
                img.load()
            return False
        except Exception as e:
            logger.debug(f"Corrupt image detected: {image_path} ({e})")
            return True

    def get_image_hash(self, image_path: str) -> str:
        """
        Computes MD5 hash of image contents for exact deduplication.
        """
        hasher = hashlib.md5()
        try:
            with open(image_path, "rb") as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception:
            return ""

    def is_duplicate(self, image_path: str) -> bool:
        """
        Checks if the exact image file has been seen before in this run.
        """
        img_hash = self.get_image_hash(image_path)
        if not img_hash:
            return True  # Can't hash it, treat as bad/duplicate

        if img_hash in self._seen_hashes:
            return True

        self._seen_hashes.add(img_hash)
        return False

    def estimate_severity(self, raw_label: str) -> Optional[float]:
        """
        Heuristic to generate pseudo-severity based on class names from public datasets
        that don't provide continuous severity values. Returns None if disabled.
        """  # noqa: E501
        if not self.allow_pseudo_severity:
            return None

        lower = raw_label.lower()
        if "healthy" in lower or "normal" in lower:
            return 0.0

        # In absence of real severity, we provide a generic active infection
        # severity
        if "early" in lower or "mild" in lower:
            return 25.0
        if "late" in lower or "severe" in lower:
            return 80.0

        return 50.0  # Default mid-severity for generic disease labels

    def reset_duplicates(self):
        """Clears the seen hashes for a new processing pass."""
        self._seen_hashes.clear()
