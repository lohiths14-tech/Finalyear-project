from datetime import datetime
import glob
import json
import logging
import os
from typing import Any, Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .data_cleaner import DataCleaner

logger = logging.getLogger(__name__)

# Standard Disease Classes defined globally for AgriPulse
DISEASE_CLASSES = [
    "Healthy",
    "Rice Blast (Magnaporthe oryzae)",
    "Brown Spot (Bipolaris oryzae)",
    "Leaf Scald (Gerlachia oryzae)",
    "Bacterial Blight (Xanthomonas oryzicola)",
]


class AgriPulseDataset(Dataset):
    """
    Unified PyTorch Dataset for AgriPulse 2.0.
    Combines Paddy Doctor, PlantVillage, PlantDoc, and real Field datasets
    into a standardized format. Applies automatic deduplication and cleaning.
    """

    def __init__(
        self, data_dirs: Dict[str, str], transform=None, is_train: bool = True
    ):
        """
        data_dirs: Dictionary mapping dataset source to root directory path.
                   e.g., {"paddy_doctor": "/path/to/paddy", "plant_village": "/path/to/pv", ...}
        """  # noqa: E501
        self.data_dirs = data_dirs
        self.is_train = is_train

        # Default DINOv2 / ImageNet transform if none provided
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.cleaner = DataCleaner(
            target_classes=DISEASE_CLASSES, allow_pseudo_severity=False
        )
        self.samples: List[Dict[str, Any]] = []

        self.report = {
            "sources": {},
            "duplicates_removed": 0,
            "corrupt_images": 0,
            "unknown_labels": 0,
            "final_images": 0,
        }

        self._build_dataset()

    def _build_dataset(self):
        """Scans all provided directories and parses them based on source name."""  # noqa: E501
        logger.info("Building unified AgriPulse dataset...")

        for source, root_dir in self.data_dirs.items():
            if not os.path.exists(root_dir):
                logger.warning(
                    f"Directory for {source} not found at {root_dir}. Skipping."  # noqa: E501
                )
                continue

            context_map = {
                "paddy_doctor": "leaf",
                "plant_village": "leaf",
                "plantdoc": "leaf",
                "uav_rice_field": "field"
            }
            context = context_map.get(source)
            if not context:
                logger.warning(
                    f"Unknown dataset source: {source}. Attempting generic folder parsing."  # noqa: E501
                )
                context = "unknown"

            raw_samples = self._parse_standard_folder_structure(
                root_dir, source, context)

            # Clean and add to unified list
            added = 0
            for sample in raw_samples:
                # Harmonize label
                harmonized_label = self.cleaner.harmonize_label(
                    sample["raw_label"]
                )
                if harmonized_label == "Unknown":
                    self.report["unknown_labels"] += 1
                    continue  # Drop unmappable classes

                # Deduplicate and check corruption
                if self.cleaner.is_duplicate(sample["image_path"]):
                    self.report["duplicates_removed"] += 1
                    continue
                if self.cleaner.is_corrupt(sample["image_path"]):
                    self.report["corrupt_images"] += 1
                    continue

                # Standardize
                severity = self.cleaner.estimate_severity(sample["raw_label"])

                self.samples.append(
                    {
                        "image_path": sample["image_path"],
                        "label": harmonized_label,
                        "severity": severity,
                        "dataset": source,
                        "source": sample["context"],
                        "metadata": {
                            "original_label": sample["raw_label"],
                            "pseudo_severity": True,
                        },
                    }
                )
                added += 1

            self.report["sources"][source] = added
            logger.info(f"Source {source}: Added {added} clean samples.")

        self.report["final_images"] = len(self.samples)
        self._print_summary()
        self._save_metadata()

    def _print_summary(self):
        print("\n=== Dataset Summary ===")
        for source, count in self.report["sources"].items():
            print(f"{source.capitalize()}\nImages : {count}\n")

        print(f"Duplicates Removed : {self.report['duplicates_removed']}")
        print(f"Corrupted Images   : {self.report['corrupt_images']}")
        print(f"Unknown Labels     : {self.report['unknown_labels']}")
        print(f"Final Images       : {self.report['final_images']}")
        print("========================\n")

    def _save_metadata(self):
        meta = {
            "dataset_names": list(self.report["sources"].keys()),
            "image_counts": self.report["sources"],
            "duplicates_removed": self.report["duplicates_removed"],
            "cleaning_date": datetime.now().isoformat(),
            "class_mapping": self.cleaner._label_map,
            "preprocessing_version": "1.0",
            "pseudo_severity_enabled": self.cleaner.allow_pseudo_severity,
        }
        os.makedirs("data_cache", exist_ok=True)
        path = os.path.join("data_cache", "dataset_version.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Dataset metadata saved to {path}")

    def _parse_standard_folder_structure(
        self, root_dir: str, dataset_name: str, context: str
    ) -> List[Dict]:
        """
        Parses a dataset where images are stored in folders named after their class.
        Returns a list of uncleaned dicts.
        """  # noqa: E501
        raw_samples = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
                for img_path in glob.glob(os.path.join(class_dir, ext)):
                    raw_samples.append(
                        {
                            "image_path": img_path,
                            "raw_label": class_name,
                            "context": context,
                        }
                    )
        return raw_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["image_path"]).convert("RGB")

        # Apply transform
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return {
            "image": img_tensor,
            "label": sample["label"],
            "severity": torch.tensor(sample["severity"], dtype=torch.float32)
            if sample["severity"] is not None
            else -1.0,
            "dataset": sample["dataset"],
            "source": sample["source"],
            "metadata": sample["metadata"],
        }
