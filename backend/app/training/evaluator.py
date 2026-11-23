import logging
import os
import time
from typing import Any, Dict

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from .losses import MultiTaskLoss
from .metrics import MetricTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Handles final evaluation on the test set. Similar to Validator but
    may include extended metrics like Calibration (ECE) and OOD detection logic.
    """  # noqa: E501

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: MultiTaskLoss,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.tracker = MetricTracker()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        self.tracker.reset()

        # Reset VRAM stats for accurate profiling
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        num_samples = 0

        for batch in tqdm(
            dataloader, desc="Evaluating (Test Set)", leave=False
        ):
            images = batch["image"].to(self.device)
            num_samples += images.size(0)

            from ..training.dataset import DISEASE_CLASSES

            target_classes = torch.tensor(
                [DISEASE_CLASSES.index(lbl) for lbl in batch["label"]],
                device=self.device,
            )
            target_severity = batch["severity"].to(self.device)

            is_cached = images.dim() == 2
            logits, pred_severity = self.model(images, is_cached=is_cached)
            loss_dict = self.criterion(
                logits, pred_severity, target_classes, target_severity
            )

            self.tracker.update(
                logits,
                target_classes,
                pred_severity,
                target_severity,
                loss_dict["total"].item(),
            )

        end_time = time.time()

        metrics = self.tracker.compute()

        # Profiling
        metrics["fps"] = (
            num_samples / (end_time - start_time)
            if end_time > start_time
            else 0.0
        )
        if torch.cuda.is_available():
            metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (
                1024**2
            )

        # OOD Evaluation
        ood_results = self.evaluate_ood()
        if ood_results:
            metrics["ood_detection_rate"] = ood_results["detection_rate"]

        logger.info("Test Set Evaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k}: {v:.4f}")
        return metrics

    @torch.no_grad()
    def evaluate_ood(
        self, ood_dir: str = "data/ood", confidence_threshold: float = 0.5
    ) -> Dict[str, float]:
        if not os.path.exists(ood_dir):
            logger.warning(
                f"OOD directory '{ood_dir}' not found. Skipping OOD evaluation."  # noqa: E501
            )
            return {}

        logger.info("Running OOD Evaluation...")
        self.model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        ood_files = []
        for ext in (".png", ".jpg", ".jpeg"):
            ood_files.extend(
                [
                    os.path.join(ood_dir, f)
                    for f in os.listdir(ood_dir)
                    if f.lower().endswith(ext)
                ]
            )

        if not ood_files:
            logger.warning(
                f"No images found in {ood_dir}. Skipping OOD evaluation."
            )
            return {}

        detected_as_ood = 0
        total = len(ood_files)

        for img_path in tqdm(ood_files, desc="OOD Inference", leave=False):
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(self.device)

            logits, _ = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            max_conf = probs.max().item()

            # If max confidence is lower than threshold, the model is "unsure",
            # successfully detecting OOD
            if max_conf < confidence_threshold:
                detected_as_ood += 1

        detection_rate = detected_as_ood / total
        logger.info(
            f"OOD Detection Rate (threshold < {confidence_threshold}): {detection_rate * 100:.1f}%"  # noqa: E501
        )

        return {"detection_rate": detection_rate}
