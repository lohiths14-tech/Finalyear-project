import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import MultiTaskLoss
from .metrics import MetricTracker

logger = logging.getLogger(__name__)


class Validator:
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
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        self.tracker.reset()

        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)

            # Map standard string labels to integer indices
            # (In a full implementation, pass the mapping in the constructor)
            # For simplicity, assuming batch["label_idx"] exists or we map it
            # here
            from ..training.dataset import DISEASE_CLASSES

            target_classes = torch.tensor(
                [DISEASE_CLASSES.index(lbl) for lbl in batch["label"]],
                device=self.device,
            )
            target_severity = batch["severity"].to(self.device)

            # Forward pass (assuming model returns logits, severity)
            # Depending on architecture, you may pass dummy spatial/temporal
            # tensors
            # This is a scaffolding for the unified forward pass
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

        return self.tracker.compute()
