import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for classification and severity regression.
    """

    def __init__(self, class_weights=None, severity_weight: float = 0.5):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.severity_loss = nn.MSELoss()
        self.severity_weight = severity_weight

    def forward(
        self,
        logits: torch.Tensor,
        pred_severity: torch.Tensor,
        target_classes: torch.Tensor,
        target_severity: torch.Tensor,
    ) -> dict:

        loss_cls = self.classification_loss(logits, target_classes)

        # Only compute severity loss for instances where severity is available
        # (>= 0)
        # Assuming unavailable severity is encoded as -1.0
        valid_mask = target_severity >= 0
        if valid_mask.any():
            loss_sev = self.severity_loss(
                pred_severity[valid_mask].squeeze(),
                target_severity[valid_mask].squeeze(),
            )
        else:
            loss_sev = torch.tensor(0.0, device=logits.device)

        total_loss = loss_cls + (self.severity_weight * loss_sev)

        return {
            "total": total_loss,
            "classification": loss_cls,
            "severity": loss_sev,
        }
