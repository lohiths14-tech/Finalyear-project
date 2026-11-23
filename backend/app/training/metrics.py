import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
import torch
import torch.nn.functional as F


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
        self.all_sev_preds = []
        self.all_sev_targets = []
        self.total_loss = 0.0
        self.batches = 0

    def update(
        self,
        logits: torch.Tensor,
        target_classes: torch.Tensor,
        pred_severity: torch.Tensor,
        target_severity: torch.Tensor,
        loss: float,
    ):

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        targs = target_classes.detach().cpu().numpy()

        self.all_preds.extend(preds)
        self.all_targets.extend(targs)
        self.all_probs.extend(probs)

        valid_mask = target_severity >= 0
        if valid_mask.any():
            s_preds = pred_severity[valid_mask].detach().cpu().numpy()
            s_targs = target_severity[valid_mask].detach().cpu().numpy()
            self.all_sev_preds.extend(s_preds)
            self.all_sev_targets.extend(s_targs)

        self.total_loss += loss
        self.batches += 1

    def compute(self) -> dict:
        acc = (
            accuracy_score(self.all_targets, self.all_preds)
            if self.all_targets
            else 0.0
        )
        f1 = (
            f1_score(self.all_targets, self.all_preds, average="macro")
            if self.all_targets
            else 0.0
        )

        cm = (
            confusion_matrix(self.all_targets, self.all_preds).tolist()
            if self.all_targets
            else []
        )

        # Brier Score (approximated for multi-class by one-vs-rest average)
        brier = 0.0
        if self.all_targets and len(self.all_probs) > 0:
            num_classes = len(self.all_probs[0])
            brier_sum = 0
            for i in range(num_classes):
                binary_targets = (np.array(self.all_targets) == i).astype(int)
                class_probs = np.array(self.all_probs)[:, i]
                brier_sum += brier_score_loss(binary_targets, class_probs)
            brier = brier_sum / num_classes

        # Expected Calibration Error (ECE) - Simple 10-bin approximation
        ece = (
            self._compute_ece(
                np.array(self.all_probs), np.array(self.all_targets)
            )
            if self.all_targets
            else 0.0
        )

        sev_mae = 0.0
        if self.all_sev_targets:
            sev_mae = mean_absolute_error(
                self.all_sev_targets, self.all_sev_preds
            )

        avg_loss = self.total_loss / self.batches if self.batches > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": f1,
            "severity_mae": sev_mae,
            "confusion_matrix": cm,
            "brier_score": brier,
            "ece": ece,
        }

    def _compute_ece(
        self, probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
    ) -> float:
        if len(probs) == 0:
            return 0.0

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == targets

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            in_bin = np.logical_and(
                confidences > bin_boundaries[i],
                confidences <= bin_boundaries[i + 1],
            )
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

        return ece
