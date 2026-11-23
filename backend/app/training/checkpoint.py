import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = 0.0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        filename: str,
    ):
        path = os.path.join(self.save_dir, filename)
        state_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler
            else None,
            "epoch": epoch,
            "best_metric": self.best_metric,
        }
        torch.save(state_dict, path)
        logger.info(f"Checkpoint saved to {path}")

    def save_if_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        current_metric: float,
        filename: str = "best_model.pth",
    ):
        """Saves the model if the current metric (e.g., F1 Score) is higher than the best recorded."""  # noqa: E501
        if current_metric > self.best_metric:
            logger.info(
                f"New best metric! {self.best_metric:.4f} -> {current_metric:.4f}"  # noqa: E501
            )
            self.best_metric = current_metric
            self.save(model, optimizer, scheduler, epoch, filename)

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        filename: str = "latest_model.pth",
    ) -> int:
        """Loads checkpoint and returns the epoch to resume from."""
        path = os.path.join(self.save_dir, filename)
        start_epoch = 1
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)

            # Handle both old style (just model state) and new style (full
            # checkpoint dict)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                if optimizer and "optimizer_state_dict" in state:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                if scheduler and state.get("scheduler_state_dict"):
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                start_epoch = state.get("epoch", 0) + 1
                self.best_metric = state.get("best_metric", 0.0)
            else:
                model.load_state_dict(state)

            logger.info(
                f"Loaded checkpoint from {path}, resuming from epoch {start_epoch}"  # noqa: E501
            )
        else:
            logger.warning(
                f"Checkpoint not found at {path}. Starting from scratch."
            )

        return start_epoch
