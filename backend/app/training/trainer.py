import datetime
import json
import logging
import os
import subprocess
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import CheckpointManager
from .config import TrainingConfig
from .losses import MultiTaskLoss
from .metrics import MetricTracker
from .validator import Validator

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Only pass trainable parameters to the optimizer (respects DINOv2
        # freezing)
        trainable_params = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.criterion = MultiTaskLoss()
        self.tracker = MetricTracker()

        self.validator = Validator(self.model, self.criterion, self.device)
        self.checkpoint = CheckpointManager(self.config.checkpoint_dir)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # TensorBoard & Tracking
        self.tb_dir = (
            f"{self.config.tensorboard_dir}/{self.config.experiment_name}"
        )
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # Early Stopping
        self.patience_counter = 0
        self.best_epoch = 1

    def train(self, resume: bool = False):
        start_epoch = 1
        if resume:
            start_epoch = self.checkpoint.load(
                self.model, self.optimizer, self.scheduler, "latest_model.pth"
            )

        logger.info(
            f"Starting training from epoch {start_epoch} to {self.config.num_epochs}..."  # noqa: E501
        )
        start_time = time.time()

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.model.train()
            self.tracker.reset()

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.config.num_epochs}",
            )
            for batch in pbar:
                images = batch["image"].to(self.device)

                from ..training.dataset import DISEASE_CLASSES

                target_classes = torch.tensor(
                    [DISEASE_CLASSES.index(lbl) for lbl in batch["label"]],
                    device=self.device,
                )
                target_severity = batch["severity"].to(self.device)

                self.optimizer.zero_grad()

                # AMP logic
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    # Check if input is a 1D/2D feature vector instead of 4D
                    # image
                    is_cached = images.dim() == 2
                    logits, pred_severity = self.model(
                        images, is_cached=is_cached
                    )
                    loss_dict = self.criterion(
                        logits, pred_severity, target_classes, target_severity
                    )

                self.scaler.scale(loss_dict["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.tracker.update(
                    logits,
                    target_classes,
                    pred_severity,
                    target_severity,
                    loss_dict["total"].item(),
                )
                pbar.set_postfix(loss=loss_dict["total"].item())

            train_metrics = self.tracker.compute()
            val_metrics = self.validator.validate(self.val_loader)

            # Step the scheduler
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f} - Val Loss: {val_metrics['loss']:.4f} - Val Macro-F1: {val_metrics['macro_f1']:.4f}"  # noqa: E501
            )

            # Log to TensorBoard
            self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
            self.writer.add_scalar(
                "Metric/Macro-F1", val_metrics["macro_f1"], epoch
            )
            self.writer.add_scalar(
                "Metric/Accuracy", val_metrics["accuracy"], epoch
            )
            self.writer.add_scalar(
                "LearningRate", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Save latest checkpoint
            self.checkpoint.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                "latest_model.pth",
            )

            # Save if best Macro-F1 score
            if val_metrics["macro_f1"] > self.checkpoint.best_metric:
                self.checkpoint.save_if_best(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics["macro_f1"],
                )
                self.patience_counter = 0
                self.best_epoch = epoch
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        elapsed_time = time.time() - start_time
        logger.info(f"Training complete in {elapsed_time / 60:.2f} minutes.")
        self._save_metadata(elapsed_time)
        self.writer.close()

    def _save_metadata(self, elapsed_time: float):
        try:
            git_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"]
                )
                .decode("ascii")
                .strip()
            )
        except Exception:
            git_hash = "unknown"

        metadata = {
            "experiment": self.config.experiment_name,
            "git_commit": git_hash,
            "seed": self.config.seed,
            "epochs_run": self.best_epoch + self.patience_counter,
            "best_macro_f1": self.checkpoint.best_metric,
            "best_epoch": self.best_epoch,
            "training_time_seconds": elapsed_time,
            "date": datetime.datetime.now().isoformat(),
        }

        path = f"{self.config.export_dir}/{self.config.experiment_name}_metadata.json"  # noqa: E501
        import shutil

        os.makedirs(self.config.export_dir, exist_ok=True)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Experiment metadata saved to {path}")

        # Copy Dataset Audit
        dataset_audit_src = "data_cache/dataset_version.json"
        if os.path.exists(dataset_audit_src):
            dataset_audit_dest = f"{self.config.export_dir}/{self.config.experiment_name}_dataset_audit.json"  # noqa: E501
            shutil.copy(dataset_audit_src, dataset_audit_dest)
            logger.info(f"Dataset audit saved to {dataset_audit_dest}")
