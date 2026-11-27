import logging

import numpy as np
from torch.utils.data import DataLoader, Subset

from .config import TrainingConfig
from .dataset import AgriPulseDataset

logger = logging.getLogger(__name__)


class AgriPulseDataModule:
    """
    Handles dataset initialization, stratified Train/Val/Test splitting,
    and DataLoader creation.
    """

    def __init__(self, data_dirs: dict[str, str], config: TrainingConfig):
        self.data_dirs = data_dirs
        self.config = config

        # This initializes the unified dataset, runs the DataCleaner, and
        # prints reports
        self.full_dataset = AgriPulseDataset(
            data_dirs=self.data_dirs, is_train=True
        )

        # Override cleaner config based on TrainingConfig
        self.full_dataset.cleaner.allow_pseudo_severity = (
            self.config.allow_pseudo_severity
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """
        Performs stratified splitting based on labels to ensure fair class distribution.
        """  # noqa: E501
        if len(self.full_dataset) == 0:
            logger.warning("Dataset is empty. Cannot perform split.")
            return

        # Extract all labels to compute stratification
        labels = [sample["label"] for sample in self.full_dataset.samples]

        # Compute exact indices for a stratified split (80/10/10)
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(self.full_dataset))

        # First split: Train vs Val+Test
        train_idx, val_test_idx, _, y_val_test = train_test_split(
            indices,
            labels,
            test_size=(self.config.val_split + self.config.test_split),
            stratify=labels,
            random_state=42,
        )

        # Second split: Val vs Test
        val_ratio = self.config.val_split / (
            self.config.val_split + self.config.test_split
        )
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=(1.0 - val_ratio),
            stratify=y_val_test,
            random_state=42,
        )

        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)
        self.test_dataset = Subset(self.full_dataset, test_idx)

        logger.info(
            f"Split complete: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"  # noqa: E501
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
