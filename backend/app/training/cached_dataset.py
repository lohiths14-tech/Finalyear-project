import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path: str):
        logger.info(f"Loading cached features from {cache_path}...")
        data = torch.load(cache_path)
        self.features = data["features"]
        self.labels = data["labels"]
        self.severities = data["severities"]
        self.datasets = data["datasets"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "image": self.features[
                idx
            ],  # It's actually the precomputed 64-dim feature vector
            "label": self.labels[idx],
            "severity": self.severities[idx],
            "dataset": self.datasets[idx],
        }
