from app.training.trainer import Trainer  # noqa: E402
from app.training.model_wrapper import AgriPulseTrainingWrapper  # noqa: E402
from app.training.datamodule import AgriPulseDataModule  # noqa: E402
from app.training.config import TrainingConfig  # noqa: E402
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

# Ensure backend root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(
        description="AgriPulse 2.0 Training Pipeline"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint if available",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "graph_only", "temporal_only", "full"],
        help="Training mode: 'baseline', 'graph_only', 'temporal_only', or 'full'",  # noqa: E501
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached DINOv2 features for fast training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config seed for multi-seed experiments",
    )
    args = parser.parse_args()

    logger.info("Initializing AgriPulse 2.0 Training Pipeline...")
    logger.info(f"Training Mode Selected: {args.mode.upper()}")

    # 1. Load Config
    config = TrainingConfig(
        batch_size=16,
        num_epochs=50,
        allow_pseudo_severity=False,  # Forced false to prevent training on fabricated data  # noqa: E501
    )

    # Allow CLI to override seed for multi-seed experiments
    if args.seed is not None:
        config.seed = args.seed

    set_seeds(config.seed)
    logger.info(f"Seed: {config.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Setup DataModule
    data_dirs = {
        "paddy_doctor": "backend/data/paddy_doctor",
        "plant_village": "backend/data/plant_village",
        "plantdoc": "backend/data/plantdoc",
        "uav_rice_field": "backend/data/uav_rice",
    }

    if args.use_cache:
        logger.info("Loading Cached Feature Datasets...")
        from torch.utils.data import DataLoader

        from app.training.cached_dataset import CachedFeatureDataset

        train_dataset = CachedFeatureDataset("data_cache/train_features.pt")
        val_dataset = CachedFeatureDataset("data_cache/val_features.pt")
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )
    else:
        logger.info("Setting up raw datasets...")
        datamodule = AgriPulseDataModule(data_dirs=data_dirs, config=config)
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

    # 3. Initialize Model Structure
    logger.info("Instantiating Model...")
    from app.training.dataset import DISEASE_CLASSES

    model = AgriPulseTrainingWrapper(
        feature_dim=64, num_classes=len(DISEASE_CLASSES), mode=args.mode
    )
    model = model.to(device)

    # 4. FREEZE DINOv2 BACKBONE (Crucial instruction)
    logger.info("Freezing DINOv2 backbone to train transformer heads...")
    for param in model.vision_encoder.encoder.parameters():
        param.requires_grad = False

    # Verify freezing
    frozen_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # 6. Start Training
    try:
        trainer.train(resume=args.resume)
    except KeyboardInterrupt:
        logger.info("Training manually interrupted.")
    except Exception as e:
        logger.error(f"Training failed: {e}")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
