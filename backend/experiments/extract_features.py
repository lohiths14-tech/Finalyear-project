import os
import sys
import torch
import logging
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.training.config import TrainingConfig
from app.training.datamodule import AgriPulseDataModule
from app.models.architectures import FoundationVisionEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_and_save(loader, encoder, device, out_path):
    features = []
    labels = []
    severities = []
    datasets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting to {out_path.name}"):
            imgs = batch["image"].to(device)
            feats = encoder(imgs).cpu()
            
            features.append(feats)
            labels.extend(batch["label"])
            severities.append(batch["severity"])
            datasets.extend(batch["dataset"])
            
    torch.save({
        "features": torch.cat(features, dim=0),
        "labels": labels,
        "severities": torch.cat(severities, dim=0),
        "datasets": datasets
    }, out_path)
    logger.info(f"Saved {len(labels)} samples to {out_path}")

def main():
    logger.info("Setting up Feature Extraction...")
    device = torch.device("cpu") # User has no CUDA
    
    config = TrainingConfig(batch_size=32)
    data_dirs = {
        "paddy_doctor": "backend/data/paddy_doctor",
        "plant_village": "backend/data/plant_village",
        "plantdoc": "backend/data/plantdoc",
        "uav_rice_field": "backend/data/uav_rice"
    }
    
    datamodule = AgriPulseDataModule(data_dirs=data_dirs, config=config)
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    encoder = FoundationVisionEncoder(feature_dim=64).to(device)
    encoder.eval()
    
    os.makedirs("data_cache", exist_ok=True)
    extract_and_save(train_loader, encoder, device, Path("data_cache/train_features.pt"))
    extract_and_save(val_loader, encoder, device, Path("data_cache/val_features.pt"))

if __name__ == "__main__":
    main()
