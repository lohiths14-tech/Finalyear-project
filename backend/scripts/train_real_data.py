"""
AgriVision AI — Real Dataset Training Script
==============================================
Trains NGVCNet + AgriVisionEnsemble on the PlantVillage dataset
(or any folder of crop disease images organised as Crop/Split/Disease/).

Expects this folder structure relative to the project root:
    <project_root>/
        Apple/Train/<DiseaseA>/  *.jpg
        Apple/Val/<DiseaseA>/    *.jpg
        Tomato/Train/<DiseaseB>/ *.jpg
        ...

Run from the `backend/` directory:
    python scripts/train_real_data.py [--data-root PATH] [--epochs N]

Outputs trained weights to ../model_weights/
    spectral_net.pt
    ensemble.pt          ← must match InferenceService loader key
    classes.json         ← class list + treatments + validation_accuracy

NOTE: Spatial and temporal features are set to zeros during this training phase
because single images cannot provide multi-frame spatial graphs. This is an
acknowledged limitation. The SST ensemble is therefore trained in spectral-only
mode here. For full SST training, use train_synthetic.py which exercises all
three sub-networks with synthetically generated sequences.
"""

import sys
import json
import argparse
from pathlib import Path

# Allow imports from the backend package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.models.architectures import NGVCNet, AgriVisionEnsemble
from app.utils.image_processing import AgriImageProcessor
from app.core.config import settings

# ---------------------------------------------------------------------------
# Crop subdirectories to scan
# ---------------------------------------------------------------------------
CROPS = [
    "Apple", "Bell Pepper", "Cherry", "Corn (Maize)", "Grape",
    "Peach", "Potato", "Strawberry", "Tomato",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AgriRealDataset(Dataset):
    def __init__(self, data_root: Path, split: str = "Train", max_per_class: int = 150, augment: bool = False):
        self.samples: list = []
        self.classes: list = []
        self.processor = AgriImageProcessor()
        self.augment = augment

        for crop in CROPS:
            crop_path = data_root / crop / split
            if not crop_path.exists():
                continue

            for disease_dir in sorted(crop_path.iterdir()):
                if not disease_dir.is_dir():
                    continue

                disease_name = f"{crop} - {disease_dir.name}"
                if disease_name not in self.classes:
                    self.classes.append(disease_name)

                class_idx = self.classes.index(disease_name)
                images = list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.JPG"))
                for img_path in images[:max_per_class]:
                    self.samples.append((str(img_path), class_idx))

        self.classes.sort()
        aug_tag = " (augmented)" if augment else ""
        print(f"[{split}] {len(self.classes)} classes, {len(self.samples)} samples{aug_tag}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        import cv2
        import numpy as np

        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(128, dtype=torch.float32), class_idx

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image-level augmentation (applied before NGVC extraction)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
            # Random brightness shift ±15%
            factor = 0.85 + np.random.rand() * 0.30
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        try:
            processed_img, _ = self.processor.remove_soil_and_shadow(img)
            ngvc = self.processor.extract_ngvc(processed_img, bins=128)
        except Exception:
            ngvc = self.processor.extract_ngvc(img, bins=128)

        x = torch.FloatTensor(ngvc)

        # NGVC-level augmentation (post-extraction)
        if self.augment:
            x = x + torch.randn_like(x) * 0.015       # Gaussian noise
            x = torch.clamp(x, min=0.0)
            total = x.sum()
            if total > 0:
                x = x / total                           # Re-normalise

        return x, class_idx


# ---------------------------------------------------------------------------
# Treatment map generator
# ---------------------------------------------------------------------------
def create_treatment_map(classes: list) -> dict:
    """Generate contextual treatment recommendations for any disease class name."""
    mapping = {}
    for c in classes:
        if "Healthy" in c:
            mapping[c] = "No treatment required. Maintain optimal watering and sunlight."
        elif "Scab" in c:
            mapping[c] = "Apply fungicide sprays containing captan or myclobutanil."
        elif "Rot" in c:
            mapping[c] = "Prune infected areas; improve air circulation; apply copper-based fungicides."
        elif "Rust" in c:
            mapping[c] = "Remove alternate hosts if applicable; apply protective fungicides."
        elif "Blight" in c:
            mapping[c] = "Use disease-free seed; apply preventative fungicides (e.g. chlorothalonil)."
        elif "Mold" in c or "Mildew" in c:
            mapping[c] = "Reduce humidity; apply sulfur-based or systemic fungicide."
        elif "Spot" in c:
            mapping[c] = "Remove infected leaves; apply Mancozeb 75% WP @ 2 g/L."
        elif "Virus" in c or "Mosaic" in c:
            mapping[c] = "No chemical cure; remove infected plants; control insect vectors."
        else:
            mapping[c] = "Monitor closely; apply broad-spectrum fungicide or bactericide if symptoms worsen."
    return mapping


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(data_root: Path, epochs: int = 15, max_per_class: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = AgriRealDataset(data_root, split="Train", max_per_class=max_per_class)
    if len(train_dataset) == 0:
        print("❌ No training data found. Check --data-root and folder structure.")
        return

    val_dataset = AgriRealDataset(data_root, split="Val", max_per_class=50)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None

    num_classes = len(train_dataset.classes)
    output_dir = Path(settings.MODEL_BASE_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classes.json immediately so InferenceService can load it on restart
    classes_path = output_dir / "classes.json"
    treatments = create_treatment_map(train_dataset.classes)
    with open(classes_path, "w") as f:
        json.dump({"classes": train_dataset.classes, "treatments": treatments, "trained_with_sst": False, "has_cm": False}, f, indent=4)
    print(f"Saved {num_classes} classes → {classes_path}")

    # Remove stale spatial/temporal weights so InferenceService correctly passes zeros to ensemble
    for f in ["spatial_net.pt", "temporal_net.pt"]:
        p = output_dir / f
        if p.exists(): p.unlink()

    # Models
    spectral_net = NGVCNet(input_dim=128, feature_dim=64).to(device)
    # NOTE: AgriVisionEnsemble is trained in spectral-only mode (spatial/temporal are zeros)
    # because single images cannot provide multi-frame spatial context.
    meta_learner = AgriVisionEnsemble(feature_dim=64, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(
        list(spectral_net.parameters()) + list(meta_learner.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        spectral_net.train()
        meta_learner.train()
        running_loss, correct, total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            spectral_feats = spectral_net(features)
            bs = features.size(0)
            # Zeros for spatial/temporal — acknowledged limitation, see module docstring
            zero_spatial = torch.zeros(bs, 64, device=device)
            zero_temporal = torch.zeros(bs, 64, device=device)
            logits, _ = meta_learner(spectral_feats, zero_spatial, zero_temporal)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(spectral_net.parameters()) + list(meta_learner.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            total += bs

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100

        val_acc_str = "N/A"
        val_acc = None

        if val_loader:
            spectral_net.eval()
            meta_learner.eval()
            v_correct, v_total = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for vf, vl in val_loader:
                    vf, vl = vf.to(device), vl.to(device)
                    vs = spectral_net(vf)
                    vb = vf.size(0)
                    v_logits, _ = meta_learner(vs, torch.zeros(vb, 64, device=device), torch.zeros(vb, 64, device=device))
                    preds = v_logits.argmax(1)
                    v_correct += (preds == vl).sum().item()
                    v_total += vb
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(vl.cpu().tolist())

            val_acc = v_correct / v_total * 100
            val_acc_str = f"{val_acc:.2f}%"
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"Epoch {epoch:>3}/{epochs} | Loss: {epoch_loss:.4f} | Train: {epoch_acc:.2f}% | Val: {val_acc_str}")

    # Save final weights
    print("\nSaving weights…")
    torch.save(spectral_net.state_dict(), output_dir / "spectral_net.pt")
    torch.save(meta_learner.state_dict(), output_dir / "ensemble.pt")   # ← must be ensemble.pt for InferenceService
    print(f"✅ Weights saved → {output_dir}/")

    # Final confusion matrix + accuracy update
    if val_loader and val_acc is not None:
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Re-run one final pass for CM
            spectral_net.eval()
            meta_learner.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for vf, vl in val_loader:
                    vf = vf.to(device)
                    vs = spectral_net(vf)
                    vb = vf.size(0)
                    v_logits, _ = meta_learner(vs, torch.zeros(vb, 64, device=device), torch.zeros(vb, 64, device=device))
                    all_preds.extend(v_logits.argmax(1).cpu().tolist())
                    all_labels.extend(vl.tolist())

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(max(12, num_classes), max(10, num_classes - 2)))
            sns.heatmap(cm, annot=(num_classes <= 15), fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix ({num_classes} Classes)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            cm_path = output_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=120)
            plt.close()
            print(f"Confusion matrix saved → {cm_path}")

            with open(classes_path, "r") as f:
                data = json.load(f)
            data["validation_accuracy"] = round(val_acc, 2)
            data["has_cm"] = True
            with open(classes_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Updated classes.json with val accuracy: {val_acc:.2f}%")
        except ImportError:
            print("⚠️  sklearn/seaborn not available — skipping confusion matrix.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriVision real-data training")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Path to project root containing Apple/, Tomato/, etc. (default: auto-detect)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--max-per-class", type=int, default=200, help="Max images per class (default: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AgriVision AI — Real Dataset Training")
    print(f"  Data root : {args.data_root}")
    print(f"  Epochs    : {args.epochs}")
    print("=" * 60)

    train(args.data_root, epochs=args.epochs, max_per_class=args.max_per_class)
