"""
AgriVision AI — Synthetic Training Script
==========================================
Generates domain-informed synthetic NGVC spectral data for 5 disease classes
and trains the full SST pipeline (NGVCNet + SpatialGraphNet + TemporalProgressionNet +
AgriVisionEnsemble) end-to-end.

Run from the `backend/` directory:
    python scripts/train_synthetic.py

Outputs trained weights to ../model_weights/ (one .pt file per sub-model).
These are then auto-loaded by InferenceService on startup.
"""

import sys
from pathlib import Path

# Allow imports from the backend package
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

from app.models.architectures import (
    NGVCNet, SpatialGraphNet, TemporalProgressionNet, AgriVisionEnsemble,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BINS = 128           # NGVC histogram bins
NUM_CLASSES = 5
SAMPLES_PER_CLASS = 800
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEVERITY_LOSS_WEIGHT = 0.3
VAL_SPLIT = 0.15
OUTPUT_DIR = Path(__file__).parent.parent.parent / "model_weights"

# Disease classes (index must match DISEASE_CLASSES in inference_service.py)
CLASS_PROFILES = {
    0: {  # Healthy
        "name": "Healthy",
        "peaks": [(160, 20)],          # (center, std) in grayscale 0-255
        "severity_range": (2.0, 15.0),
    },
    1: {  # Rice Blast — bimodal: dark lesions + bright background
        "name": "Rice Blast",
        "peaks": [(60, 15), (170, 25)],
        "severity_range": (60.0, 85.0),
    },
    2: {  # Brown Spot — brownish pigmentation: mid-low histogram
        "name": "Brown Spot",
        "peaks": [(100, 20), (140, 15)],
        "severity_range": (40.0, 65.0),
    },
    3: {  # Leaf Scald — bleached: concentrated at high values
        "name": "Leaf Scald",
        "peaks": [(200, 18), (140, 20)],
        "severity_range": (30.0, 55.0),
    },
    4: {  # Bacterial Blight — broad water-soaked distribution
        "name": "Bacterial Blight",
        "peaks": [(130, 45)],
        "severity_range": (70.0, 92.0),
    },
}


# ---------------------------------------------------------------------------
# Synthetic NGVC generator
# ---------------------------------------------------------------------------
def generate_ngvc(profile: dict, bins: int = BINS, noise_level: float = 0.015) -> np.ndarray:
    """
    Build a synthetic NGVC histogram from a Gaussian mixture model.
    Adds realistic noise and normalises to a probability distribution.
    """
    x = np.arange(bins, dtype=np.float32) * (255.0 / bins)
    hist = np.zeros(bins, dtype=np.float32)

    for center, std in profile["peaks"]:
        component = np.exp(-0.5 * ((x - center) / std) ** 2)
        hist += component

    # Add noise + small uniform floor to simulate real sensor noise
    hist += np.random.uniform(0.0, noise_level, bins)
    hist += np.random.uniform(0.0, noise_level / 4.0)   # uniform floor

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def make_dataset(samples_per_class: int = SAMPLES_PER_CLASS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_class, y_severity)."""
    X, y_class, y_severity = [], [], []
    rng = np.random.default_rng(42)

    for cls_id, profile in CLASS_PROFILES.items():
        lo, hi = profile["severity_range"]
        for _ in range(samples_per_class):
            # Vary noise per sample for diversity
            noise = rng.uniform(0.005, 0.03)
            ngvc = generate_ngvc(profile, noise_level=noise)
            severity = rng.uniform(lo, hi)
            X.append(ngvc)
            y_class.append(cls_id)
            y_severity.append(severity)

    return (
        np.array(X, dtype=np.float32),
        np.array(y_class, dtype=np.int64),
        np.array(y_severity, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class NGVCDataset(Dataset):
    def __init__(self, X: np.ndarray, y_class: np.ndarray, y_severity: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X)
        self.y_class = torch.tensor(y_class)
        self.y_severity = torch.tensor(y_severity)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Online augmentation: Gaussian noise + random scale + intensity shift
            x = x + torch.randn_like(x) * 0.02          # Gaussian noise
            x = x * (0.9 + torch.rand(1).item() * 0.2)   # Scale ±10%
            shift = (torch.rand(1).item() - 0.5) * 0.05   # Shift ±2.5%
            x = x + shift
            x = torch.clamp(x, min=0.0)                   # Non-negative
            total = x.sum()
            if total > 0:
                x = x / total                              # Re-normalise to PDF
        return x, self.y_class[idx], self.y_severity[idx]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def build_trivial_graph(batch_size: int, feat_dim: int, device: torch.device):
    """
    For synthetic training without real spatial data, use a fully-connected
    single-node graph (identity adjacency) per sample, treated as batch nodes.
    Each sample in the batch is treated as an isolated graph node.
    """
    # Self-loop only — each node aggregates only itself
    adj = torch.eye(batch_size, device=device)
    return adj


def train_one_epoch(
    spectral_net, spatial_net, temporal_net, meta_learner,
    loader, optimizer, ce_loss, mse_loss, device, severity_w: float,
) -> Tuple[float, float]:
    """Returns (avg_loss, avg_accuracy)."""
    for m in [spectral_net, spatial_net, temporal_net, meta_learner]:
        m.train()

    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_class, y_sev in loader:
        X_batch = X_batch.to(device)
        y_class = y_class.to(device)
        y_sev = y_sev.to(device)

        bs = X_batch.size(0)
        optimizer.zero_grad()

        # 1. Spectral embedding
        spec_feat = spectral_net(X_batch)                              # (bs, 64)

        # 2. Spatial — trivial single-node graph per sample in batch
        adj = build_trivial_graph(bs, 64, device)
        spat_feat = spatial_net(spec_feat, adj)                        # (bs, 64)

        # 3. Temporal — simulate 5-step sequence with slight noise
        noise_seq = [spat_feat + torch.randn_like(spat_feat) * 0.05 * (t + 1)
                     for t in range(5)]
        temp_input = torch.stack(noise_seq, dim=1)                     # (bs, 5, 64)
        temp_feat = temporal_net(temp_input)                           # (bs, 64)

        # 4. Ensemble
        logits, severity_pred = meta_learner(spec_feat, spat_feat, temp_feat)

        # 5. Loss
        loss_cls = ce_loss(logits, y_class)
        loss_sev = mse_loss(severity_pred.squeeze(), y_sev)
        loss = loss_cls + severity_w * loss_sev

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(spectral_net.parameters()) + list(spatial_net.parameters()) +
            list(temporal_net.parameters()) + list(meta_learner.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == y_class).sum().item()
        total += bs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    spectral_net, spatial_net, temporal_net, meta_learner,
    loader, ce_loss, mse_loss, device, severity_w: float,
) -> Tuple[float, float]:
    for m in [spectral_net, spatial_net, temporal_net, meta_learner]:
        m.eval()

    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_class, y_sev in loader:
        X_batch, y_class, y_sev = X_batch.to(device), y_class.to(device), y_sev.to(device)
        bs = X_batch.size(0)

        spec_feat = spectral_net(X_batch)
        adj = build_trivial_graph(bs, 64, device)
        spat_feat = spatial_net(spec_feat, adj)
        temp_input = spat_feat.unsqueeze(1).repeat(1, 5, 1)
        temp_feat = temporal_net(temp_input)
        logits, severity_pred = meta_learner(spec_feat, spat_feat, temp_feat)

        loss = ce_loss(logits, y_class) + severity_w * mse_loss(severity_pred.squeeze(), y_sev)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y_class).sum().item()
        total += bs

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AgriVision AI — Synthetic Training Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")
    print(f"  Classes    : {NUM_CLASSES}")
    print(f"  Samples    : {SAMPLES_PER_CLASS * NUM_CLASSES} ({SAMPLES_PER_CLASS}/class)")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Output dir : {OUTPUT_DIR}\n")

    # ── Build dataset ──────────────────────────────────────────────────────
    print("Generating synthetic NGVC dataset...")
    X, y_class, y_severity = make_dataset()

    val_size = int(len(X) * VAL_SPLIT)
    train_size = len(X) - val_size

    # Shuffle with fixed seed for reproducibility
    rng_idx = np.random.default_rng(42)
    indices = rng_idx.permutation(len(X))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_ds = NGVCDataset(X[train_idx], y_class[train_idx], y_severity[train_idx], augment=True)
    val_ds = NGVCDataset(X[val_idx], y_class[val_idx], y_severity[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {train_size} (augmented) | Val: {val_size}\n")

    # ── Build models ───────────────────────────────────────────────────────
    spectral_net = NGVCNet(input_dim=BINS, feature_dim=64).to(device)
    spatial_net = SpatialGraphNet(in_channels=64, out_channels=64).to(device)
    temporal_net = TemporalProgressionNet(input_dim=64).to(device)
    meta_learner = AgriVisionEnsemble(feature_dim=64, num_classes=NUM_CLASSES).to(device)

    # Xavier initialization for better convergence
    for model in [spectral_net, spatial_net, temporal_net, meta_learner]:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    all_params = (
        list(spectral_net.parameters()) +
        list(spatial_net.parameters()) +
        list(temporal_net.parameters()) +
        list(meta_learner.parameters())
    )

    optimizer = optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse_loss = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    patience = 15
    no_improve = 0

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            spectral_net, spatial_net, temporal_net, meta_learner,
            train_loader, optimizer, ce_loss, mse_loss, device, SEVERITY_LOSS_WEIGHT,
        )
        val_loss, val_acc = evaluate(
            spectral_net, spatial_net, temporal_net, meta_learner,
            val_loader, ce_loss, mse_loss, device, SEVERITY_LOSS_WEIGHT,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        marker = " ★" if val_acc > best_val_acc else ""
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.1%}  {val_loss:>8.4f}  {val_acc:>6.1%}  {current_lr:>8.2e}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            # Save best checkpoint weights
            _save_weights(spectral_net, spatial_net, temporal_net, meta_learner, suffix="_best")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Re-run eval on best weights to generate CM
    _load_weights(spectral_net, spatial_net, temporal_net, meta_learner, suffix="_best")
    _generate_cm_and_classes(val_loader, spectral_net, spatial_net, temporal_net, meta_learner, device, best_val_acc)

    # ── Save final weights ─────────────────────────────────────────────────
    _save_weights(spectral_net, spatial_net, temporal_net, meta_learner, suffix="")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best val accuracy : {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"  Weights saved to  : {OUTPUT_DIR}/")
    print("=" * 60)


def _save_weights(spectral_net, spatial_net, temporal_net, meta_learner, suffix=""):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save final weights (no suffix) and best checkpoint (suffix="_best")
    base = OUTPUT_DIR
    torch.save(spectral_net.state_dict(), base / f"spectral_net{suffix}.pt")
    torch.save(spatial_net.state_dict(), base / f"spatial_net{suffix}.pt")
    torch.save(temporal_net.state_dict(), base / f"temporal_net{suffix}.pt")
    torch.save(meta_learner.state_dict(), base / f"ensemble{suffix}.pt")

def _load_weights(spectral_net, spatial_net, temporal_net, meta_learner, suffix=""):
    base = OUTPUT_DIR
    spectral_net.load_state_dict(torch.load(base / f"spectral_net{suffix}.pt"))
    spatial_net.load_state_dict(torch.load(base / f"spatial_net{suffix}.pt"))
    temporal_net.load_state_dict(torch.load(base / f"temporal_net{suffix}.pt"))
    meta_learner.load_state_dict(torch.load(base / f"ensemble{suffix}.pt"))

def _generate_cm_and_classes(val_loader, spectral_net, spatial_net, temporal_net, meta_learner, device, val_acc):
    import json
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    for m in [spectral_net, spatial_net, temporal_net, meta_learner]:
        m.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_class, _ in val_loader:
            X_batch, y_class = X_batch.to(device), y_class.to(device)
            bs = X_batch.size(0)
            spec_feat = spectral_net(X_batch)
            adj = build_trivial_graph(bs, 64, device)
            spat_feat = spatial_net(spec_feat, adj)
            temp_input = spat_feat.unsqueeze(1).repeat(1, 5, 1)
            temp_feat = temporal_net(temp_input)
            logits, _ = meta_learner(spec_feat, spat_feat, temp_feat)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y_class.cpu().tolist())

    classes_list = [CLASS_PROFILES[i]["name"] for i in range(len(CLASS_PROFILES))]
    num_classes = len(classes_list)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(max(12, num_classes), max(10, num_classes - 2)))
    sns.heatmap(cm, annot=(num_classes <= 15), fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (SST Synthetic)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()

    # Generate dummy treatments for demo
    treatments = {c: "Monitor closely." for c in classes_list}
    classes_path = OUTPUT_DIR / "classes.json"
    with open(classes_path, "w") as f:
        json.dump({
            "classes": classes_list, 
            "treatments": treatments, 
            "trained_with_sst": True,
            "has_cm": True,
            "validation_accuracy": round(val_acc * 100, 2)
        }, f, indent=4)



if __name__ == "__main__":
    main()
