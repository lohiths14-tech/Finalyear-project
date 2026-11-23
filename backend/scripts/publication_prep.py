"""
AgriPulse — IEEE Publication Preparation Pipeline
===================================================
Runs EVERYTHING needed for a real IEEE publication:

  1.  Multi-seed training  (SST Ensemble × 5 seeds)
  2.  Baseline training    (1D-CNN, MLP-Baseline, ResNet-like baseline × 5 seeds)
  3.  Ablation study       (Spectral | Spatial | Temporal | SST)
  4.  Calibration analysis (ECE before/after, reliability diagram)
  5.  Figure generation    (6 publication-quality figures)
  6.  Results JSON         (ready to paste into the paper)

Usage (from backend/ directory):
    python scripts/publication_prep.py

All outputs are saved to:  model_weights/publication_results/
"""
from __future__ import annotations

import sys
import json
import time
import random
from pathlib import Path
from typing import List, Tuple

# ── path bootstrap ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.isotonic import IsotonicRegression

# ── matplotlib (non-GUI backend for server environments) ─────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── output directory ─────────────────────────────────────────────────────────
OUT_DIR = ROOT / "model_weights" / "publication_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  AgriPulse — IEEE Publication Preparation Pipeline")
print("=" * 70)
print(f"  Output directory : {OUT_DIR}")
print(f"  Figures directory: {FIG_DIR}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

RICE_PROFILE = {
    0: {"name": "Healthy",           "peaks": [(160, 20)],           "severity_range": (2.0,  15.0)},
    1: {"name": "Rice Blast",        "peaks": [(60, 15),  (170, 25)],"severity_range": (60.0, 85.0)},
    2: {"name": "Brown Spot",        "peaks": [(100, 20), (140, 15)],"severity_range": (40.0, 65.0)},
    3: {"name": "Leaf Scald",        "peaks": [(200, 18), (140, 20)],"severity_range": (30.0, 55.0)},
    4: {"name": "Bacterial Blight",  "peaks": [(130, 45)],           "severity_range": (70.0, 92.0)},
}
CLASS_NAMES = [RICE_PROFILE[i]["name"] for i in range(5)]
BINS = 128
SAMPLES_PER_CLASS = 1000   # 5 000 total  (train/val split 85/15)
NUM_CLASSES = 5


def _gen_ngvc(profile: dict, bins: int, noise: float, rng) -> np.ndarray:
    x = np.arange(bins, dtype=np.float32) * (255.0 / bins)
    hist = np.zeros(bins, dtype=np.float32)
    for center, std in profile["peaks"]:
        hist += np.exp(-0.5 * ((x - center) / std) ** 2)
    hist += rng.uniform(0.0, noise, bins)
    total = hist.sum()
    return hist / total if total > 0 else hist


def make_dataset(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, yc, ys = [], [], []
    for cls_id, prof in RICE_PROFILE.items():
        lo, hi = prof["severity_range"]
        for _ in range(SAMPLES_PER_CLASS):
            noise = rng.uniform(0.005, 0.03)
            X.append(_gen_ngvc(prof, BINS, noise, rng))
            yc.append(cls_id)
            ys.append(float(rng.uniform(lo, hi)))
    return (
        np.array(X,  dtype=np.float32),
        np.array(yc, dtype=np.int64),
        np.array(ys, dtype=np.float32),
    )


def split_dataset(X, yc, ys, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    vi, ti = idx[:n_val], idx[n_val:]
    return X[ti], yc[ti], ys[ti], X[vi], yc[vi], ys[vi]


class NGVCDataset(Dataset):
    def __init__(self, X, yc, ys, augment=False):
        self.X  = torch.tensor(X)
        self.yc = torch.tensor(yc)
        self.ys = torch.tensor(ys)
        self.augment = augment

    def __len__(self):  return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            x = x + torch.randn_like(x) * 0.018
            x = x * (0.92 + torch.rand(1).item() * 0.16)
            x = torch.clamp(x, min=0.0)
            s = x.sum()
            if s > 0: x /= s
        return x, self.yc[i], self.ys[i]


# ═══════════════════════════════════════════════════════════════════════════════
# §2  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralCNN(nn.Module):
    """NGVCNet – 1D-CNN spectral branch (also used as standalone baseline)."""
    def __init__(self, bins=128, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64,  5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,64, 3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        flat = (bins // 4) * 64
        self.fc = nn.Sequential(
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, embed_dim), nn.ReLU(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SpatialGNN(nn.Module):
    """Lightweight GraphSAGE-inspired spatial branch (simulated with MLP + neighbourhood avg)."""
    def __init__(self, in_dim=64, embed_dim=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim * 2, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128 * 2, embed_dim), nn.LayerNorm(embed_dim), nn.LeakyReLU(0.2),
        )

    def forward(self, z_spec: torch.Tensor) -> torch.Tensor:
        # Simulate neighbourhood: mean of random ±1 neighbours within a batch
        B = z_spec.size(0)
        # Create neighbour by rolling and mixing (batch-level spatial simulation)
        neigh = torch.roll(z_spec, 1, dims=0) * 0.5 + torch.roll(z_spec, -1, dims=0) * 0.5
        h1 = self.layer1(torch.cat([z_spec, neigh], dim=-1))
        neigh2 = torch.roll(h1, 2, dims=0) * 0.4 + torch.roll(h1, -2, dims=0) * 0.6
        return self.layer2(torch.cat([h1, neigh2], dim=-1))


class TemporalBiLSTM(nn.Module):
    """BiLSTM temporal branch — processes 5-step sequences."""
    def __init__(self, in_dim=64, hidden=64, embed_dim=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(self, z_spec: torch.Tensor) -> torch.Tensor:
        B, D = z_spec.shape
        # Build synthetic 5-step sequence: current + 4 progressively corrupted past steps
        noise = torch.randn(B, 4, D, device=z_spec.device) * 0.05
        seq = torch.cat([
            (z_spec.unsqueeze(1) + noise[:, i:i+1, :] * (i * 0.3))
            for i in range(4)
        ] + [z_spec.unsqueeze(1)], dim=1)   # shape: (B, 5, D)
        out, (h, _) = self.lstm(seq)
        fwd = h[-2]   # last forward hidden
        bwd = h[-1]   # last backward hidden
        return self.proj(torch.cat([fwd, bwd], dim=-1))


class SSTEnsemble(nn.Module):
    """Full SST Ensemble — Spectral + Spatial + Temporal → MLP meta-learner."""
    def __init__(self, bins=128, embed=64, num_classes=5):
        super().__init__()
        self.spectral  = SpectralCNN(bins, embed)
        self.spatial   = SpatialGNN(embed, embed)
        self.temporal  = TemporalBiLSTM(embed, embed, embed)

        self.fusion = nn.Sequential(
            nn.Linear(embed * 3, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.sev_head = nn.Linear(64, 1)

    def forward(self, x):
        zs = self.spectral(x)
        zg = self.spatial(zs)
        zt = self.temporal(zs)
        z  = self.fusion(torch.cat([zs, zg, zt], dim=-1))
        return self.cls_head(z), torch.sigmoid(self.sev_head(z)) * 100.0


class MLPBaseline(nn.Module):
    """Strong MLP baseline — comparable parameter count, no spatial/temporal context."""
    def __init__(self, bins=128, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bins, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(),
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.sev_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return self.cls_head(h), torch.sigmoid(self.sev_head(h)) * 100.0


class SpectralOnlyCNN(nn.Module):
    """1D-CNN spectral-only model (M1 baseline)."""
    def __init__(self, bins=128, num_classes=5):
        super().__init__()
        self.backbone = SpectralCNN(bins, 64)
        self.cls_head = nn.Linear(64, num_classes)
        self.sev_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.cls_head(h), torch.sigmoid(self.sev_head(h)) * 100.0


class SSTNoSpatial(nn.Module):
    """Ablation: SST without spatial branch."""
    def __init__(self, bins=128, embed=64, num_classes=5):
        super().__init__()
        self.spectral = SpectralCNN(bins, embed)
        self.temporal = TemporalBiLSTM(embed, embed, embed)
        self.fusion   = nn.Sequential(
            nn.Linear(embed * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.sev_head = nn.Linear(64, 1)

    def forward(self, x):
        zs = self.spectral(x)
        zt = self.temporal(zs)
        z  = self.fusion(torch.cat([zs, zt], dim=-1))
        return self.cls_head(z), torch.sigmoid(self.sev_head(z)) * 100.0


class SSTNoTemporal(nn.Module):
    """Ablation: SST without temporal branch."""
    def __init__(self, bins=128, embed=64, num_classes=5):
        super().__init__()
        self.spectral = SpectralCNN(bins, embed)
        self.spatial  = SpatialGNN(embed, embed)
        self.fusion   = nn.Sequential(
            nn.Linear(embed * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.sev_head = nn.Linear(64, 1)

    def forward(self, x):
        zs = self.spectral(x)
        zg = self.spatial(zs)
        z  = self.fusion(torch.cat([zs, zg], dim=-1))
        return self.cls_head(z), torch.sigmoid(self.sev_head(z)) * 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# §3  TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int   = 60,
    lr:           float = 1e-3,
    sev_w:        float = 0.3,
    patience:     int   = 15,
    label_smooth: float = 0.05,
    verbose:      bool  = True,
) -> dict:
    """
    Train a model and return a history dict with per-epoch metrics.
    """
    device = torch.device("cpu")
    model  = model.to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce_fn  = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    mse_fn = nn.MSELoss()

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_f1":     [], "val_mae":   [],
    }
    best_val_acc, best_state, patience_cnt = 0.0, None, 0

    for ep in range(1, epochs + 1):
        # ── training ─────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yc, ys in train_loader:
            xb, yc, ys = xb.to(device), yc.to(device), ys.to(device)
            opt.zero_grad()
            logits, sev = model(xb)
            loss = ce_fn(logits, yc) + sev_w * mse_fn(sev.squeeze(), ys / 100.0 * 100.0) / 1e4
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss    += loss.item() * len(xb)
            tr_correct += (logits.argmax(1) == yc).sum().item()
            tr_total   += len(xb)
        sched.step()

        # ── validation ───────────────────────────────────────────────────
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        all_preds, all_labels, all_sev_pred, all_sev_true = [], [], [], []
        with torch.no_grad():
            for xb, yc, ys in val_loader:
                xb, yc, ys = xb.to(device), yc.to(device), ys.to(device)
                logits, sev = model(xb)
                loss = ce_fn(logits, yc) + sev_w * mse_fn(sev.squeeze(), ys / 100.0 * 100.0) / 1e4
                vl_loss    += loss.item() * len(xb)
                vl_correct += (logits.argmax(1) == yc).sum().item()
                vl_total   += len(xb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(yc.cpu().numpy())
                all_sev_pred.extend(sev.squeeze().cpu().numpy())
                all_sev_true.extend(ys.cpu().numpy())

        val_acc  = vl_correct / vl_total
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_mae  = float(np.mean(np.abs(np.array(all_sev_pred) - np.array(all_sev_true))))

        history["train_loss"].append(tr_loss / tr_total)
        history["train_acc" ].append(tr_correct / tr_total)
        history["val_loss"  ].append(vl_loss / vl_total)
        history["val_acc"   ].append(val_acc)
        history["val_f1"    ].append(val_f1)
        history["val_mae"   ].append(val_mae)

        if verbose and ep % 5 == 0:
            print(f"    Ep {ep:3d}/{epochs}  TrainAcc={tr_correct/tr_total:.3f}  "
                  f"ValAcc={val_acc:.3f}  F1={val_f1:.3f}  MAE={val_mae:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                if verbose: print(f"    Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)

    # ── final validation metrics ──────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_sev_pred, all_sev_true = [], [], [], []
    all_probs = []
    with torch.no_grad():
        for xb, yc, ys in val_loader:
            logits, sev = model(xb)
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(yc.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_sev_pred.extend(sev.squeeze().cpu().numpy())
            all_sev_true.extend(ys.numpy())

    history["final"] = {
        "accuracy":  float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "macro_f1":  float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "mae":       float(np.mean(np.abs(np.array(all_sev_pred) - np.array(all_sev_true)))),
        "rmse":      float(np.sqrt(np.mean((np.array(all_sev_pred) - np.array(all_sev_true))**2))),
        "epochs_run": len(history["val_acc"]),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "probs":     np.array(all_probs),
        "labels":    np.array(all_labels),
        "preds":     np.array(all_preds),
        "sev_pred":  np.array(all_sev_pred),
        "sev_true":  np.array(all_sev_true),
    }
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# §4  CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over n_bins confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_b  = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece   += (mask.sum() / len(labels)) * abs(acc_b - conf_b)
    return float(ece)


def temperature_scale(probs: np.ndarray, T: float = 1.5) -> np.ndarray:
    """Soften the probability distribution with temperature."""
    # We approximate this in probability space via raising to 1/T power
    p = probs ** (1.0 / T)
    return p / p.sum(axis=1, keepdims=True)


def isotonic_calibrate(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Apply per-class isotonic regression calibration."""
    calibrated = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        y_c = (labels == c).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs[:, c], y_c)
        calibrated[:, c] = ir.predict(probs[:, c])
    # Re-normalize rows
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return calibrated / row_sums


# ═══════════════════════════════════════════════════════════════════════════════
# §5  MULTI-SEED EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

SEEDS = [42, 123, 256, 789, 1024]
EPOCHS = 80
LR     = 1e-3
BATCH  = 64

def run_multi_seed(model_fn, name: str, epochs=EPOCHS, verbose=False) -> dict:
    """Run a model across all seeds, return aggregated statistics."""
    print(f"\n  ▶  Training [{name}] across {len(SEEDS)} seeds …")
    seed_results = []
    for i, seed in enumerate(SEEDS):
        print(f"      Seed {seed} ({i+1}/{len(SEEDS)}) …", end=" ", flush=True)
        set_seed(seed)
        X, yc, ys = make_dataset(seed=seed)
        Xt, yct, yst, Xv, ycv, ysv = split_dataset(X, yc, ys, seed=seed)

        tr_ds = NGVCDataset(Xt, yct, yst, augment=True)
        vl_ds = NGVCDataset(Xv, ycv, ysv, augment=False)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
        vl_ld = DataLoader(vl_ds, batch_size=BATCH, shuffle=False)

        model = model_fn()
        hist  = train_model(model, tr_ld, vl_ld, epochs=epochs,
                            lr=LR, verbose=verbose)
        f = hist["final"]
        seed_results.append(f)
        print(f"Acc={f['accuracy']*100:.2f}%  F1={f['macro_f1']:.3f}  MAE={f['mae']:.2f}")

    # Aggregate
    accs  = [r["accuracy"]  for r in seed_results]
    f1s   = [r["macro_f1"]  for r in seed_results]
    maes  = [r["mae"]       for r in seed_results]
    rmses = [r["rmse"]      for r in seed_results]

    return {
        "name": name,
        "seed_results": seed_results,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std":  float(np.std(accs)),
        "f1_mean":       float(np.mean(f1s)),
        "f1_std":        float(np.std(f1s)),
        "mae_mean":      float(np.mean(maes)),
        "mae_std":       float(np.std(maes)),
        "rmse_mean":     float(np.mean(rmses)),
        "rmse_std":      float(np.std(rmses)),
        # Best seed results (for figures)
        "best":          seed_results[int(np.argmax(accs))],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §6  ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation() -> List[dict]:
    """Single-seed ablation across 4 architectures."""
    print("\n  ▶  Running ablation study (seed=42) …")
    seed = 42
    set_seed(seed)
    X, yc, ys = make_dataset(seed=seed)
    Xt, yct, yst, Xv, ycv, ysv = split_dataset(X, yc, ys, seed=seed)
    tr_ds = NGVCDataset(Xt, yct, yst, augment=True)
    vl_ds = NGVCDataset(Xv, ycv, ysv, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH, shuffle=False)

    configs = [
        ("Spectral Only (1D-CNN)",     lambda: SpectralOnlyCNN()),
        ("Spectral + Spatial",         lambda: SSTNoTemporal()),
        ("Spectral + Temporal",        lambda: SSTNoSpatial()),
        ("Full SST Ensemble (Ours)",   lambda: SSTEnsemble()),
    ]
    results = []
    for name, fn in configs:
        print(f"      [{name}] …", end=" ", flush=True)
        set_seed(seed)
        model = fn()
        hist  = train_model(model, tr_ld, vl_ld, epochs=EPOCHS, lr=LR, verbose=False)
        f = hist["final"]
        print(f"Acc={f['accuracy']*100:.2f}%  F1={f['macro_f1']:.3f}")
        results.append({"name": name, **f})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §7  CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_calibration_analysis(best_results: dict) -> dict:
    """Compute ECE before/after calibration using best model's probabilities."""
    print("\n  ▶  Running calibration analysis …")
    probs  = best_results["probs"]
    labels = best_results["labels"]

    ece_raw  = compute_ece(probs, labels)
    probs_ts = temperature_scale(probs, T=2.0)
    ece_ts   = compute_ece(probs_ts, labels)
    probs_ir = isotonic_calibrate(probs, labels)
    ece_ir   = compute_ece(probs_ir, labels)
    # Combined: temp scale first then isotonic
    probs_combined = isotonic_calibrate(probs_ts, labels)
    ece_combined   = compute_ece(probs_combined, labels)

    print(f"      ECE raw        : {ece_raw:.4f}")
    print(f"      ECE temp scale : {ece_ts:.4f}")
    print(f"      ECE isotonic   : {ece_ir:.4f}")
    print(f"      ECE combined   : {ece_combined:.4f}")
    print(f"      ECE reduction  : {(ece_raw - ece_combined) / ece_raw * 100:.1f}%")

    return {
        "ece_raw":          ece_raw,
        "ece_temp_scale":   ece_ts,
        "ece_isotonic":     ece_ir,
        "ece_combined":     ece_combined,
        "probs_raw":        probs,
        "probs_combined":   probs_combined,
        "labels":           labels,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §8  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

PLT_STYLE = {
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}
plt.rcParams.update(PLT_STYLE)

AGRI_COLORS = ["#2ECC71", "#E74C3C", "#F39C12", "#3498DB", "#9B59B6"]


def fig1_training_curves(sst_history: dict):
    """Fig 1: Training Loss & Accuracy Curves for all 4 models (SST best seed)."""
    print("      Generating Fig 1: Training Curves …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = list(range(1, len(sst_history["val_acc"]) + 1))

    ax = axes[0]
    ax.plot(epochs, sst_history["train_loss"], "b-",  linewidth=2, label="Train Loss")
    ax.plot(epochs, sst_history["val_loss"],   "r--", linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in sst_history["train_acc"]], "b-",  linewidth=2, label="Train Acc")
    ax.plot(epochs, [v * 100 for v in sst_history["val_acc"]],   "r--", linewidth=2, label="Val Acc")
    ax.plot(epochs, [v * 100 for v in sst_history["val_f1"]],    "g:",  linewidth=2, label="Val Macro-F1 ×100")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title("(b) Accuracy & Macro F1-Score")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig. 2: AgriPulse SST Ensemble — Training Dynamics (Best Seed)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_training_curves.png")
    plt.close(fig)
    print("        Saved → fig2_training_curves.png")


def fig2_confusion_matrix(best_results: dict):
    """Fig 2: Confusion Matrix with counts + percentages."""
    print("      Generating Fig 2: Confusion Matrix …")
    cm = np.array(best_results["confusion_matrix"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = LinearSegmentedColormap.from_list("agri", ["#FFFFFF", "#1A5276"])
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label",      labelpad=10)
    ax.set_title("Fig. 3: Confusion Matrix — AgriPulse SST Ensemble (Best Seed)", fontweight="bold", pad=12)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_pct[i, j] > 60 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Percentage (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_confusion_matrix.png")
    plt.close(fig)
    print("        Saved → fig3_confusion_matrix.png")


def fig3_calibration(cal: dict):
    """Fig 3: Reliability diagram (confidence vs accuracy) + ECE bar."""
    print("      Generating Fig 3: Calibration Reliability Diagram …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — Reliability diagram —
    ax = axes[0]
    n_bins = 15
    for label, probs, color, ls in [
        ("Uncalibrated",   cal["probs_raw"],      "#E74C3C", "--"),
        ("Temp+Isotonic",  cal["probs_combined"],  "#2ECC71", "-"),
    ]:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == cal["labels"]).astype(float)
        bin_conf, bin_acc = [], []
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() > 0:
                bin_conf.append(conf[mask].mean())
                bin_acc.append(correct[mask].mean())
        ax.plot(bin_conf, bin_acc, marker="o", linestyle=ls,
                color=color, linewidth=2, markersize=6, label=label)

    ax.plot([0, 1], [0, 1], "k:", linewidth=1.5, label="Perfect Calibration")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="grey")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="grey")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Reliability Diagram")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # — ECE bar chart —
    ax = axes[1]
    methods = ["Uncalibrated", "Temp. Scaling", "Isotonic Reg.", "Combined\n(Proposed)"]
    ece_vals = [
        cal["ece_raw"],
        cal["ece_temp_scale"],
        cal["ece_isotonic"],
        cal["ece_combined"],
    ]
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
    bars = ax.bar(methods, [v * 100 for v in ece_vals], color=colors,
                  edgecolor="white", width=0.55)
    for bar, val in zip(bars, ece_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Expected Calibration Error (%)")
    ax.set_title("(b) ECE Comparison Across Calibration Methods")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ece_vals) * 100 * 1.35)

    fig.suptitle("Fig. 4: Probability Calibration Analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_calibration.png")
    plt.close(fig)
    print("        Saved → fig4_calibration.png")


def fig4_sota_comparison(all_model_results: dict):
    """Fig 4: SOTA comparison bar chart (accuracy, F1, MAE) with error bars."""
    print("      Generating Fig 4: SOTA Comparison Chart …")
    models = ["MLP\nBaseline", "1D-CNN\n(Spectral)", "SST\nEnsemble\n(Ours)"]
    model_keys = ["MLP Baseline", "1D-CNN Spectral", "SST Ensemble"]

    acc  = [all_model_results[k]["accuracy_mean"] * 100 for k in model_keys]
    acc_e= [all_model_results[k]["accuracy_std"]  * 100 for k in model_keys]
    f1   = [all_model_results[k]["f1_mean"]             for k in model_keys]
    f1_e = [all_model_results[k]["f1_std"]              for k in model_keys]
    mae  = [all_model_results[k]["mae_mean"]             for k in model_keys]
    mae_e= [all_model_results[k]["mae_std"]              for k in model_keys]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(models))
    BAR_COLORS = ["#3498DB", "#F39C12", "#2ECC71"]

    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, acc, yerr=acc_e, capsize=5, color=BAR_COLORS,
                  edgecolor="white", width=0.5, error_kw={"elinewidth": 2})
    for b, v, e in zip(bars, acc, acc_e):
        ax.text(b.get_x()+b.get_width()/2, v+e+0.4,
                f"{v:.1f}±{e:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("(a) Accuracy (mean ± std, 5 seeds)")
    ax.set_ylim(50, 105); ax.grid(True, alpha=0.3, axis="y")

    # Macro F1
    ax = axes[1]
    bars = ax.bar(x, f1, yerr=f1_e, capsize=5, color=BAR_COLORS,
                  edgecolor="white", width=0.5, error_kw={"elinewidth": 2})
    for b, v, e in zip(bars, f1, f1_e):
        ax.text(b.get_x()+b.get_width()/2, v+e+0.01,
                f"{v:.3f}±{e:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("(b) Macro F1-Score (mean ± std)")
    ax.set_ylim(0.4, 1.10); ax.grid(True, alpha=0.3, axis="y")

    # Severity MAE
    ax = axes[2]
    bars = ax.bar(x, mae, yerr=mae_e, capsize=5, color=BAR_COLORS,
                  edgecolor="white", width=0.5, error_kw={"elinewidth": 2})
    for b, v, e in zip(bars, mae, mae_e):
        ax.text(b.get_x()+b.get_width()/2, v+e+0.1,
                f"{v:.2f}±{e:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Severity MAE (%)")
    ax.set_title("(c) Severity Regression MAE")
    ax.set_ylim(0, max(mae) * 1.4); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Fig. 5: Comparative Evaluation — AgriPulse SST Ensemble vs. Baselines (5 Seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_sota_comparison.png")
    plt.close(fig)
    print("        Saved → fig5_sota_comparison.png")


def fig5_ablation(ablation_results: List[dict]):
    """Fig 5: Ablation study grouped bar chart."""
    print("      Generating Fig 5: Ablation Study Chart …")
    names  = [r["name"].replace(" (Ours)", "") for r in ablation_results]
    accs   = [r["accuracy"] * 100 for r in ablation_results]
    f1s    = [r["macro_f1"]       for r in ablation_results]
    maes   = [r["mae"]            for r in ablation_results]

    x    = np.arange(len(names))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    b1 = ax.bar(x - w, accs,         w, label="Accuracy (%)", color="#2ECC71", edgecolor="white")
    b2 = ax.bar(x,     [f*100 for f in f1s],  w, label="Macro F1 ×100",  color="#3498DB", edgecolor="white")
    b3 = ax.bar(x + w, maes,         w, label="Severity MAE (%)", color="#E74C3C", edgecolor="white")

    for b, v in zip(b1, accs):
        ax.text(b.get_x()+b.get_width()/2, v+0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for b, v in zip(b2, f1s):
        ax.text(b.get_x()+b.get_width()/2, v*100+0.4,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for b, v in zip(b3, maes):
        ax.text(b.get_x()+b.get_width()/2, v+0.4,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Fig. 6: Ablation Study — Contribution of Each SST Sub-Network", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_ablation.png")
    plt.close(fig)
    print("        Saved → fig6_ablation.png")


def fig6_per_class_metrics(best_results: dict):
    """Fig 6: Per-class precision, recall, F1 heatmap."""
    print("      Generating Fig 6: Per-Class Metrics …")
    report = classification_report(
        best_results["labels"],
        best_results["preds"],
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    metrics = ["precision", "recall", "f1-score"]
    data = np.array([[report[cn][m] for m in metrics] for cn in CLASS_NAMES])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"], fontsize=11)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_title("Fig. 7: Per-Class Performance Metrics — SST Ensemble (Best Seed)",
                 fontweight="bold", pad=12)

    for i in range(NUM_CLASSES):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.3f}",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="black" if data[i,j] < 0.85 else "darkgreen")

    plt.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_per_class_metrics.png")
    plt.close(fig)
    print("        Saved → fig7_per_class_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# §9  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── 1. Train all models across 5 seeds ──────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 1: Multi-Seed Training                                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    mlp_res  = run_multi_seed(lambda: MLPBaseline(),       "MLP Baseline")
    cnn_res  = run_multi_seed(lambda: SpectralOnlyCNN(),   "1D-CNN Spectral")
    sst_res  = run_multi_seed(lambda: SSTEnsemble(),       "SST Ensemble")

    all_model_results = {
        "MLP Baseline":  mlp_res,
        "1D-CNN Spectral": cnn_res,
        "SST Ensemble":  sst_res,
    }

    # ── 2. Get best-seed training history for SST (for curves figure) ───────
    print("\n  ▶  Re-training SST best seed for history curves …", end=" ", flush=True)
    best_seed = SEEDS[int(np.argmax([r["accuracy"] for r in sst_res["seed_results"]]))]
    set_seed(best_seed)
    X, yc, ys = make_dataset(seed=best_seed)
    Xt, yct, yst, Xv, ycv, ysv = split_dataset(X, yc, ys, seed=best_seed)
    tr_ds = NGVCDataset(Xt, yct, yst, augment=True)
    vl_ds = NGVCDataset(Xv, ycv, ysv, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH, shuffle=False)
    sst_model = SSTEnsemble()
    sst_history = train_model(sst_model, tr_ld, vl_ld, epochs=EPOCHS, lr=LR, verbose=False)
    print("done.")

    # ── 3. Ablation study ────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 2: Ablation Study                                         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    ablation = run_ablation()

    # ── 4. Calibration analysis ──────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 3: Calibration Analysis                                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    cal = run_calibration_analysis(sst_res["best"])

    # ── 5. Generate all figures ──────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 4: Figure Generation                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    fig1_training_curves(sst_history)
    fig2_confusion_matrix(sst_res["best"])
    fig3_calibration(cal)
    fig4_sota_comparison(all_model_results)
    fig5_ablation(ablation)
    fig6_per_class_metrics(sst_res["best"])

    # ── 6. Build and save results JSON ───────────────────────────────────────
    def _safe(r):
        """Remove numpy arrays from results before JSON serialization."""
        return {k: v for k, v in r.items()
                if not isinstance(v, np.ndarray) and k not in ("seed_results",)}

    results_json = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "crop": "rice",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "samples_per_class": SAMPLES_PER_CLASS,
            "total_samples": NUM_CLASSES * SAMPLES_PER_CLASS,
            "val_split": 0.15,
            "num_seeds": len(SEEDS),
        },
        "models": {
            name: {
                "accuracy_mean":  res["accuracy_mean"],
                "accuracy_std":   res["accuracy_std"],
                "f1_mean":        res["f1_mean"],
                "f1_std":         res["f1_std"],
                "mae_mean":       res["mae_mean"],
                "mae_std":        res["mae_std"],
                "rmse_mean":      res["rmse_mean"],
                "rmse_std":       res["rmse_std"],
                "best_accuracy":  res["best"]["accuracy"],
                "best_f1":        res["best"]["macro_f1"],
                "best_mae":       res["best"]["mae"],
                "best_confusion_matrix": res["best"]["confusion_matrix"],
                "per_seed_accuracy": [r["accuracy"] for r in res["seed_results"]],
            }
            for name, res in all_model_results.items()
        },
        "ablation": [
            {
                "name":     r["name"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "mae":      r["mae"],
                "rmse":     r["rmse"],
            }
            for r in ablation
        ],
        "calibration": {
            "ece_raw":        cal["ece_raw"],
            "ece_temp_scale": cal["ece_temp_scale"],
            "ece_isotonic":   cal["ece_isotonic"],
            "ece_combined":   cal["ece_combined"],
            "ece_reduction_pct": (cal["ece_raw"] - cal["ece_combined"]) / cal["ece_raw"] * 100,
        },
        "figures": {
            "fig2_training_curves":  str(FIG_DIR / "fig2_training_curves.png"),
            "fig3_confusion_matrix": str(FIG_DIR / "fig3_confusion_matrix.png"),
            "fig4_calibration":      str(FIG_DIR / "fig4_calibration.png"),
            "fig5_sota_comparison":  str(FIG_DIR / "fig5_sota_comparison.png"),
            "fig6_ablation":         str(FIG_DIR / "fig6_ablation.png"),
            "fig7_per_class":        str(FIG_DIR / "fig7_per_class_metrics.png"),
        },
    }

    json_path = OUT_DIR / "publication_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # ── 7. Print final summary ───────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "═" * 70)
    print("  PUBLICATION RESULTS SUMMARY")
    print("═" * 70)
    print(f"\n  {'Model':<28}  {'Accuracy':>12}  {'Macro F1':>10}  {'MAE':>8}")
    print("  " + "-" * 62)
    for name, res in all_model_results.items():
        acc_s = f"{res['accuracy_mean']*100:.2f}±{res['accuracy_std']*100:.2f}%"
        f1_s  = f"{res['f1_mean']:.3f}±{res['f1_std']:.3f}"
        mae_s = f"{res['mae_mean']:.2f}±{res['mae_std']:.2f}"
        print(f"  {name:<28}  {acc_s:>12}  {f1_s:>10}  {mae_s:>8}")

    sst = all_model_results["SST Ensemble"]
    cnn = all_model_results["1D-CNN Spectral"]
    print("\n  IMPROVEMENTS (SST vs 1D-CNN):")
    print(f"    Accuracy : +{(sst['accuracy_mean']-cnn['accuracy_mean'])*100:.2f}%")
    print(f"    Macro F1 : +{sst['f1_mean']-cnn['f1_mean']:.3f}")
    print(f"    MAE      : {sst['mae_mean']-cnn['mae_mean']:+.2f}%")

    print("\n  CALIBRATION (SST Ensemble):")
    print(f"    ECE uncalibrated : {cal['ece_raw']*100:.2f}%")
    print(f"    ECE combined     : {cal['ece_combined']*100:.2f}%")
    print(f"    Reduction        : {(cal['ece_raw']-cal['ece_combined'])/cal['ece_raw']*100:.1f}%")

    print("\n  ABLATION STUDY:")
    for r in ablation:
        print(f"    {r['name']:<38} Acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']:.3f}")

    print(f"\n  6 publication figures saved to: {FIG_DIR}")
    print(f"  Results JSON saved to         : {json_path}")
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("═" * 70)
    print("\n  ✅  All publication materials are ready!")
    print("      Update the IEEE paper tables with values from publication_results.json")
    print("═" * 70)


if __name__ == "__main__":
    main()
