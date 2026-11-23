"""
AgriPulse 2.0 — Real Paddy Doctor Training Pipeline
===================================================
Trains NGVCNet and AgriVisionEnsemble on the 10,407 real rice field images
from the Paddy Doctor dataset across 10 genuine pathology classes.

Complies strictly with the Real AI Policy:
- No synthetic Gaussian distributions or mock labels.
- Real ExG + Otsu vegetation segmentation and NGVC spectral extraction.
- Stratified train/val split with full evaluation (Macro F1, ECE, Brier, CM).
- Documented ICAR / IRRI evidence-grounded agronomic treatment guidelines.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add backend to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

try:
    from app.evaluation.metrics import ModelEvaluator
    from app.models.architectures import (
        AgriVisionEnsemble,
        NGVCNet,
        SpatialGraphNet,
    )
    from app.training.experiment_tracker import AgriExperimentTracker
    from app.utils.image_processing import AgriImageProcessor
except ImportError:
    from backend.app.evaluation.metrics import ModelEvaluator
    from backend.app.models.architectures import (
        AgriVisionEnsemble,
        NGVCNet,
        SpatialGraphNet,
    )
    from backend.app.training.experiment_tracker import AgriExperimentTracker
    from backend.app.utils.image_processing import AgriImageProcessor

# ---------------------------------------------------------------------------
# ICAR & IRRI Evidence-Grounded Rice Treatment Guidelines
# ---------------------------------------------------------------------------
PADDY_TREATMENTS = {
    "normal": (
        "Canopy health is optimal. Maintain standard N-P-K (120:60:40 kg/ha) "
        "fertilization and alternate wetting and drying (AWD) irrigation."
    ),
    "bacterial_leaf_blight": (
        "Spray Copper Oxychloride 50% WP @ 2.5 g/L combined with "
        "Streptocycline @ 0.1 g/L. Drain stagnant field water and withhold "
        "excess nitrogen top-dressing until lesion expansion halts."
    ),
    "bacterial_leaf_streak": (
        "Apply Copper Hydroxide 53.8% DF @ 2.0 g/L or Validamycin 3% L @ 2.0 "
        "ml/L. Ensure balanced potassium fertilization."
    ),
    "bacterial_panicle_blight": (
        "Apply Kasugamycin 3% SL @ 2.0 ml/L or Oxytetracycline formulations "
        "at booting and heading. Disinfect seed lots before next sowing."
    ),
    "blast": (
        "Apply Tricyclazole 75% WP @ 0.6 g/L or Isoprothiolane 40% EC @ 1.5 "
        "ml/L at early tillering and panicle emergence. Avoid excess urea."
    ),
    "brown_spot": (
        "Foliar spray of Mancozeb 75% WP @ 2.0 g/L or Propiconazole 25% EC @ "
        "1.0 ml/L at boot-leaf stage. Ensure adequate potassium nutrition."
    ),
    "dead_heart": (
        "Stem borer (Scirpophaga incertulas) infestation. Broadcast Cartap "
        "Hydrochloride 4G @ 25 kg/ha or spray Chlorantraniliprole 18.5% SC."
    ),
    "downy_mildew": (
        "Apply Metalaxyl + Mancozeb @ 2.5 g/L upon early chlorotic stripe "
        "detection. Ensure optimal drainage in low-lying field zones."
    ),
    "hispa": (
        "Leaf miner (Dicladispa armigera). Spray Chlorpyrifos 20% EC @ 2.5 "
        "ml/L or Quinalphos 25% EC @ 2.0 ml/L. Clip seedling tips."
    ),
    "tungro": (
        "Viral complex spread by Green Leafhopper (Nephotettix virescens). "
        "Spray Imidacloprid 17.8% SL @ 0.25 ml/L or Thiamethoxam 25% WG."
    ),
}


def _process_single_image(args: tuple[str, int]) -> tuple[np.ndarray, int]:
    """Worker function for parallel NGVC extraction."""
    img_path, label_idx = args
    processor = AgriImageProcessor()
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(128, dtype=np.float32), label_idx
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        proc_img, _ = processor.remove_soil_and_shadow(img)
        ngvc = processor.extract_ngvc(proc_img, bins=128)
    except Exception:
        ngvc = processor.extract_ngvc(img, bins=128)
    return ngvc, label_idx


class RealPaddyDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_or_load_features(
    data_dir: Path, cache_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads cached features or extracts them from real image files."""
    classes = sorted(
        [
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    )

    if cache_path.exists():
        print(
            f"Loading cached real features from {cache_path}...", flush=True
        )
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], list(data["classes"])

    print(
        f"Extracting real features from {data_dir} ({len(classes)} classes)...",
        flush=True,
    )
    tasks = []
    for cls_idx, cls_name in enumerate(classes):
        cls_folder = data_dir / cls_name
        for img_file in cls_folder.glob("*.jpg"):
            tasks.append((str(img_file), cls_idx))

    total_tasks = len(tasks)
    print(
        f"Found {total_tasks} real field images. Processing...", flush=True
    )
    t0 = time.time()

    workers = min(16, (os.cpu_count() or 4) * 2)
    features_list = []
    labels_list = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, (feat, lbl) in enumerate(
            executor.map(_process_single_image, tasks, chunksize=50)
        ):
            features_list.append(feat)
            labels_list.append(lbl)
            if (idx + 1) % 2000 == 0 or (idx + 1) == total_tasks:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(0.1, elapsed)
                print(
                    f"  [{idx + 1:5d}/{total_tasks:5d}] processed "
                    f"({rate:.1f} img/s)...",
                    flush=True,
                )

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        classes=np.array(classes),
    )
    print(
        f"Extracted & cached in {time.time() - t0:.1f}s to {cache_path}.",
        flush=True,
    )
    return features, labels, classes


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    output_path: Path,
    accuracy: float,
    macro_f1: float,
    ece: float,
) -> None:
    """Renders a high-resolution publication-quality confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=9,
            )

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix — Paddy Doctor\n"
        f"Val Acc: {accuracy:.1f}% | Macro F1: {macro_f1:.3f} | "
        f"ECE: {ece:.3f}",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_paddy_pipeline(
    data_dir: Path,
    output_dir: Path,
    cache_path: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    """Executes real training, calibration, and weight serialization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load real data
    features, labels, classes = extract_or_load_features(data_dir, cache_path)
    num_classes = len(classes)

    # 2. Stratified train/val split (80% train, 20% validation)
    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(
        f"Split: {len(x_train)} train, {len(x_val)} val samples.",
        flush=True,
    )

    train_loader = DataLoader(
        RealPaddyDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        RealPaddyDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )

    # 3. Models
    spectral_net = NGVCNet(input_dim=128, feature_dim=64).to(device)
    spatial_net = SpatialGraphNet(in_channels=64, out_channels=64).to(device)
    meta_learner = AgriVisionEnsemble(
        feature_dim=64, num_classes=num_classes
    ).to(device)

    optimizer = optim.AdamW(
        list(spectral_net.parameters())
        + list(spatial_net.parameters())
        + list(meta_learner.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    tracker = AgriExperimentTracker(
        experiment_name="AgriPulse_Real_Paddy_Doctor",
        tracking_uri="backend/logs/mlruns",
    )
    tracker.start_run(
        run_name="PaddyDoctor_Real_Training",
        tags={"dataset": "PaddyDoctor_10407", "device": str(device)},
    )
    tracker.log_params(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "num_classes": num_classes,
            "classes": classes,
        }
    )

    best_val_f1 = 0.0
    best_weights = {}

    print(
        "\nStarting model training on real Paddy Doctor data...", flush=True
    )
    for epoch in range(1, epochs + 1):
        spectral_net.train()
        spatial_net.train()
        meta_learner.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()

            spec_feats = spectral_net(feats)
            bs = feats.size(0)
            adj = torch.eye(bs, device=device)
            spat_feats = spatial_net(spec_feats, adj)
            zero_temp = torch.zeros(bs, 64, device=device)

            logits, _ = meta_learner(spec_feats, spat_feats, zero_temp)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(spectral_net.parameters())
                + list(spatial_net.parameters())
                + list(meta_learner.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += bs

        scheduler.step()
        train_acc = (correct / total) * 100.0
        train_loss = running_loss / total

        # Validation pass
        spectral_net.eval()
        spatial_net.eval()
        meta_learner.eval()

        val_correct = 0
        val_total = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                spec_feats = spectral_net(feats)
                bs = feats.size(0)
                adj = torch.eye(bs, device=device)
                spat_feats = spatial_net(spec_feats, adj)
                zero_temp = torch.zeros(bs, 64, device=device)

                logits, _ = meta_learner(spec_feats, spat_feats, zero_temp)
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += bs
                all_logits.append(logits.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_acc = (val_correct / val_total) * 100.0
        val_logits = np.concatenate(all_logits, axis=0)
        val_targets = np.concatenate(all_targets, axis=0)
        val_probs = np.exp(val_logits - val_logits.max(axis=-1, keepdims=True))
        val_probs = val_probs / val_probs.sum(axis=-1, keepdims=True)

        evaluator = ModelEvaluator(classes=classes)
        for i in range(len(val_targets)):
            evaluator.register_classification(
                predicted_class_idx=int(np.argmax(val_probs[i])),
                ground_truth_class_idx=int(val_targets[i]),
                severity_pred=20.0,
                severity_gt=20.0,
            )
        cls_eval = evaluator.compute_classification()
        cal_eval = evaluator.compute_calibration(
            probs=val_probs, labels=val_targets
        )

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                f"Macro F1: {cls_eval.macro_f1:.3f} | ECE: {cal_eval.ece:.3f}",
                flush=True,
            )

        if cls_eval.macro_f1 > best_val_f1:
            best_val_f1 = cls_eval.macro_f1
            best_weights = {
                "spectral": spectral_net.state_dict(),
                "spatial": spatial_net.state_dict(),
                "ensemble": meta_learner.state_dict(),
                "accuracy": val_acc,
                "macro_f1": cls_eval.macro_f1,
                "ece": cal_eval.ece,
                "adaptive_ece": cal_eval.adaptive_ece,
                "brier": cal_eval.brier_score,
                "nll": cal_eval.nll,
                "probs": val_probs,
                "targets": val_targets,
            }

    # 4. Save best models and real metrics report
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights["spectral"], output_dir / "spectral_net.pt")
    torch.save(best_weights["spatial"], output_dir / "spatial_net.pt")
    torch.save(best_weights["ensemble"], output_dir / "ensemble.pt")

    # Real confusion matrix
    cm = confusion_matrix(
        best_weights["targets"],
        best_weights["probs"].argmax(axis=-1),
        labels=list(range(num_classes)),
    )
    cm_path = output_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(
        cm,
        classes,
        cm_path,
        best_weights["accuracy"],
        best_weights["macro_f1"],
        best_weights["ece"],
    )

    # Save authentic classes.json
    classes_json_path = output_dir / "classes.json"
    classes_payload = {
        "classes": classes,
        "treatments": {
            cls_name: PADDY_TREATMENTS.get(
                cls_name, "Maintain standard agronomic field practices."
            )
            for cls_name in classes
        },
        "trained_on_real_data": True,
        "dataset_name": "Paddy Doctor (10,407 Indian field images)",
        "total_samples": len(features),
        "train_samples": len(x_train),
        "validation_samples": len(x_val),
        "validation_accuracy": round(best_weights["accuracy"], 2),
        "macro_f1": round(best_weights["macro_f1"], 4),
        "ece_score": round(best_weights["ece"], 4),
        "adaptive_ece": round(best_weights["adaptive_ece"], 4),
        "brier_score": round(best_weights["brier"], 4),
        "nll": round(best_weights["nll"], 4),
        "has_cm": True,
    }
    with open(classes_json_path, "w", encoding="utf-8") as f:
        json.dump(classes_payload, f, indent=4)

    tracker.log_metrics(
        {
            "val_accuracy": best_weights["accuracy"],
            "val_macro_f1": best_weights["macro_f1"],
            "val_ece": best_weights["ece"],
            "val_brier": best_weights["brier"],
        }
    )
    tracker.end_run()

    print("\n" + "=" * 60, flush=True)
    print("  ✅ Real Paddy Doctor Training Complete!", flush=True)
    print(f"  - Classes: {len(classes)} ({', '.join(classes)})", flush=True)
    print(
        f"  - Best Validation Accuracy: {best_weights['accuracy']:.2f}%",
        flush=True,
    )
    print(f"  - Best Macro F1: {best_weights['macro_f1']:.4f}", flush=True)
    print(f"  - Real ECE: {best_weights['ece']:.4f}", flush=True)
    print(f"  - Weights Saved to: {output_dir}", flush=True)
    print(f"  - Confusion Matrix: {cm_path}", flush=True)
    print(f"  - Classes Config: {classes_json_path}", flush=True)
    print("=" * 60, flush=True)
    return classes_payload


def main():
    parser = argparse.ArgumentParser(
        description="Train AgriPulse on Paddy Doctor"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backend/data/paddy_doctor",
        help="Path to paddy_doctor directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_weights",
        help="Path to save weights and classes.json",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="backend/data_cache/paddy_features.npz",
        help="Path to cache extracted features",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    args = parser.parse_args()

    train_paddy_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        cache_path=Path(args.cache_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
