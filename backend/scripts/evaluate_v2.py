"""
AgriPulse 2.0 — Comprehensive Evaluation Script
=================================================
Loads the E4_full checkpoint and the cached validation set.
Computes and saves:

  1. test_predictions.json   — Confusion matrix + per-class F1
  2. calibration_results.json (delegated to calibration_v2.py)
  3. ood_evaluation.json     — Entropy-based OOD metrics
  4. per_dataset_results.json — Per-dataset Macro F1 breakdown

NO dummy data. Every metric is derived from real model inference.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, classification_report

sys.path.insert(0, os.path.abspath("backend"))
from app.training.model_wrapper import AgriPulseTrainingWrapper
from app.training.dataset import DISEASE_CLASSES

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(DISEASE_CLASSES)
CKPT_PATH   = "checkpoints/best_model.pth"
CACHE_PATH  = "data_cache/val_features.pt"
OUT_DIR     = "backend/experiments/results/E4_full"


def load_model():
    model = AgriPulseTrainingWrapper(feature_dim=64, num_classes=NUM_CLASSES, mode="full")
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def run_inference(model, feats: np.ndarray, batch_size: int = 256):
    all_logits = []
    with torch.no_grad():
        for i in tqdm(range(0, len(feats), batch_size), desc="Inferring"):
            x = torch.tensor(feats[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            out = model(x, is_cached=True)
            lgts = out[0] if isinstance(out, tuple) else out
            all_logits.append(lgts.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def main():
    print("=" * 60)
    print("AgriPulse 2.0 — Comprehensive Evaluation")
    print("=" * 60)

    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: Checkpoint not found: {CKPT_PATH}"); return
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: Cache not found: {CACHE_PATH}"); return

    # ── Load cache ────────────────────────────────────────────────────────────
    print("Loading cached features …")
    data = torch.load(CACHE_PATH, weights_only=False)
    feats   = data["features"].numpy().astype(np.float32)
    labels_raw = data["labels"]   # list of strings
    datasets_raw = data.get("datasets", ["unknown"] * len(labels_raw))

    label_map = {cls: i for i, cls in enumerate(DISEASE_CLASSES)}
    y_true = np.array([label_map.get(l, 0) for l in labels_raw])
    dataset_tags = list(datasets_raw)

    print(f"Samples: {len(y_true)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading E4_full model …")
    model = load_model()

    # ── Run inference ─────────────────────────────────────────────────────────
    logits = run_inference(model, feats)
    probs  = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    confs  = probs.max(axis=1)

    # ── 1. Confusion Matrix + Per-Class F1 ────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    test_predictions = {
        "class_names":    DISEASE_CLASSES,
        "confusion_matrix": cm.tolist(),
        "macro_f1":       round(macro_f1, 4),
        "per_class_f1":   {cls: round(float(f), 4) for cls, f in zip(DISEASE_CLASSES, per_class_f1)},
        "overall_accuracy": float((y_true == y_pred).mean()),
    }

    # ── 2. OOD Evaluation via Predictive Entropy ──────────────────────────────
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=1)
    max_entropy = np.log(NUM_CLASSES)

    # Use high-entropy samples as proxy OOD (top 10% by entropy)
    threshold = np.percentile(entropy, 90)
    ood_mask  = entropy > threshold
    id_mask   = ~ood_mask

    # OOD detection accuracy at this threshold:
    # For an in-distribution dataset, OOD detector should flag low-entropy (confident) samples as ID
    # and high-entropy as OOD. Since we have no true OOD data, we report entropy statistics.
    ood_evaluation = {
        "method":         "Predictive entropy thresholding (in-distribution only)",
        "note":           (
            "True OOD evaluation requires a separate OOD dataset (e.g., non-plant images). "
            "These statistics characterize the entropy distribution of in-distribution predictions."
        ),
        "entropy_mean":   float(entropy.mean()),
        "entropy_std":    float(entropy.std()),
        "entropy_max":    float(max_entropy),
        "normalized_entropy_mean": float((entropy / max_entropy).mean()),
        "high_entropy_fraction":   float(ood_mask.mean()),
        "threshold_used":  float(threshold),
        "correct_when_low_entropy":  float((y_true[id_mask] == y_pred[id_mask]).mean()),
        "correct_when_high_entropy": float((y_true[ood_mask] == y_pred[ood_mask]).mean()),
    }

    # ── 3. Per-Dataset Breakdown ───────────────────────────────────────────────
    per_dataset = {}
    unique_datasets = set(dataset_tags)
    for ds in sorted(unique_datasets):
        mask = np.array([d == ds for d in dataset_tags])
        if mask.sum() == 0:
            continue
        ds_f1  = float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
        ds_acc = float((y_true[mask] == y_pred[mask]).mean())
        per_dataset[ds] = {
            "n_samples":  int(mask.sum()),
            "macro_f1":   round(ds_f1,  4),
            "accuracy":   round(ds_acc, 4),
        }

    per_dataset_results = {
        "model":    "E4_full",
        "datasets": per_dataset,
        "combined": {
            "n_samples": int(len(y_true)),
            "macro_f1":  round(macro_f1, 4),
            "accuracy":  round(float((y_true == y_pred).mean()), 4),
        },
    }

    # ── Save all ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{OUT_DIR}/test_predictions.json",   "w") as f: json.dump(test_predictions,    f, indent=4)
    with open(f"{OUT_DIR}/ood_evaluation.json",     "w") as f: json.dump(ood_evaluation,      f, indent=4)
    with open(f"{OUT_DIR}/per_dataset_results.json","w") as f: json.dump(per_dataset_results, f, indent=4)

    print("\n  ✓ test_predictions.json")
    print("  ✓ ood_evaluation.json")
    print("  ✓ per_dataset_results.json")

    print(f"\n  Overall Macro-F1 : {macro_f1:.4f}")
    print(f"  Overall Accuracy : {(y_true==y_pred).mean():.4f}")

    print("\n  Per-class F1:")
    for cls, f in zip(DISEASE_CLASSES, per_class_f1):
        print(f"    {cls:45s}  {f:.4f}")

    print("\n  Per-dataset breakdown:")
    for ds, stats in per_dataset.items():
        print(f"    {ds:20s}  F1={stats['macro_f1']:.4f}  Acc={stats['accuracy']:.4f}  N={stats['n_samples']}")


if __name__ == "__main__":
    main()
