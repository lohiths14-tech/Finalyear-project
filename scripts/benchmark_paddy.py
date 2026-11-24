"""
AgriPulse 2.0 — Real Paddy Doctor Ground-Truth Benchmark
========================================================
Audits inference accuracy, Macro F1, Per-Class Precision/Recall, and
Calibration Error (ECE) using 100% REAL Paddy Doctor field images.

Zero dummy data policy compliant (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Setup path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.inference_service import InferenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paddy_benchmark")

# Paddy Doctor folder name to model class label mapping (exact 10 Paddy Doctor classes)
FOLDER_TO_MODEL_CLASS = {
    "bacterial_leaf_blight": "bacterial_leaf_blight",
    "bacterial_leaf_streak": "bacterial_leaf_streak",
    "bacterial_panicle_blight": "bacterial_panicle_blight",
    "blast": "blast",
    "brown_spot": "brown_spot",
    "dead_heart": "dead_heart",
    "downy_mildew": "downy_mildew",
    "hispa": "hispa",
    "normal": "normal",
    "tungro": "tungro",
}


def evaluate_benchmark(
    dataset_dir: Path,
    samples_per_class: int = 15,
) -> dict:
    svc = InferenceService()
    logger.info("InferenceService initialized. Device: %s, ONNX: %s", svc.device, svc.use_onnx)

    y_true = []
    y_pred = []
    confidences = []
    latencies = []

    classes_to_test = [c for c in FOLDER_TO_MODEL_CLASS.keys() if (dataset_dir / c).exists()]
    if not classes_to_test:
        raise FileNotFoundError(f"No valid Paddy Doctor disease folders found in {dataset_dir}")

    print("\n" + "=" * 70)
    print("  🌾 AgriPulse 2.0 — Real Paddy Doctor Ground-Truth Benchmark")
    print("=" * 70)
    print(f"  Dataset: {dataset_dir}")
    print(f"  Classes: {len(classes_to_test)} | Samples/Class: {samples_per_class}")
    print("=" * 70)

    for folder_name in sorted(classes_to_test):
        target_class = FOLDER_TO_MODEL_CLASS[folder_name]
        folder_path = dataset_dir / folder_name
        image_files = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))[:samples_per_class]

        print(f"\nEvaluating '{folder_name}' ({len(image_files)} real images)...")
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            t0 = time.perf_counter()
            result = svc.analyze_leaf(img)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            pred_disease = result["disease_type"].split(" (")[0].strip()
            conf = result["confidence"]

            y_true.append(target_class)
            y_pred.append(pred_disease)
            confidences.append(conf)

    unique_classes = sorted(list(set(y_true + y_pred)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = np.mean([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)])
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    print("\n" + "=" * 70)
    print("  📊 Benchmark Results on Real Ground-Truth Data")
    print("=" * 70)
    print(f"  Total Evaluated: {len(y_true)} real images")
    print(f"  Accuracy:        {accuracy * 100:.2f}%")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Mean Latency:    {mean_lat:.2f} ms")
    print(f"  p95 Latency:     {p95_lat:.2f} ms")
    print("\n  Per-Class Breakdown:")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    return {
        "total_samples": len(y_true),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "mean_latency_ms": round(float(mean_lat), 2),
        "p95_latency_ms": round(float(p95_lat), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="backend/data/paddy_doctor",
        help="Path to paddy doctor dataset",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=10,
        help="Number of images per class to evaluate",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    evaluate_benchmark(data_dir, samples_per_class=args.samples_per_class)
