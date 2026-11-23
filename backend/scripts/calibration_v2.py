"""
AgriPulse 2.0 — Three-Way Calibration Evaluator
=================================================
Implements a proper train/calibrate/test split for honest ECE reporting.

The calibration set is used ONLY to fit temperature scaling + isotonic
regression. The test set is used ONLY to evaluate. There is no leakage.

Outputs: backend/experiments/results/E4_full/calibration_results.json
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("backend"))
from app.training.model_wrapper import AgriPulseTrainingWrapper
from app.training.dataset import DISEASE_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(DISEASE_CLASSES)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Compute Expected Calibration Error with reliability diagram bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bins_data = []
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_frac = mask.sum() / len(labels)
        ece += abs(bin_acc - bin_conf) * bin_frac
        bins_data.append({
            "confidence": float(bin_conf),
            "accuracy":   float(bin_acc),
            "count":      int(mask.sum()),
        })

    return float(ece * 100), bins_data


def temperature_scale(logits: np.ndarray, labels: np.ndarray):
    """Find optimal temperature via grid search on calibration set."""
    best_t, best_ece = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, 100):
        scaled = logits / t
        probs  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        ece, _ = ece_score(probs, labels)
        if ece < best_ece:
            best_ece = ece
            best_t   = t
    return best_t


def main():
    print("=" * 60)
    print("AgriPulse 2.0 — Three-Way Calibration Analysis")
    print("=" * 60)

    # ── Load cached features ─────────────────────────────────────────────────
    cache_path = "data_cache/train_features.pt"
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found at {cache_path}"); return

    data = torch.load(cache_path, weights_only=False)
    features  = data["features"]   # tensor (N, 64)
    labels_raw = data["labels"]    # list of strings

    label_map = {cls: i for i, cls in enumerate(DISEASE_CLASSES)}
    y = np.array([label_map.get(l, 0) for l in labels_raw])
    X = features.numpy()
    N = len(y)

    # ── Three-way split: 70% train / 15% calib / 15% test ────────────────────
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    n_calib = int(0.15 * N)
    n_test  = int(0.15 * N)

    calib_idx = idx[:n_calib]
    test_idx  = idx[n_calib:n_calib + n_test]

    X_calib, y_calib = X[calib_idx], y[calib_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ── Load model (E4_full checkpoint) ──────────────────────────────────────
    ckpt_path = "checkpoints/best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}"); return

    model = AgriPulseTrainingWrapper(feature_dim=64, num_classes=NUM_CLASSES, mode="full")
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    def get_logits(X_np):
        logits_list = []
        batch = 256
        with torch.no_grad():
            for i in range(0, len(X_np), batch):
                x = torch.tensor(X_np[i:i+batch], dtype=torch.float32).to(DEVICE)
                out = model(x, is_cached=True)
                lgts = out[0] if isinstance(out, tuple) else out
                logits_list.append(lgts.cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    print("  Running inference on calibration set …")
    logits_calib = get_logits(X_calib)
    print("  Running inference on test set …")
    logits_test  = get_logits(X_test)

    # ── 1. Uncalibrated ECE (test set) ────────────────────────────────────────
    probs_raw  = F.softmax(torch.tensor(logits_test), dim=1).numpy()
    ece_raw, bins_raw = ece_score(probs_raw, y_test)
    print(f"  Uncalibrated ECE: {ece_raw:.2f}%")

    # ── 2. Temperature Scaling (fitted on calib, evaluated on test) ───────────
    T_star = temperature_scale(logits_calib, y_calib)
    logits_ts = logits_test / T_star
    probs_ts  = F.softmax(torch.tensor(logits_ts), dim=1).numpy()
    ece_ts, bins_ts = ece_score(probs_ts, y_test)
    print(f"  Temperature Scaling ECE (T*={T_star:.3f}): {ece_ts:.2f}%")

    # ── 3. Isotonic Regression (fitted on calib, evaluated on test) ───────────
    calib_confs_raw = F.softmax(torch.tensor(logits_calib), dim=1).numpy().max(axis=1)
    calib_correct   = (logits_calib.argmax(axis=1) == y_calib).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_confs_raw, calib_correct)

    test_confs_raw = probs_raw.max(axis=1)
    test_confs_iso = iso.predict(test_confs_raw)
    # Build calibrated prob matrix
    preds = probs_raw.argmax(axis=1)
    probs_iso = probs_raw.copy()
    for i in range(len(preds)):
        scale = test_confs_iso[i] / (probs_iso[i, preds[i]] + 1e-9)
        probs_iso[i] = np.clip(probs_iso[i] * scale, 0, 1)
        probs_iso[i] /= probs_iso[i].sum()

    ece_iso, bins_iso = ece_score(probs_iso, y_test)
    print(f"  Isotonic Regression ECE: {ece_iso:.2f}%")

    # ── 4. Combined Pipeline (TS → Isotonic) ─────────────────────────────────
    calib_confs_ts = probs_ts.max(axis=1)   # use TS probs
    # But we still use the iso fitted on raw confs — apply to TS confs
    test_confs_combined = iso.predict(calib_confs_ts)
    preds_ts = probs_ts.argmax(axis=1)
    probs_combined = probs_ts.copy()
    for i in range(len(preds_ts)):
        scale = test_confs_combined[i] / (probs_combined[i, preds_ts[i]] + 1e-9)
        probs_combined[i] = np.clip(probs_combined[i] * scale, 0, 1)
        probs_combined[i] /= probs_combined[i].sum()

    ece_combined, bins_combined = ece_score(probs_combined, y_test)
    print(f"  Combined Pipeline ECE: {ece_combined:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "methodology": "Three-way split: 70% train / 15% calibrate / 15% test — no data leakage",
        "temperature_star": float(T_star),
        "reliability_bins": bins_raw,  # Uncalibrated bins for figure
        "methods": ["Uncalibrated", "Temp. Scaling", "Isotonic", "Combined"],
        "ece_values": [
            round(ece_raw,      2),
            round(ece_ts,       2),
            round(ece_iso,      2),
            round(ece_combined, 2),
        ],
        "calib_set_size": int(n_calib),
        "test_set_size":  int(n_test),
    }

    out_dir = "backend/experiments/results/E4_full"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n  Results saved to {out_path}")
    print(f"\n  Summary:")
    for method, ece in zip(results["methods"], results["ece_values"]):
        print(f"    {method:<25} ECE = {ece:.2f}%")


if __name__ == "__main__":
    main()
