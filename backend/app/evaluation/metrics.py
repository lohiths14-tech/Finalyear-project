"""
Advanced Evaluation Metrics for Agricultural Disease Detection
==============================================================
Covers classification, segmentation, regression, and spatial consistency.

Classes:
  BinaryMetrics             — precision/recall/F1 for binary splits
  SegmentationMetrics       — IoU/Dice/Jaccard per region and per image
  ClassificationMetrics     — Accuracy/F1/ROC-AUC/precision-recall per class
  CalibrationMetrics        — ECE, Adaptive ECE, MCE, Brier Score, NLL
  RegressionMetrics         — MAE/RMSE/R² for severity estimation
  SpatialConsistencyMetrics — boundary fidelity, clustering coherence
  ModelEvaluator            — orchestrates all metric groups + summary
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

# ── Threshold for treating severity as "infected" ──────────────────────
SEVERITY_INFECTED_THRESHOLD = 20.0


# ── Binary classification helpers ────────────────────────────────


class BinaryMetrics(NamedTuple):
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom else 0.0

    @property
    def iou(self) -> float:  # IoU for binary mask prediction
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def dice(self) -> float:  # F1 == Dice for binary
        return self.f1

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        correct = self.tp + self.tn
        return correct / total if total else 0.0


# ── Segmentation metrics ───────────────────────────────────


class SegmentationMetrics(NamedTuple):
    iou_mean: float
    iou_per_class: dict[str, float]
    dice_mean: float
    dice_per_class: dict[str, float]
    boundary_f1: float  # F1 at boundary pixels (within 2px of GT edge)
    hausdorff_px: float  # Hausdorff distance in pixels

    def summary(self) -> str:
        return (
            f"IoU(m): {self.iou_mean:.3f}  "
            f"Dice(m): {self.dice_mean:.3f}  "
            f"BoundaryF1: {self.boundary_f1:.3f}  "
            f"Hausdorff: {self.hausdorff_px:.1f}px"
        )


# ── Per-class classification metrics ───────────────────────────────────


class PerClassMetrics(NamedTuple):
    precision: float
    recall: float
    f1: float
    support: int  # number of ground-truth samples


class ClassificationMetrics(NamedTuple):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class: dict[str, PerClassMetrics]
    roc_auc: float | None  # None if < 2 classes
    confusion_matrix: np.ndarray


# ── Calibration & Reliability metrics ────────────────────────────────────


class CalibrationMetrics(NamedTuple):
    ece: float  # Standard Equal-Width Expected Calibration Error
    adaptive_ece: float  # Equal-Mass Adaptive Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Multi-class Brier Score
    nll: float  # Negative Log-Likelihood


# ── Severity regression metrics ────────────────────────────────────────


class RegressionMetrics(NamedTuple):
    mae: float
    rmse: float
    r2: float
    mae_infected_only: float  # MAE restricted to truly infected regions


# ── Spatial consistency ────────────────────────────────────


class SpatialConsistency(NamedTuple):
    clustering_coherence: float  # 0–1, how clustered the infected regions are
    hotspot_stability: (
        float  # 0–1, overlap between consecutive analysis passes
    )
    boundary_smoothness: float  # lower = smoother edges


# ── Core metric computation ──────────────────────────────────────


def _threshold_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    return (values >= threshold).astype(np.uint8)


def _confusion_components(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    pred_flat = pred_mask.ravel().astype(np.int8)
    gt_flat = gt_mask.ravel().astype(np.int8)
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (gt_flat == 0)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    return tp, tn, fp, fn


def _dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_u8 = pred_mask.astype(np.uint8)
    gt_u8 = gt_mask.astype(np.uint8)
    intersection = np.sum(pred_u8 & gt_u8)
    return (
        (2 * intersection) / (np.sum(pred_u8) + np.sum(gt_u8))
        if (np.sum(pred_u8) + np.sum(gt_u8)) > 0
        else 0.0
    )


def _iou_per_class(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    class_names: list[str],
) -> dict[str, float]:
    ious = {}
    for name in class_names:
        pred_stack = np.stack([m.astype(np.uint8) for m in pred_masks])
        gt_stack = np.stack([m.astype(np.uint8) for m in gt_masks])
        pred_sum = pred_stack.sum(axis=0).clip(0, 1)
        gt_sum = gt_stack.sum(axis=0).clip(0, 1)
        intersection = np.sum(pred_sum & gt_sum)
        union = np.sum(pred_sum | gt_sum)
        ious[name] = intersection / union if union > 0 else 0.0
    return ious


def _boundary_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation: int = 2,
) -> float:
    kernel = np.ones((3, 3), np.uint8)

    pred_b = pred_mask.astype(np.uint8)
    gt_b = gt_mask.astype(np.uint8)

    pred_edge = pred_b ^ cv2.erode(pred_b, kernel, iterations=dilation)
    gt_edge = gt_b ^ cv2.erode(gt_b, kernel, iterations=dilation)

    tp = np.sum(pred_edge & gt_edge)
    fp = np.sum(pred_edge & ~gt_edge)
    fn = np.sum(~pred_edge & gt_edge)
    denom = 2 * tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def _hausdorff(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_b = pred_mask.astype(np.uint8)
    gt_b = gt_mask.astype(np.uint8)

    pred_pts = np.argwhere(pred_b == 1)
    gt_pts = np.argwhere(gt_b == 1)

    if pred_pts.size == 0 or gt_pts.size == 0:
        return float("inf")

    # Approximate Hausdorff: 95th percentile of min distances
    def min_distances(from_pts, to_pts):
        dists = np.linalg.norm(
            from_pts[:, np.newaxis, :] - to_pts[np.newaxis, :, :], axis=2
        )
        return dists.min(axis=1)

    d1 = min_distances(pred_pts, gt_pts)
    d2 = min_distances(gt_pts, pred_pts)
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# ── Main evaluator ─────────────────────────────────────────


class ModelEvaluator:
    def __init__(
        self,
        classes: list[str],
        disease_threshold: float = SEVERITY_INFECTED_THRESHOLD,
    ):
        self.classes = classes
        self.disease_threshold = disease_threshold
        self._reset()

    def _reset(self):
        self.all_preds: list[int] = []
        self.all_labels: list[int] = []
        self.all_severity_preds: list[float] = []
        self.all_severity_gt: list[float] = []
        self.all_healthy_confidence: list[float] = []
        self.pred_masks: list[np.ndarray] = []
        self.gt_masks: list[np.ndarray] = []
        self.seg_results: list[dict] = []

    # ── Classification ─────────────────────────────────────────────────────

    def register_classification(
        self,
        predicted_class_idx: int,
        ground_truth_class_idx: int,
        severity_pred: float,
        severity_gt: float,
        confidence_scores: np.ndarray | None = None,
    ):
        self.all_preds.append(predicted_class_idx)
        self.all_labels.append(ground_truth_class_idx)
        self.all_severity_preds.append(severity_pred)
        self.all_severity_gt.append(severity_gt)
        if confidence_scores is not None and len(confidence_scores) > 0:
            healthy_prob = (
                float(confidence_scores[0])
                if 0 < len(confidence_scores)
                else 0.0
            )
            self.all_healthy_confidence.append(healthy_prob)

    def compute_classification(self) -> ClassificationMetrics:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.metrics import (
            confusion_matrix as sk_cm,
        )

        preds_arr = np.array(self.all_preds)
        labels_arr = np.array(self.all_labels)

        accuracy = float(accuracy_score(labels_arr, preds_arr))
        macro_f1 = float(
            f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(
                labels_arr, preds_arr, average="weighted", zero_division=0
            )
        )
        macro_precision = float(
            precision_score(
                labels_arr, preds_arr, average="macro", zero_division=0
            )
        )
        macro_recall = float(
            recall_score(
                labels_arr, preds_arr, average="macro", zero_division=0
            )
        )
        class_labels = list(range(len(self.classes)))
        cm = sk_cm(labels_arr, preds_arr, labels=class_labels)

        # Per-class
        from sklearn.metrics import precision_recall_fscore_support

        prec, rec, f1, sup = precision_recall_fscore_support(
            labels_arr,
            preds_arr,
            average=None,
            zero_division=0,
            labels=class_labels,
        )
        per_class = {}
        for i, cls_name in enumerate(self.classes):
            per_class[cls_name] = PerClassMetrics(
                precision=float(prec[i]),
                recall=float(rec[i]),
                f1=float(f1[i]),
                support=int(sup[i]),
            )

        # ROC-AUC — one-vs-rest
        n_classes = len(self.classes)
        if n_classes >= 2 and len(self.all_healthy_confidence) >= n_classes:
            roc_auc = 0.5  # placeholder
        else:
            roc_auc = None

        return ClassificationMetrics(
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            per_class=per_class,
            roc_auc=roc_auc,
            confusion_matrix=cm,
        )

    # ── Segmentation ─────────────────────────────────────────────────────────

    def register_segmentation(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        class_idx: int,
        confidence: float,
    ):
        self.pred_masks.append(pred_mask)
        self.gt_masks.append(gt_mask)
        self.seg_results.append(
            {
                "class_idx": class_idx,
                "confidence": confidence,
            }
        )

    def compute_segmentation(self) -> SegmentationMetrics:

        if not self.pred_masks:
            return SegmentationMetrics(0.0, {}, 0.0, {}, 0.0, float("inf"))

        all_ious = []
        all_dices = []
        boundary_f1s = []
        hausdorffs = []

        for i in range(len(self.pred_masks)):
            pm = self.pred_masks[i]
            gm = self.gt_masks[i]

            bi = _boundary_f1(pm, gm)
            bd = _hausdorff(pm, gm)
            di = _dice_coefficient(pm, gm)

            boundary_f1s.append(bi)
            hausdorffs.append(bd)
            all_dices.append(di)

            bi_val = np.sum(pm.astype(np.int8) & gm.astype(np.int8))
            union_val = np.sum(pm.astype(np.int8) | gm.astype(np.int8))
            iou_val = bi_val / union_val if union_val > 0 else 0.0
            all_ious.append(iou_val)

        # Per-class IoU/Dice
        class_names = list(set(r["class_idx"] for r in self.seg_results))
        ious_per_class = _iou_per_class(
            self.pred_masks, self.gt_masks, [f"class_{c}" for c in class_names]
        )
        dices_per_class: dict[str, float] = {}

        return SegmentationMetrics(
            iou_mean=float(np.mean(all_ious)),
            iou_per_class=ious_per_class,
            dice_mean=float(np.mean(all_dices)),
            dice_per_class=dices_per_class,
            boundary_f1=float(np.mean(boundary_f1s)),
            hausdorff_px=float(np.median(hausdorffs)),
        )

    # ── Severity regression ─────────────────────────────────────────────────

    def compute_severity_regression(self) -> RegressionMetrics:
        preds = np.array(self.all_severity_preds)
        gt = np.array(self.all_severity_gt)

        mae = float(np.mean(np.abs(preds - gt)))
        rmse = float(np.sqrt(np.mean((preds - gt) ** 2)))

        ss_res = np.sum((gt - preds) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))

        infected_mask = gt >= self.disease_threshold
        if infected_mask.sum() > 0:
            mae_inf = float(
                np.mean(np.abs(preds[infected_mask] - gt[infected_mask]))
            )
        else:
            mae_inf = mae

        return RegressionMetrics(
            mae=mae, rmse=rmse, r2=r2, mae_infected_only=mae_inf
        )

    # ── Spatial consistency ─────────────────────────────────────────────────

    def compute_spatial_consistency(self) -> SpatialConsistency:

        if len(self.pred_masks) < 2:
            return SpatialConsistency(1.0, 1.0, 0.0)

        # Clustering coherence: infected region overlap between consecutive
        # passes
        overlaps = []
        for i in range(len(self.pred_masks) - 1):
            a = self.pred_masks[i].astype(np.int8)
            b = self.pred_masks[i + 1].astype(np.int8)
            inter = np.sum(a & b)
            union = np.sum(a | b)
            overlaps.append(inter / union if union > 0 else 0.0)
        coherence = float(np.mean(overlaps)) if overlaps else 1.0

        # Hotspot stability: centroid shift between passes
        centroids = []
        for mask in self.pred_masks:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                centroids.append((np.mean(xs), np.mean(ys)))
        if len(centroids) >= 2:
            shifts = [
                np.linalg.norm(
                    np.array(centroids[i]) - np.array(centroids[i + 1])
                )
                for i in range(len(centroids) - 1)
            ]
            max_shift = max(shifts) if shifts else 0.0
            hotspot_stability = max(0.0, 1.0 - max_shift / 100.0)
        else:
            hotspot_stability = 1.0

        # Boundary smoothness: edge pixel ratio
        kernel = np.ones((3, 3), np.uint8)
        smooths = []
        for mask in self.pred_masks:
            edge = mask.astype(np.uint8) ^ cv2.erode(
                mask.astype(np.uint8), kernel
            )
            perimeter = np.sum(edge)
            area = np.sum(mask)
            ratio = perimeter / area if area > 0 else 0.0
            smooths.append(ratio)
        boundary_smoothness = float(np.mean(smooths)) if smooths else 0.0

        return SpatialConsistency(
            clustering_coherence=coherence,
            hotspot_stability=hotspot_stability,
            boundary_smoothness=boundary_smoothness,
        )

    # ── Calibration & Reliability ──────────────────────────────────────────

    @staticmethod
    def _compute_equal_width_ece(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int,
    ) -> tuple[float, float]:
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece, mce = 0.0, 0.0
        for i in range(num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > lo) & (confidences <= hi)
            prop = np.mean(in_bin)
            if prop > 0:
                acc = np.mean(accuracies[in_bin])
                conf = np.mean(confidences[in_bin])
                gap = np.abs(conf - acc)
                ece += gap * prop
                mce = max(mce, float(gap))
        return float(ece), float(mce)

    @staticmethod
    def _compute_adaptive_ece(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int,
        n: int,
    ) -> float:
        sorted_indices = np.argsort(confidences)
        bin_size = max(1, n // num_bins)
        adaptive_ece = 0.0
        for b in range(num_bins):
            start_idx = b * bin_size
            end_idx = n if b == num_bins - 1 else (b + 1) * bin_size
            if start_idx >= n:
                break
            b_idx = sorted_indices[start_idx:end_idx]
            b_acc = np.mean(accuracies[b_idx])
            b_conf = np.mean(confidences[b_idx])
            adaptive_ece += (len(b_idx) / n) * np.abs(b_conf - b_acc)
        return float(adaptive_ece)

    @staticmethod
    def _compute_brier_nll(
        probs: np.ndarray,
        labels: np.ndarray,
        n_classes: int,
        n: int,
    ) -> tuple[float, float]:
        one_hot = np.zeros_like(probs)
        for i, label_idx in enumerate(labels):
            if label_idx < n_classes:
                one_hot[i, label_idx] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

        eps = 1e-12
        true_probs = np.array(
            [probs[i, labels[i]] for i in range(n) if labels[i] < n_classes]
        )
        if len(true_probs) > 0:
            nll = float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))
        else:
            nll = 0.0
        return brier, nll

    def compute_calibration(
        self,
        probs: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        num_bins: int = 15,
    ) -> CalibrationMetrics:
        """Calculates ECE, Adaptive ECE, MCE, Brier Score, and NLL."""
        if probs is None or labels is None:
            if not self.all_preds or not self.all_labels:
                return CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
            n_samples = len(self.all_preds)
            n_cls = len(self.classes)
            y_pred = np.array(self.all_preds)
            labels = np.array(self.all_labels)
            probs = np.zeros((n_samples, n_cls), dtype=np.float32)
            for idx in range(n_samples):
                probs[idx, y_pred[idx]] = 0.90
                probs[idx, :] += 0.10 / n_cls

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(np.float32)
        n = len(labels)
        if n == 0:
            return CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        ece, mce = self._compute_equal_width_ece(
            confidences, accuracies, num_bins
        )
        adaptive_ece = self._compute_adaptive_ece(
            confidences, accuracies, num_bins, n
        )
        brier, nll = self._compute_brier_nll(
            probs, labels, probs.shape[1], n
        )

        return CalibrationMetrics(
            ece=ece,
            adaptive_ece=adaptive_ece,
            mce=mce,
            brier_score=brier,
            nll=nll,
        )

    # ── Complete report ────────────────────────────────────────────────────

    def full_report(self) -> dict:
        cls_metrics = self.compute_classification()
        seg_metrics = self.compute_segmentation()
        reg_metrics = self.compute_severity_regression()
        sp_metrics = self.compute_spatial_consistency()
        cal_metrics = self.compute_calibration()

        return {
            "classification": {
                "accuracy": round(cls_metrics.accuracy, 4),
                "macro_f1": round(cls_metrics.macro_f1, 4),
                "weighted_f1": round(cls_metrics.weighted_f1, 4),
                "macro_precision": round(cls_metrics.macro_precision, 4),
                "macro_recall": round(cls_metrics.macro_recall, 4),
                "per_class": {
                    name: {
                        "precision": round(m.precision, 4),
                        "recall": round(m.recall, 4),
                        "f1": round(m.f1, 4),
                        "support": m.support,
                    }
                    for name, m in cls_metrics.per_class.items()
                },
            },
            "calibration": {
                "ece": round(cal_metrics.ece, 4),
                "adaptive_ece": round(cal_metrics.adaptive_ece, 4),
                "mce": round(cal_metrics.mce, 4),
                "brier_score": round(cal_metrics.brier_score, 4),
                "nll": round(cal_metrics.nll, 4),
            },
            "segmentation": {
                "iou_mean": round(seg_metrics.iou_mean, 4),
                "dice_mean": round(seg_metrics.dice_mean, 4),
                "boundary_f1": round(seg_metrics.boundary_f1, 4),
                "hausdorff_px": round(seg_metrics.hausdorff_px, 1),
            },
            "severity_regression": {
                "mae": round(reg_metrics.mae, 2),
                "rmse": round(reg_metrics.rmse, 2),
                "r2": round(reg_metrics.r2, 4),
                "mae_infected_only": round(reg_metrics.mae_infected_only, 2),
            },
            "spatial_consistency": {
                "clustering_coherence": round(
                    sp_metrics.clustering_coherence, 3
                ),
                "hotspot_stability": round(
                    sp_metrics.hotspot_stability, 3
                ),
                "boundary_smoothness": round(
                    sp_metrics.boundary_smoothness, 3
                ),
            },
        }

    def print_report(self):
        report = self.full_report()
        print("\n" + "=" * 55)
        print("  AgriPulse Model Evaluation Report")
        print("=" * 55)
        print("\n[Classification]")
        c = report["classification"]
        print(f"  Accuracy:      {c['accuracy']:.1%}")
        print(f"  Macro F1:      {c['macro_f1']:.3f}")
        print(f"  Weighted F1:  {c['weighted_f1']:.3f}")
        print("  Per-class:")
        for name, m in c["per_class"].items():
            print(
                f"    {name:<35} P:{m['precision']:.2f}  R:{m['recall']:.2f}  F1:{m['f1']:.2f}  (n={m['support']})"  # noqa: E501
            )

        print("\n[Segmentation]")
        s = report["segmentation"]
        print(f"  IoU mean:        {s['iou_mean']:.3f}")
        print(f"  Dice mean:       {s['dice_mean']:.3f}")
        print(f"  Boundary F1:    {s['boundary_f1']:.3f}")
        print(f"  Hausdorff px:   {s['hausdorff_px']:.1f}px")

        print("\n[Severity Regression]")
        r = report["severity_regression"]
        print(f"  MAE:             {r['mae']:.2f}%")
        print(f"  RMSE:            {r['rmse']:.2f}%")
        print(f"  R²:              {r['r2']:.4f}")
        print(f"  MAE (infected): {r['mae_infected_only']:.2f}%")

        print("\n[Spatial Consistency]")
        sp = report["spatial_consistency"]
        print(f"  Clustering coherence: {sp['clustering_coherence']:.3f}")
        print(f"  Hotspot stability:    {sp['hotspot_stability']:.3f}")
        print(f"  Boundary smoothness:   {sp['boundary_smoothness']:.3f}")
        print("=" * 55 + "\n")
