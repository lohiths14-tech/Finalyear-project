"""
Confidence Calibration & Reliability System
==========================================
Calibrates model confidence scores so they reflect true accuracy rates.
Handles both leaf-level and plot-level calibration.

Components:
  TemperatureScaler      — Single-parameter temperature scaling for logits → calibrated probs
  IsotonicCalibrator     — Non-parametric isotonic regression calibration
  ConfidenceCalibrator   — Combines temperature + isotonic for end-to-end calibration
  CalibrationEvaluator   — Evaluates calibration quality (ECE, MCE, NLL)
  OODDetector            — Out-of-distribution detection via max-probability + entropy thresholds
  PredictionStabilizer   — Stabilizes repeated scans via temporal smoothing
  ReliabilityScore       — Combines quality + calibration + OOD into single reliability score
"""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn.functional as F

# ── Calibration data structures ────────────────────────────────────


@dataclass
class CalibrationResult:
    """Output of confidence calibration."""

    calibrated_probs: np.ndarray  # shape: (batch, num_classes)
    temperature: float
    is_calibrated: bool
    confidence: np.ndarray  # max probability per sample
    method: str  # "temperature" | "isotonic" | "hybrid"


class ECEBin(NamedTuple):
    lower: float
    upper: float
    accuracy: float
    confidence: float
    count: int


# ── Temperature scaling ────────────────────────────────────


class TemperatureScaler:
    """
    Single-parameter calibration: divides logits by a learned temperature T > 1
    before softmax, which softens overconfident predictions.

    T = 1.0 → no calibration (original probs)
    T → ∞  → uniform distribution
    T slightly > 1 → mild softening (typical best)
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.05, temperature)

    def calibrate(
        self, logits: np.ndarray, temperature: Optional[float] = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        scaled = logits / T
        # Numerically stable softmax
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        return exp / (exp.sum(axis=-1, keepdims=True) + 1e-10)

    @staticmethod
    def fit_temperature(
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> float:
        """
        Find optimal temperature via gradient descent on NLL.
        Returns the fitted temperature value.
        """
        T = torch.tensor(1.0, requires_grad=True)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / T.clamp(min=0.05)
            nll = F.cross_entropy(scaled, labels_t)
            nll.backward()
            return nll

        optimizer.step(closure)
        return max(0.05, float(T.item()))


# ── Isotonic regression ──────────────────────────────────────────────


class IsotonicCalibrator:
    """
    Non-parametric isotonic regression calibration.
    Maps raw confidences to calibrated probabilities using a monotonic piecewise function.
    More expressive than temperature scaling but needs sufficient calibration data.
    """  # noqa: E501

    def __init__(self):
        self.mapping: Optional[np.ndarray] = None
        self.bin_edges: Optional[np.ndarray] = None
        self.fitted = False

    def fit(
        self,
        raw_probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic mapping from raw probabilities to true labels.
        Uses pool-adjacent-violators (PAV) algorithm.
        """
        max_probs = raw_probs.max(axis=1)
        self.bin_edges = np.linspace(0, 1, num_bins + 1)

        bin_accs = []
        bin_confs = []

        for i in range(num_bins):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            in_bin = (max_probs >= lo) & (max_probs < hi)
            if i == num_bins - 1:
                in_bin = (max_probs >= lo) & (max_probs <= hi)

            if in_bin.sum() < 1:
                bin_accs.append(0.5)
                bin_confs.append((lo + hi) / 2)
            else:
                accuracy = (
                    labels[in_bin] == raw_probs[in_bin].argmax(axis=1)
                ).mean()
                bin_accs.append(float(accuracy))
                bin_confs.append(max_probs[in_bin].mean())

        bin_accs = np.array(bin_accs)
        bin_confs = np.array(bin_confs)

        # PAV: enforce monotonicity
        self.mapping = self._pav(bin_accs)
        self.fitted = True
        return self

    @staticmethod
    def _pav(y: np.ndarray) -> np.ndarray:
        """Pool-adjacent-violators algorithm for isotonic regression."""
        n = len(y)
        if n == 0:
            return y
        isotonic = np.copy(y).astype(np.float64)
        weights = np.ones(n)
        stack = [(0, n - 1)]

        while stack:
            start, end = stack.pop()
            if start >= end:
                continue

            # Find best partition
            best_val = sum(isotonic[start: end + 1]) / (end - start + 1)
            for mid in range(start, end + 1):
                left = sum(
                    isotonic[start: mid + 1] * weights[start: mid + 1]
                ) / max(1, sum(weights[start: mid + 1]))
                right = sum(
                    isotonic[mid: end + 1] * weights[mid: end + 1]
                ) / max(1, sum(weights[mid: end + 1]))
                if abs(left - right) < abs(best_val - left):
                    best_val = left

            # Check if partition respects isotonic constraint
            for i in range(start, end):
                if isotonic[i] > isotonic[i + 1]:
                    # Merge violated adjacent blocks
                    avg = (
                        isotonic[i] * weights[i]
                        + isotonic[i + 1] * weights[i + 1]
                    ) / (weights[i] + weights[i + 1])
                    isotonic[i] = isotonic[i + 1] = avg
                    weights[i] = weights[i + 1] = weights[i] + weights[i + 1]
                    stack.append((start, end))
                    break

        return isotonic

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply fitted isotonic mapping."""
        if not self.fitted:
            return raw_probs

        max_probs = raw_probs.max(axis=1)
        calibrated = np.zeros_like(max_probs)

        for i in range(len(max_probs)):
            bin_idx = np.searchsorted(self.bin_edges[1:], max_probs[i])
            bin_idx = min(bin_idx, len(self.mapping) - 1)
            calibrated[i] = self.mapping[bin_idx]

        # Preserve class ranking: scale calibrated confidence per class
        result = raw_probs * (calibrated / (max_probs + 1e-10)).reshape(-1, 1)
        result = np.clip(result, 0, 1)
        # Re-normalize
        result = result / (result.sum(axis=1, keepdims=True) + 1e-10)
        return result


# ── End-to-end calibrator ────────────────────────────────────────


class ConfidenceCalibrator:
    """
    Combines temperature scaling + isotonic regression for robust calibration.

    Pipeline:
      1. Temperature scaling to soften extreme logits
      2. Isotonic regression to map confidences to real accuracy rates
      3. Optional: Platt scaling fallback for small datasets

    When no real calibration data is available, falls back to lightweight
    self-calibration using the model's own confidence distribution.
    """

    def __init__(self, min_calibration_samples: int = 100):
        self.temp_scaler = TemperatureScaler(temperature=1.0)
        self.isotonic = IsotonicCalibrator()
        self.fitted_temperature: float = 1.0
        self.isotonic_fitted = False
        self.min_samples = min_calibration_samples
        self._self_calibrated = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        use_isotonic: bool = True,
    ) -> "ConfidenceCalibrator":
        """Fit calibration parameters from held-out calibration set."""
        n = len(labels)
        if n < self.min_samples:
            # Too few samples — just temperature scaling
            self.fitted_temperature = TemperatureScaler.fit_temperature(
                logits, labels
            )
            self.temp_scaler = TemperatureScaler(
                temperature=self.fitted_temperature
            )
            self.isotonic_fitted = False
            return self

        # Split: 60% temperature fitting, 40% isotonic validation
        split = int(n * 0.6)
        idx = np.arange(n)
        np.random.shuffle(idx)
        temp_idx, isot_idx = idx[:split], idx[split:]

        # 1. Fit temperature on first split
        self.fitted_temperature = TemperatureScaler.fit_temperature(
            logits[temp_idx], labels[temp_idx]
        )
        self.temp_scaler = TemperatureScaler(
            temperature=self.fitted_temperature
        )

        # 2. Fit isotonic on second split (if available)
        if use_isotonic and len(isot_idx) >= 20:
            temp_probs = self.temp_scaler.calibrate(logits[isot_idx])
            self.isotonic.fit(temp_probs, labels[isot_idx])
            self.isotonic_fitted = True
        else:
            self.isotonic_fitted = False

        return self

    def fit_from_synthetic(
        self,
        spectral_net,
        num_classes: int = 5,
        num_samples: int = 300,
    ) -> "ConfidenceCalibrator":
        """
        Self-calibrate using synthetic NGVC samples run through the spectral network.

        Generates diverse NGVC histograms spanning the feature space, runs them
        through the spectral network to get embeddings, then forms synthetic class
        logits by projecting embeddings onto class prototypes.

        This is a fallback — real calibration requires real labeled data.
        """  # noqa: E501
        import torch

        device = next(spectral_net.parameters()).device

        # Generate diverse synthetic NGVC histograms
        # Cover the spectrum: healthy, early-stage, mid-stage, late-stage
        # disease
        profile_means = [
            np.random.dirichlet(np.ones(128) * 0.3),  # sparse
            np.random.dirichlet(np.ones(128) * 2.0),  # concentrated
            np.random.dirichlet(np.ones(128) * 0.8),  # moderate spread
        ]

        all_feats = []
        for i in range(num_samples):
            # Blend between profiles for diversity
            w1 = np.random.rand()
            w2 = np.random.rand()
            base = w1 * profile_means[i % 3] + w2 * profile_means[(i + 1) % 3]
            base = base / (base.sum() + 1e-10)
            # Add noise
            noisy = base + np.random.randn(128) * 0.02
            noisy = np.clip(noisy, 0, None)
            noisy = noisy / (noisy.sum() + 1e-10)

            ngvc_t = torch.FloatTensor(noisy).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = spectral_net(ngvc_t).cpu().numpy().flatten()

            # Create synthetic class logits by projecting features onto
            # class prototypes with varying sharpness
            sharpness = np.random.uniform(1.0, 3.0)
            class_logits = np.zeros(num_classes)
            for c in range(num_classes):
                offset = np.random.randn(64) * (0.5 + c * 0.2)
                similarity = np.dot(feat, feat + offset) / 64.0
                class_logits[c] = similarity * sharpness
            all_feats.append(class_logits)

        logits_arr = np.array(all_feats)

        # Synthetic "correct" labels: highest logit = correct class
        synthetic_labels = logits_arr.argmax(axis=1)
        # Add 15% label noise to simulate imperfect accuracy
        flip_mask = np.random.rand(num_samples) < 0.15
        alt_labels = np.random.randint(0, num_classes, size=num_samples)
        synthetic_labels = np.where(flip_mask, alt_labels, synthetic_labels)

        self.fitted_temperature = TemperatureScaler.fit_temperature(
            logits_arr, synthetic_labels
        )
        self.temp_scaler = TemperatureScaler(
            temperature=self.fitted_temperature
        )
        self.isotonic_fitted = False
        self._self_calibrated = True
        return self

    def calibrate(self, logits: np.ndarray) -> CalibrationResult:
        """Apply calibration pipeline to raw logits."""
        # Temperature scaling
        probs = self.temp_scaler.calibrate(logits)
        temperature = self.fitted_temperature

        # Isotonic refinement
        if self.isotonic_fitted:
            probs = self.isotonic.calibrate(probs)
            method = "hybrid"
        else:
            method = "temperature"

        confidence = np.max(probs, axis=1)
        calibrated_probs = probs

        return CalibrationResult(
            calibrated_probs=calibrated_probs,
            temperature=temperature,
            is_calibrated=self.isotonic_fitted
            or self.fitted_temperature != 1.0,
            confidence=confidence,
            method=method,
        )


# ── Calibration evaluation ────────────────────────────────────────────


class CalibrationEvaluator:
    """
    Measures calibration quality using Expected Calibration Error (ECE)
    and Negative Log-Likelihood (NLL).
    """

    @staticmethod
    def compute_ece(
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
    ) -> tuple[float, list[ECEBin]]:
        """
        Expected Calibration Error (Guo et al., 2017).
        Lower is better. Perfect calibration = 0.0.
        """
        max_probs = probs.max(axis=1)
        bins = np.linspace(0, 1, num_bins + 1)
        ece_bins: list[ECEBin] = []

        total_correct = 0
        total_count = 0
        ece_sum = 0.0

        for i in range(num_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (max_probs >= lo) & (max_probs < hi)
            if i == num_bins - 1:
                in_bin = (max_probs >= lo) & (max_probs <= hi)

            count = int(in_bin.sum())
            if count == 0:
                ece_bins.append(ECEBin(lo, hi, 0.0, 0.0, 0))
                continue

            avg_confidence = float(max_probs[in_bin].mean())
            predictions = probs[in_bin].argmax(axis=1)
            accuracy = float((predictions == labels[in_bin]).mean())

            ece_bins.append(ECEBin(lo, hi, accuracy, avg_confidence, count))
            ece_sum += abs(accuracy - avg_confidence) * count
            total_correct += int((predictions == labels[in_bin]).sum())
            total_count += count

        ece = ece_sum / max(total_count, 1)
        return float(ece), ece_bins

    @staticmethod
    def compute_nll(
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Negative log-likelihood of the predictions. Lower is better."""
        nll = 0.0
        for i in range(len(labels)):
            p = probs[i, labels[i]]
            nll -= np.log(max(p, 1e-10))
        return float(nll / len(labels))

    @staticmethod
    def compute_mce(
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
    ) -> float:
        """Maximum Calibration Error. Worst per-bin calibration error."""
        ece, bins = CalibrationEvaluator.compute_ece(probs, labels, num_bins)
        return float(
            max(abs(b.accuracy - b.confidence) for b in bins if b.count > 0)
        )

    @staticmethod
    def full_evaluation(
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
    ) -> dict:
        """Comprehensive calibration metrics."""
        ece, ece_bins = CalibrationEvaluator.compute_ece(
            probs, labels, num_bins
        )
        nll = CalibrationEvaluator.compute_nll(probs, labels)
        mce = CalibrationEvaluator.compute_mce(probs, labels, num_bins)

        # Per-bin breakdown for diagnostic
        per_bin = [
            {
                "bin": f"[{b.lower:.2f}, {b.upper:.2f})",
                "accuracy": round(b.accuracy, 4),
                "confidence": round(b.confidence, 4),
                "count": b.count,
                "delta": round(abs(b.accuracy - b.confidence), 4),
            }
            for b in ece_bins
            if b.count > 0
        ]

        return {
            "ECE": round(ece, 4),
            "MCE": round(mce, 4),
            "NLL": round(nll, 4),
            "per_bin": per_bin,
        }


class OODDetector:
    """
    Detects out-of-distribution (OOD) / unknown inputs.
    An OOD input is something the model hasn't been trained to recognize.

    Methods:
      - MSP: Maximum Softmax Probability
      - Entropy: predictive entropy threshold
      - Energy: energy score (Liu et al., 2020)
      - Mahalanobis: distance to closest class Gaussian
      - Hybrid: Energy + Mahalanobis
    """

    def __init__(
        self,
        msp_threshold: float = 0.60,
        entropy_threshold: float = 1.2,
        energy_threshold: float = -10.0,
        mahalanobis_threshold: float = 15.0,
        method: str = "hybrid",
    ):
        self.msp_threshold = msp_threshold
        self.entropy_threshold = entropy_threshold
        self.energy_threshold = energy_threshold
        self.mahalanobis_threshold = mahalanobis_threshold
        self.method = method
        self.class_means = None
        self.cov_inv = None

    def fit_mahalanobis(
        self, features: np.ndarray, labels: np.ndarray, num_classes: int
    ):
        """Fit empirical class means and covariance matrix for Mahalanobis OOD."""  # noqa: E501
        self.class_means = []
        for c in range(num_classes):
            class_feats = features[labels == c]
            if len(class_feats) > 0:
                self.class_means.append(class_feats.mean(axis=0))
            else:
                self.class_means.append(np.zeros(features.shape[1]))
        self.class_means = np.array(self.class_means)

        # Pooled covariance
        diffs = []
        for i in range(len(features)):
            c = labels[i]
            diffs.append(features[i] - self.class_means[c])
        diffs = np.array(diffs)
        cov = np.cov(diffs, rowvar=False)
        # Add slight regularization for stability
        self.cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)

    def score(
        self, logits: np.ndarray, features: Optional[np.ndarray] = None
    ) -> float:
        """Compute OOD score — higher = more OOD-like."""
        if self.method == "msp":
            probs = self._softmax(logits)
            return 1.0 - float(probs.max())

        elif self.method == "entropy":
            probs = self._softmax(logits)
            return float(-np.sum(probs * np.log(probs + 1e-10)))

        elif self.method == "energy":
            # Free energy: -T * log(sum(exp(f(x)/T)))
            val = -logsumexp(logits, axis=-1)
            return float(np.atleast_1d(val).ravel()[0])

        elif self.method == "mahalanobis":
            if features is not None and self.cov_inv is not None:
                return self._mahalanobis_dist(features)
            return 1.0

        elif self.method == "hybrid":
            # Combine Energy and Mahalanobis
            val = -logsumexp(logits, axis=-1)
            energy_score = float(np.atleast_1d(val).ravel()[0])
            if features is not None and self.cov_inv is not None:
                m_score = self._mahalanobis_dist(features)
                return energy_score + 0.1 * m_score
            return energy_score

        return 1.0

    def _mahalanobis_dist(self, features: np.ndarray) -> float:
        """Compute minimum squared Mahalanobis distance to class means."""
        if self.class_means is None or self.cov_inv is None:
            return 0.0
        feat = features.squeeze()
        if feat.ndim > 1:
            feat = feat.mean(axis=0)
        distances = []
        for c_mean in self.class_means:
            diff = feat - c_mean
            dist = float(np.dot(np.dot(diff, self.cov_inv), diff.T))
            distances.append(dist)
        return float(np.min(distances)) if distances else 0.0

    def is_ood(
        self, logits: np.ndarray, features: Optional[np.ndarray] = None
    ) -> tuple[bool, float, str]:
        """Returns (is_ood, ood_score, reason)."""
        score = self.score(logits, features)
        probs = self._softmax(logits)
        max_prob = float(probs.max())

        if self.method == "msp":
            if max_prob < self.msp_threshold:
                return (
                    True,
                    score,
                    f"low_max_prob ({max_prob:.3f} < {self.msp_threshold})",
                )
        elif self.method == "entropy":
            if score > self.entropy_threshold:
                return (
                    True,
                    score,
                    f"high_entropy ({score:.3f} > {self.entropy_threshold})",
                )
        elif self.method in ["energy", "hybrid"]:
            if (
                score > self.energy_threshold
            ):  # Modified logic: energy score should be typically negative for ID  # noqa: E501
                return (
                    True,
                    score,
                    f"high_energy ({score:.3f} > {self.energy_threshold})",
                )
        elif self.method == "mahalanobis":
            if score > self.mahalanobis_threshold:
                return (
                    True,
                    score,
                    f"high_mahalanobis ({score:.3f} > {self.mahalanobis_threshold})",  # noqa: E501
                )

        return False, score, "in_distribution"

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / (exp.sum(axis=-1, keepdims=True) + 1e-10)


# ── Prediction stabilizer ────────────────────────────────────────


class PredictionStabilizer:
    """
    Stabilizes predictions across repeated scans of the same region.
    Uses exponential moving average on logits + variance-based uncertainty.

    When a farmer scans the same leaf multiple times, this ensures
    the predictions don't flicker between similar disease classes.
    """

    def __init__(self, decay: float = 0.7, min_scans: int = 2):
        self.decay = decay
        self.min_scans = min_scans
        self._logit_buffer: list[np.ndarray] = []
        self._history: list[dict] = []

    def add_scan(self, logits: np.ndarray, metadata: Optional[dict] = None):
        """Add a new scan to the buffer."""
        self._logit_buffer.append(logits.copy())
        self._history.append(metadata or {})

    def get_stabilized(
        self,
        return_variance: bool = False,
    ) -> tuple[np.ndarray, Optional[float]]:
        """
        Returns (stabilized_probs, variance_if_requested).
        Variance is the per-class variance across scans — low variance = stable.
        """  # noqa: E501
        if len(self._logit_buffer) < self.min_scans:
            logits = self._logit_buffer[-1]
            probs = OODDetector._softmax(logits)
            return probs, None

        # Exponential moving average of logits
        ema_logit = self._logit_buffer[0].copy()
        for i, logits in enumerate(self._logit_buffer):
            w = self.decay ** (len(self._logit_buffer) - i - 1)
            ema_logit = ema_logit * (1 - w) + logits * w

        probs = OODDetector._softmax(ema_logit)

        # Per-class variance across scans
        if return_variance and len(self._logit_buffer) >= 3:
            all_probs = np.stack(
                [OODDetector._softmax(logit) for logit in self._logit_buffer]
            )
            variance = float(np.mean(np.var(all_probs, axis=0)))
        else:
            variance = None

        return probs, variance

    def reset(self):
        self._logit_buffer.clear()
        self._history.clear()

    @property
    def num_scans(self) -> int:
        return len(self._logit_buffer)


# ── Reliability scorer ─────────────────────────────────────────


@dataclass
class ReliabilityScore:
    """Combined reliability assessment."""

    score: float  # 0-1, higher = more reliable
    is_reliable: bool
    quality_factor: float  # from image quality assessment
    calibration_factor: float  # from confidence calibration
    ood_factor: float  # from OOD detection (1 = in-distribution)
    stability_factor: float  # from prediction stabilizer (1 = stable)
    rejection_reason: Optional[str]  # None if reliable


class ReliabilityScorer:
    """
    Combines all reliability signals into a single reliability score.

    Rejection reasons:
      - "ood": out-of-distribution input detected
      - "low_confidence": calibrated confidence below threshold
      - "poor_quality": image quality score below threshold
      - "unstable": predictions vary too much across scans
      - "multiple_signals": combination of quality + confidence issues
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        quality_threshold: float = 0.55,
        ood_threshold: float = 0.50,
        stability_threshold: float = 0.05,
    ):
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        self.ood_threshold = ood_threshold
        self.stability_threshold = stability_threshold

    def score(
        self,
        calibrated_confidence: float,
        quality_iqi: float,
        ood_score: float,  # 0 = OOD, 1 = in-distribution
        prediction_variance: Optional[float] = None,
    ) -> ReliabilityScore:
        """Compute combined reliability score with rejection reasons."""
        reasons: list[str] = []

        qual_factor = quality_iqi
        calib_factor = min(1.0, calibrated_confidence / 0.8)
        ood_factor = ood_score  # 1 = clearly in-distribution
        stab_factor = 1.0
        if prediction_variance is not None:
            stab_factor = max(0.0, 1.0 - prediction_variance / 0.2)
            if prediction_variance > self.stability_threshold:
                reasons.append("unstable")

        # Weighted combination
        score = (
            0.25 * qual_factor
            + 0.35 * calib_factor
            + 0.20 * ood_factor
            + 0.20 * stab_factor
        )

        if qual_factor < self.quality_threshold:
            reasons.append("poor_quality")
        if calib_factor < self.confidence_threshold:
            reasons.append("low_confidence")
        if ood_score < self.ood_threshold:
            reasons.append("ood")

        if not reasons and score < 0.40:
            reasons.append("multiple_signals")

        is_reliable = len(reasons) == 0 and score >= 0.40
        rejection_reason = reasons[0] if not is_reliable else None

        return ReliabilityScore(
            score=round(score, 3),
            is_reliable=is_reliable,
            quality_factor=round(qual_factor, 3),
            calibration_factor=round(calib_factor, 3),
            ood_factor=round(ood_factor, 3),
            stability_factor=round(stab_factor, 3),
            rejection_reason=rejection_reason,
        )
