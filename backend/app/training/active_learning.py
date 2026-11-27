"""
Active Learning & Semi-Supervised Training Pipeline
====================================================
Collects difficult/unlabeled samples, triggers human labeling workflows,
and supports semi-supervised learning with pseudo-labeling.

Components:
  SampleSelector     — Uncertainty + diversity-based sample selection
  PseudoLabeler     — High-confidence pseudo-label generation
  LabelingTask      — Celery task for human labeling workflow integration
  ActiveLearningLoop — End-to-end active learning loop orchestrator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import threading

import numpy as np

# ── Core data structures ─────────────────────────────────────────


@dataclass
class SampleCandidate:
    """A candidate sample for active learning."""

    ngvc: np.ndarray
    image_id: str
    tile_id: int
    entropy: float
    margin: float  # difference between top-2 class probabilities
    feature_vector: np.ndarray | None = None
    pseudo_label: int | None = None
    pseudo_confidence: float = 0.0
    selected: bool = False
    labeled: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    @property
    def uncertainty_score(self) -> float:
        """Combine entropy and margin into a single uncertainty score."""
        return self.entropy * (1.0 - self.margin + 1e-6)


@dataclass
class LabeledSample:
    """A sample with a human-provided ground-truth label."""

    ngvc: np.ndarray
    label: int
    severity: float
    source: str  # "human" | "pseudo"
    image_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


# ── Uncertainty-based sample selection ───────────────────────────────


class SampleSelector:
    """
    Selects the most informative samples for labeling using uncertainty sampling
    combined with diversity-based representative selection.

    Strategies:
      - entropy: Select samples with highest predictive entropy
      - margin: Select samples where top-2 class probabilities are closest
      - least_confident: Select samples with lowest max-probability
      - random_baseline: Uniform random selection (baseline)
      - diversity: k-Means cluster + pick centroid-proximate samples
    """  # noqa: E501

    def __init__(
        self, strategy: str = "entropy", diversity_weight: float = 0.3
    ):
        self.strategy = strategy
        self.diversity_weight = diversity_weight

    def compute_uncertainty(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
    ) -> tuple[float, float, float]:
        """
        Compute uncertainty metrics from raw logits.

        Returns: (entropy, margin, max_prob)
        """
        # Temperature scaling (higher = softer distribution)
        scaled = logits / max(temperature, 1e-8)
        probs = softmax(scaled)
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        sorted_probs = np.sort(probs)[::-1]
        margin = (
            float(sorted_probs[0] - sorted_probs[1])
            if len(sorted_probs) > 1
            else 0.0
        )
        max_prob = float(sorted_probs[0])
        return entropy, margin, max_prob

    def select(
        self,
        candidates: list[SampleCandidate],
        budget: int,
    ) -> list[SampleCandidate]:
        """
        Select `budget` samples from candidates using the configured strategy.
        Returns selected candidates (marked selected=True).
        """
        if not candidates or budget <= 0:
            return []

        if self.strategy == "random_baseline":
            selected = np.random.choice(
                len(candidates), min(budget, len(candidates)), replace=False
            )
            for i in selected:
                candidates[i].selected = True
            return [candidates[i] for i in selected]

        # Compute uncertainty scores
        for c in candidates:
            if c.entropy == 0.0 and c.margin == 0.0:
                # Default: use uncertainty score if not pre-computed
                _ = c.uncertainty_score

        if self.strategy == "diversity":
            return self._select_diverse(candidates, budget)

        sort_keys = {
            "entropy": lambda c: -c.entropy,
            "margin": lambda c: c.margin,
            "least_confident": lambda c: -c.uncertainty_score
        }
        key_fn = sort_keys.get(self.strategy, lambda c: -c.uncertainty_score)
        sorted_c = sorted(candidates, key=key_fn)

        selected = sorted_c[:budget]
        for c in selected:
            c.selected = True
        return selected

    def _select_diverse(
        self,
        candidates: list[SampleCandidate],
        budget: int,
    ) -> list[SampleCandidate]:
        """k-Means-style diversity: pick samples closest to cluster centroids."""  # noqa: E501
        if not candidates:
            return []

        # Build feature matrix
        features = []
        for c in candidates:
            if c.feature_vector is not None:
                features.append(c.feature_vector)
            else:
                features.append(c.ngvc)
        features = np.stack(features)

        # Simple k-Means clustering
        n_clusters = min(budget, len(candidates))
        indices = np.random.choice(len(features), n_clusters, replace=False)
        centroids = features[indices].copy()

        # Assign each candidate to nearest centroid
        assignments = np.zeros(len(candidates), dtype=int)
        for i, f in enumerate(features):
            dists = np.linalg.norm(centroids - f, axis=1)
            assignments[i] = int(np.argmin(dists))

        # Pick one representative per cluster (lowest uncertainty among
        # assigned)
        selected: list[SampleCandidate] = []
        for k in range(n_clusters):
            cluster_mask = assignments == k
            cluster_candidates = [
                c for i, c in enumerate(candidates) if cluster_mask[i]
            ]
            if not cluster_candidates:
                continue
            best = min(cluster_candidates, key=lambda c: -c.uncertainty_score)
            best.selected = True
            selected.append(best)

        return selected


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x))


# ── Pseudo-labeling ────────────────────────────────────────


class PseudoLabeler:
    """
    Generates pseudo-labels for unlabeled samples using model confidence thresholds.
    Only high-confidence predictions become pseudo-labels.

    Config:
      confidence_threshold: minimum probability to accept a pseudo-label
      entropy_threshold: maximum entropy to accept a pseudo-label
      min_support: minimum number of samples required per class for semi-supervised training
    """  # noqa: E501

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        entropy_threshold: float = 0.4,
        min_support: int = 32,
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.min_support = min_support

    def generate_pseudo_labels(
        self,
        candidates: list[SampleCandidate],
    ) -> list[SampleCandidate]:
        """
        Assign pseudo-labels to high-confidence candidates.
        Returns candidates with pseudo_label and pseudo_confidence set.
        """
        accepted = []
        for c in candidates:
            if c.selected or c.labeled:
                continue

            # Check entropy threshold
            if c.entropy > self.entropy_threshold:
                continue

            # Check confidence threshold (1 - margin approximation)
            if 1.0 - c.margin < self.confidence_threshold:
                continue

            # Use existing pseudo_label if already set
            if (
                c.pseudo_label is not None
                and c.pseudo_confidence >= self.confidence_threshold
            ):
                c.source = "pseudo"
                accepted.append(c)
                continue

        return accepted

    def filter_by_class_support(
        self,
        candidates: list[SampleCandidate],
    ) -> list[SampleCandidate]:
        """
        Remove pseudo-labels for classes with fewer than min_support samples.
        """
        from collections import Counter

        labels = [
            c.pseudo_label for c in candidates if c.pseudo_label is not None
        ]
        counts = Counter(labels)
        return [
            c
            for c in candidates
            if c.pseudo_label is None
            or counts.get(c.pseudo_label, 0) >= self.min_support
        ]


# ── Active learning loop ─────────────────────────────────


class ActiveLearningLoop:
    """
    End-to-end active learning orchestrator:

      1. Collect inference results as sample candidates
      2. Compute uncertainty metrics
      3. Select most informative samples via SampleSelector
      4. Trigger human labeling (via Celery task or API)
      5. Append labeled samples to training set
      6. Retrain/reload model with new samples

    Usage:
      loop = ActiveLearningLoop(model_fn, get_dataloader_fn)
      loop.add_inference_result(ngvc, logits, image_id, tile_id)
      loop.select_samples(budget=50)
      loop.trigger_labeling()   # Returns samples needing human labels
      loop.add_labeled_samples(labeled_batch)
      if loop.should_retrain():
          loop.trigger_retraining()
    """

    def __init__(
        self,
        model_fn,  # () -> nn.Module — returns current model
        get_dataloader_fn,  # () -> DataLoader — returns training dataloader
        strategy: str = "entropy",
        diversity_weight: float = 0.3,
        confidence_threshold: float = 0.85,
        retrain_budget: int = 200,
        temperature: float = 1.0,
    ):
        self.model_fn = model_fn
        self.get_dataloader_fn = get_dataloader_fn
        self.selector = SampleSelector(
            strategy=strategy, diversity_weight=diversity_weight
        )
        self.pseudo_labeler = PseudoLabeler(
            confidence_threshold=confidence_threshold
        )
        self.temperature = temperature
        self.retrain_budget = retrain_budget

        self._candidates: list[SampleCandidate] = []
        self._labeled_samples: list[LabeledSample] = []
        self._lock = threading.Lock()
        self._retrain_triggered = False
        self._retrain_count = 0

    def add_inference_result(
        self,
        ngvc: np.ndarray,
        logits: np.ndarray,
        image_id: str,
        tile_id: int,
        feature_vector: np.ndarray | None = None,
    ):
        """Record an inference result as a candidate for active learning."""
        entropy, margin, max_prob = self.selector.compute_uncertainty(
            logits, temperature=self.temperature
        )
        candidate = SampleCandidate(
            ngvc=ngvc,
            image_id=image_id,
            tile_id=tile_id,
            entropy=entropy,
            margin=margin,
            feature_vector=feature_vector,
            pseudo_confidence=max_prob,
        )
        with self._lock:
            self._candidates.append(candidate)

    def select_samples(self, budget: int) -> list[SampleCandidate]:
        """Select `budget` most uncertain samples for labeling."""
        with self._lock:
            available = [
                c for c in self._candidates if not c.selected and not c.labeled
            ]
        selected = self.selector.select(available, budget)
        return selected

    def trigger_labeling(self) -> list[dict]:
        """
        Return list of samples requiring human labeling.
        Format: [{"ngvc": ..., "image_id": ..., "tile_id": ..., "id": ...}]
        """
        with self._lock:
            pending = [
                c for c in self._candidates if c.selected and not c.labeled
            ]
        return [
            {
                "id": f"{c.image_id}_tile_{c.tile_id}",
                "image_id": c.image_id,
                "tile_id": c.tile_id,
                "entropy": round(c.entropy, 4),
                "margin": round(c.margin, 4),
                "pseudo_label": c.pseudo_label,
            }
            for c in pending
        ]

    def add_labeled_samples(self, batch: list[LabeledSample]):
        """Add a batch of human-labeled samples to the training set."""
        with self._lock:
            self._labeled_samples.extend(batch)
            # Mark candidates as labeled
            for c in self._candidates:
                for s in batch:
                    if c.image_id == s.image_id and c.tile_id == s.tile_id:
                        c.labeled = True
        self._check_retrain()

    def add_pseudo_labeled_samples(self, batch: list[SampleCandidate]):
        """Add a batch of pseudo-labeled samples to the training set."""
        labeled = [
            LabeledSample(
                ngvc=c.ngvc,
                label=c.pseudo_label,
                severity=0.0,  # Pseudo-labels don't have severity
                source="pseudo",
                image_id=c.image_id,
            )
            for c in batch
            if c.pseudo_label is not None
        ]
        with self._lock:
            self._labeled_samples.extend(labeled)
            for c in batch:
                c.labeled = True
        self._check_retrain()

    def should_retrain(self) -> bool:
        return len(self._labeled_samples) >= self.retrain_budget

    def _check_retrain(self):
        if len(self._labeled_samples) >= self.retrain_budget:
            self._retrain_triggered = True
            self._retrain_count += 1

    def reset_retrain_flag(self):
        self._retrain_triggered = False

    @property
    def stats(self) -> dict:
        """Return current active learning statistics."""
        with self._lock:
            total = len(self._candidates)
            selected = sum(1 for c in self._candidates if c.selected)
            labeled = sum(1 for c in self._candidates if c.labeled)
            pseudo = sum(
                1
                for c in self._candidates
                if c.pseudo_label is not None and c.source == "pseudo"
            )
            human = sum(
                1 for c in self._candidates if c.labeled and not c.pseudo_label
            )
            avg_entropy = (
                np.mean([c.entropy for c in self._candidates])
                if total > 0
                else 0.0
            )
        return {
            "total_candidates": total,
            "selected_for_labeling": selected,
            "human_labeled": human,
            "pseudo_labeled": pseudo,
            "total_labeled": human + pseudo,
            "avg_entropy": round(float(avg_entropy), 4),
            "retrain_count": self._retrain_count,
            "pending_labeling": selected - labeled,
        }


# ── Labeling task (Celery integration) ───────────────────────────


def create_labeling_task(
    samples: list[dict], priority: str = "normal"
) -> dict:
    """
    Create a labeling task payload for human review.

    Returns a task dict compatible with the Celery task queue.
    The task can be submitted to a labeling service (e.g., Label Studio, internal UI).
    """  # noqa: E501
    return {
        "task_name": "active_learning.label_review",
        "samples": samples,
        "priority": priority,
        "model_version": "current",
        "created_at": datetime.now(UTC).isoformat(),
        "callback_url": "/api/v1/active-learning/feedback",
    }


def parse_labeling_feedback(payload: dict) -> list[LabeledSample]:
    """
    Parse human feedback from labeling service into LabeledSample objects.

    Expected payload format:
      {"labels": [{"image_id": ..., "tile_id": ..., "label": int, "severity": float}]}
    """  # noqa: E501
    samples = []
    for item in payload.get("labels", []):
        samples.append(
            LabeledSample(
                ngvc=np.zeros(
                    128, dtype=np.float32
                ),  # NGVC not included in feedback payload
                label=int(item["label"]),
                severity=float(item.get("severity", 0.0)),
                source="human",
                image_id=str(item["image_id"]),
            )
        )
    return samples
