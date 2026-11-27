"""
Out-of-Distribution (OOD) Detection Tests.

Validates that:
  1. OODDetector correctly flags anomalous / uncertain predictions.
  2. Pure noise or non-crop inputs result in is_ood=True.
  3. Confident in-distribution predictions result in is_ood=False.
  4. InferenceService returns OOD telemetry in live responses.
"""

import numpy as np
import pytest

from app.quality.calibration import OODDetector
from app.services.inference_service import InferenceService


class TestOODDetector:
    """Unit tests for the OODDetector module."""

    def test_confident_in_distribution_not_ood(self):
        """A peaked, confident logit vector should be recognized as in-distribution."""
        detector = OODDetector(msp_threshold=0.35, method="msp")
        confident_logits = np.array([8.0, 1.0, 0.5, 0.2, 0.1])

        is_ood, score, reason = detector.is_ood(confident_logits)
        assert is_ood is False
        assert score < 0.35

    def test_flat_uniform_is_ood(self):
        """Completely uniform logits (maximum uncertainty) should be flagged as OOD."""
        detector = OODDetector(msp_threshold=0.35, method="msp")
        uniform_logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        is_ood, score, reason = detector.is_ood(uniform_logits)
        assert is_ood is True
        assert "low_max_prob" in reason

    def test_entropy_based_ood(self):
        """Entropy-based OOD detection rejects high-entropy predictions."""
        detector = OODDetector(entropy_threshold=1.2, method="entropy")
        high_entropy_logits = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        low_entropy_logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        assert detector.is_ood(high_entropy_logits)[0] is True
        assert detector.is_ood(low_entropy_logits)[0] is False


class TestInferenceOODIntegration:
    """Integration tests verifying OOD detection in InferenceService."""

    @pytest.fixture
    def svc(self):
        return InferenceService()

    def test_leaf_analysis_includes_ood_telemetry(self, svc):
        """Leaf analysis response must include the 'ood' telemetry dictionary."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[:, :, 1] = 180  # Green leaf

        result = svc.analyze_leaf(image)
        assert "ood" in result
        assert "is_ood" in result["ood"]
        assert "ood_score" in result["ood"]
        assert "reason" in result["ood"]
        assert isinstance(result["ood"]["is_ood"], bool)

    def test_field_analysis_includes_ood_telemetry(self, svc):
        """Field analysis response must include the 'ood' telemetry dictionary."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[50:400, 50:590, 1] = 160

        result = svc.analyze_field(image)
        assert "ood" in result
        assert "is_ood" in result["ood"]
        assert "ood_score" in result["ood"]
        assert "reason" in result["ood"]
        assert isinstance(result["ood"]["is_ood"], bool)
