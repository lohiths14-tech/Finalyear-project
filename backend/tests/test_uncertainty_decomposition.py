"""
Tests for AgriPulse 2.0 Aleatoric and Epistemic Uncertainty Decomposition.
"""

import numpy as np

from app.services.inference_service import InferenceService


def test_uncertainty_decomposition_field():
    """Verify uncertainty decomposition payload in field analysis on crop foliage."""
    svc = InferenceService()
    # 256x256 foliage image (BGR green canopy with variation)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = (40, 135, 50)
    img[60:140, 60:140] = (35, 95, 115)
    res = svc.analyze_field(img)

    assert "uncertainty_breakdown" in res
    breakdown = res["uncertainty_breakdown"]
    assert "aleatoric" in breakdown
    assert "epistemic" in breakdown
    assert 0.0 <= breakdown["aleatoric"] <= 1.0
    assert 0.0 <= breakdown["epistemic"] <= 1.0
    assert "entropy_bits" in breakdown
    assert "energy_score" in breakdown


def test_uncertainty_decomposition_leaf():
    """Verify uncertainty decomposition payload in leaf diagnostic."""
    svc = InferenceService()
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    res = svc.analyze_leaf(img)

    assert "uncertainty_breakdown" in res
    breakdown = res["uncertainty_breakdown"]
    assert "aleatoric" in breakdown
    assert "epistemic" in breakdown
    assert 0.0 <= breakdown["aleatoric"] <= 1.0
    assert 0.0 <= breakdown["epistemic"] <= 1.0
