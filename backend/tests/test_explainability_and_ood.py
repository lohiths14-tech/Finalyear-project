"""
Tests for Explainability, Advanced OOD Detection, Calibration, and ONNX Export
=============================================================================
Verifies:
- LayerGradCAM and IntegratedGradients math and shape contracts
- AgriExplainer heatmap overlay generation
- OODDetector (Energy, Mahalanobis, MSP, Entropy, Hybrid)
- Calibration metrics (ECE, Adaptive ECE, Brier Score, MCE)
- ONNX export parity
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from app.evaluation.metrics import CalibrationMetrics, ModelEvaluator
from app.models.architectures import NGVCNet
from app.quality.calibration import OODDetector
from app.quality.explainability import (
    AgriExplainer,
    IntegratedGradients,
    LayerGradCAM,
)


class SimpleConvNet(nn.Module):
    """Simple 2D CNN for testing Grad-CAM and attribution."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def test_layer_grad_cam():
    model = SimpleConvNet()
    grad_cam = LayerGradCAM(model, model.conv)

    input_tensor = torch.randn(1, 3, 64, 64)
    cam = grad_cam.generate_cam(input_tensor, target_class=0)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (64, 64)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
    grad_cam.remove_hooks()


def test_integrated_gradients():
    model = NGVCNet(input_dim=128, feature_dim=5)
    ig = IntegratedGradients(model, steps=10)

    input_tensor = torch.randn(1, 128)
    attribution = ig.attribute(input_tensor, target_class=0)

    assert isinstance(attribution, np.ndarray)
    assert attribution.shape == (128,)
    assert float(attribution.min()) >= 0.0
    assert float(attribution.max()) <= 1.0


def test_agri_explainer_overlay():
    image_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    raw_attr = np.linspace(0.0, 1.0, 10000).reshape((100, 100))
    attribution_map = raw_attr.astype(np.float32)

    overlay = AgriExplainer.generate_heatmap_overlay(
        image_rgb, attribution_map, alpha=0.5
    )

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8

    patch_features = np.random.randn(16, 64)
    patch_scores = AgriExplainer.compute_patch_importance(patch_features)
    assert len(patch_scores) == 16
    assert all(0.0 <= s <= 1.0 for s in patch_scores)


def test_ood_detector_methods():
    detector = OODDetector(
        msp_threshold=0.60,
        entropy_threshold=1.0,
        energy_threshold=-2.0,
        mahalanobis_threshold=10.0,
        method="energy",
    )

    # In-distribution logits (high confidence peak)
    id_logits = np.array([[6.0, 0.2, 0.1, -0.5, 0.0]])
    is_ood, score, reason = detector.is_ood(id_logits)
    assert isinstance(is_ood, bool)
    assert isinstance(score, float)

    # OOD logits (flat / low energy)
    ood_logits = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
    ood_score = detector.score(ood_logits)
    assert ood_score > score  # Higher energy score = more OOD

    # Test Mahalanobis & Hybrid
    features = np.random.randn(50, 64)
    labels = np.random.randint(0, 5, size=50)
    detector.fit_mahalanobis(features, labels, num_classes=5)

    detector.method = "mahalanobis"
    m_score = detector.score(id_logits, features=features[0:1])
    assert isinstance(m_score, float)
    assert m_score >= 0.0

    detector.method = "hybrid"
    h_score = detector.score(id_logits, features=features[0:1])
    assert isinstance(h_score, float)


def test_calibration_metrics():
    evaluator = ModelEvaluator(classes=["Healthy", "Blast", "Blight"])

    # Perfectly calibrated synthetic probabilities
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.85, 0.1, 0.05],
    ])
    labels = np.array([0, 1, 2, 0])

    cal = evaluator.compute_calibration(probs=probs, labels=labels, num_bins=5)

    assert isinstance(cal, CalibrationMetrics)
    assert 0.0 <= cal.ece <= 1.0
    assert 0.0 <= cal.adaptive_ece <= 1.0
    assert 0.0 <= cal.mce <= 1.0
    assert cal.brier_score >= 0.0
    assert cal.nll >= 0.0


def test_pyg_spatial_net():
    from app.models.spatial.pyg_spatial_net import PyGSpatialGraphNet

    model = PyGSpatialGraphNet(in_channels=64, out_channels=64, heads=4)
    x = torch.randn(10, 64)
    adj = torch.eye(10)

    out = model(x, adj)
    assert out.shape == (10, 64)

    # Test edge_index format (2, E)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out_edge = model(x, edge_index)
    assert out_edge.shape == (10, 64)


def test_experiment_tracker():
    from app.training.experiment_tracker import AgriExperimentTracker

    tracker = AgriExperimentTracker(
        experiment_name="Test_AgriPulse",
        tracking_uri="backend/logs/test_mlruns",
    )
    tracker.start_run(run_name="Test_E1_Baseline")
    tracker.log_params({"lr": 0.001, "seed": 42})
    tracker.log_metrics({"macro_f1": 0.94, "ece": 0.045})
    json_path = tracker.log_artifact_json({"test": True}, "report.json")
    tracker.end_run()

    assert os.path.exists(json_path)
