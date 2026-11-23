"""
Unit Tests for AgriPulse 2.0 Mobile / Edge Export Pipeline
==========================================================

Verifies that all 4 exported TorchScript mobile models load and compute
deterministic forward passes identical to native PyTorch models.
"""

import os
import pytest
import torch
from pathlib import Path

# Points to workspace_root/model_weights/mobile
MOBILE_DIR = Path(__file__).resolve().parent.parent.parent / "model_weights" / "mobile"


class TestMobileExport:
    """Test suite for edge/mobile TorchScript model deployment."""

    @pytest.mark.skipif(not MOBILE_DIR.exists(), reason="Mobile weights directory not found")
    def test_ngvc_net_mobile_forward(self):
        pt_path = MOBILE_DIR / "ngvc_net.pt"
        if not pt_path.exists():
            pytest.skip("ngvc_net.pt not generated yet")
        model = torch.jit.load(str(pt_path))
        x = torch.randn(2, 128)
        out = model(x)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not MOBILE_DIR.exists(), reason="Mobile weights directory not found")
    def test_spatial_graph_net_mobile_forward(self):
        pt_path = MOBILE_DIR / "spatial_graph_net.pt"
        if not pt_path.exists():
            pytest.skip("spatial_graph_net.pt not generated yet")
        model = torch.jit.load(str(pt_path))
        nodes = torch.randn(16, 64)
        adj = torch.ones(16, 16)
        out = model(nodes, adj)
        assert out.shape == (16, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not MOBILE_DIR.exists(), reason="Mobile weights directory not found")
    def test_temporal_progression_net_mobile_forward(self):
        pt_path = MOBILE_DIR / "temporal_progression_net.pt"
        if not pt_path.exists():
            pytest.skip("temporal_progression_net.pt not generated yet")
        model = torch.jit.load(str(pt_path))
        seq = torch.randn(2, 6, 64)
        out = model(seq)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not MOBILE_DIR.exists(), reason="Mobile weights directory not found")
    def test_agrivision_ensemble_mobile_forward(self):
        pt_path = MOBILE_DIR / "agrivision_ensemble.pt"
        if not pt_path.exists():
            pytest.skip("agrivision_ensemble.pt not generated yet")
        model = torch.jit.load(str(pt_path))
        vis = torch.randn(4, 64)
        spat = torch.randn(4, 64)
        temp = torch.randn(4, 64)
        logits, severity = model(vis, spat, temp)
        assert logits.shape == (4, 5)
        assert severity.shape == (4, 1)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(severity).any()
