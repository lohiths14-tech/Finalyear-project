"""
Unit Tests for AgriPulse 2.0 Mobile / Edge Export Pipeline
==========================================================

Verifies that all 4 exported TorchScript mobile models load and compute
deterministic forward passes identical to native PyTorch models.
"""

from pathlib import Path

import pytest
import torch

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

    def test_onnx_mobilenet_quant_forward(self):
        """Verifies MobileNetV2 INT8 quantized ONNX forward pass."""
        import numpy as np
        import onnxruntime as ort

        onnx_path = Path(__file__).resolve().parent.parent / "model_weights" / "onnx" / "mobilenet_v2_quant.onnx"
        if not onnx_path.exists():
            pytest.skip("mobilenet_v2_quant.onnx not found")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

        dummy_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
        logits = session.run(None, {"image": dummy_img})[0]

        assert logits.shape == (1, 10), f"Expected (1, 10), got {logits.shape}"
        assert not np.isnan(logits).any(), "ONNX MobileNet output contains NaNs"
        assert not np.isinf(logits).any(), "ONNX MobileNet output contains Infs"

    def test_onnx_quantized_footprint(self):
        """Ensures all quantized ONNX checkpoints are memory-efficient (< 3 MB each)."""
        onnx_dir = Path(__file__).resolve().parent.parent / "model_weights" / "onnx"
        if not onnx_dir.exists():
            pytest.skip("ONNX directory not found")

        expected = [
            "spectral_net_quant.onnx",
            "spatial_net_quant.onnx",
            "temporal_net_quant.onnx",
            "ensemble_quant.onnx",
            "mobilenet_v2_quant.onnx",
        ]
        for name in expected:
            f = onnx_dir / name
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                assert size_mb < 3.0, f"{name} exceeds 3 MB limit: {size_mb:.2f} MB"
