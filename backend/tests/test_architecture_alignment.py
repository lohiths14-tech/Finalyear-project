"""
Architecture alignment tests.

Validates that:
  1. Training wrapper can be instantiated in all 4 modes
  2. Forward pass succeeds for each mode
  3. All architecture classes can be independently instantiated
  4. Inference architectures match expected input/output shapes
"""

import pytest
import torch


class TestTrainingArchitectures:
    """Tests for training-time teacher architectures."""

    def test_foundation_vision_encoder_instantiates(self):
        """FoundationVisionEncoder should instantiate (with fallback)."""
        from app.models.architectures import FoundationVisionEncoder

        encoder = FoundationVisionEncoder(feature_dim=64)
        assert encoder is not None

    def test_graph_transformer_forward(self):
        """GraphTransformer should process node features with adjacency."""
        from app.models.architectures import GraphTransformer

        gt = GraphTransformer(in_channels=64, out_channels=64, num_heads=4)
        x = torch.randn(5, 64)
        adj = torch.eye(5)
        adj[0, 1] = 1
        adj[1, 0] = 1

        out = gt(x, adj)
        assert out.shape == (5, 64)

    def test_temporal_transformer_forward(self):
        """TemporalTransformer should process sequences."""
        from app.models.architectures import TemporalTransformer

        tt = TemporalTransformer(input_dim=64, num_heads=4, num_layers=2)
        x = torch.randn(2, 5, 64)  # batch=2, seq_len=5

        out = tt(x)
        assert out.shape == (2, 64)

    def test_cross_attention_fusion_forward(self):
        """CrossAttentionFusion should fuse three modalities."""
        from app.models.architectures import CrossAttentionFusion

        fusion = CrossAttentionFusion(
            feature_dim=64, num_classes=5, num_heads=4
        )
        visual = torch.randn(2, 64)
        spatial = torch.randn(2, 64)
        temporal = torch.randn(2, 64)

        logits, severity = fusion(visual, spatial, temporal)
        assert logits.shape == (2, 5)
        assert severity.shape == (2, 1)
        assert (severity >= 0).all() and (severity <= 100).all()


class TestTrainingWrapper:
    """Tests for AgriPulseTrainingWrapper in all 4 ablation modes."""

    @pytest.fixture
    def dummy_input(self):
        """2D cached feature input (batch=4, feature_dim=64)."""
        return torch.randn(4, 64)

    def test_baseline_mode(self, dummy_input):
        """Baseline mode: vision features → linear heads."""
        from app.training.model_wrapper import AgriPulseTrainingWrapper

        model = AgriPulseTrainingWrapper(
            feature_dim=64, num_classes=5, mode="baseline"
        )
        model.eval()
        logits, severity = model(dummy_input, is_cached=True)
        assert logits.shape == (4, 5)
        assert severity.shape == (4, 1)

    def test_graph_only_mode(self, dummy_input):
        """Graph-only mode: vision + spatial → linear heads."""
        from app.training.model_wrapper import AgriPulseTrainingWrapper

        model = AgriPulseTrainingWrapper(
            feature_dim=64, num_classes=5, mode="graph_only"
        )
        model.eval()
        logits, severity = model(dummy_input, is_cached=True)
        assert logits.shape == (4, 5)

    def test_temporal_only_mode(self, dummy_input):
        """Temporal-only mode: vision + temporal → linear heads."""
        from app.training.model_wrapper import AgriPulseTrainingWrapper

        model = AgriPulseTrainingWrapper(
            feature_dim=64, num_classes=5, mode="temporal_only"
        )
        model.eval()
        logits, severity = model(dummy_input, is_cached=True)
        assert logits.shape == (4, 5)

    def test_full_mode(self, dummy_input):
        """Full mode: vision + spatial + temporal + cross-attention fusion."""
        from app.training.model_wrapper import AgriPulseTrainingWrapper

        model = AgriPulseTrainingWrapper(
            feature_dim=64, num_classes=5, mode="full"
        )
        model.eval()
        logits, severity = model(dummy_input, is_cached=True)
        assert logits.shape == (4, 5)
        assert severity.shape == (4, 1)


class TestInferenceArchitectures:
    """Tests for inference-time student architectures."""

    def test_ngvcnet_forward(self):
        """NGVCNet should embed spectral histograms."""
        from app.models.architectures import NGVCNet

        net = NGVCNet(input_dim=128, feature_dim=64)
        x = torch.randn(3, 128)
        out = net(x)
        assert out.shape == (3, 64)

    def test_spatial_graph_net_forward(self):
        """SpatialGraphNet should aggregate spatial features."""
        from app.models.architectures import SpatialGraphNet

        net = SpatialGraphNet(in_channels=64, out_channels=64)
        x = torch.randn(5, 64)
        adj = torch.eye(5)
        out = net(x, adj)
        assert out.shape == (5, 64)

    def test_temporal_progression_net_forward(self):
        """TemporalProgressionNet should summarize sequences."""
        from app.models.architectures import TemporalProgressionNet

        net = TemporalProgressionNet(input_dim=64, hidden_dim=128)
        x = torch.randn(2, 5, 64)
        out = net(x)
        assert out.shape == (2, 64)

    def test_agrivision_ensemble_forward(self):
        """AgriVisionEnsemble should fuse and classify."""
        from app.models.architectures import AgriVisionEnsemble

        net = AgriVisionEnsemble(feature_dim=64, num_classes=5)
        spectral = torch.randn(2, 64)
        spatial = torch.randn(2, 64)
        temporal = torch.randn(2, 64)
        logits, severity = net(spectral, spatial, temporal)
        assert logits.shape == (2, 5)
        assert severity.shape == (2, 1)

    def test_weights_load_successfully(self):
        """Pre-trained weights should load without errors."""
        from app.core.config import settings
        from app.models.architectures import (
            AgriVisionEnsemble,
            NGVCNet,
            SpatialGraphNet,
            TemporalProgressionNet,
        )

        model_paths = settings.get_model_paths()
        if not model_paths:
            pytest.skip("No model weights available")

        classes_path = settings.resolve_model_base_dir() / "classes.json"
        num_classes = 5
        if classes_path.exists():
            import json
            with open(classes_path, encoding="utf-8") as f:
                data = json.load(f)
                num_classes = len(data.get("classes", [])) or 5

        models = {
            "spectral": NGVCNet(input_dim=128, feature_dim=64),
            "spatial": SpatialGraphNet(in_channels=64, out_channels=64),
            "temporal": TemporalProgressionNet(input_dim=64),
            "ensemble": AgriVisionEnsemble(
                feature_dim=64, num_classes=num_classes
            ),
        }

        for key, model in models.items():
            path = model_paths.get(key)
            if path:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
