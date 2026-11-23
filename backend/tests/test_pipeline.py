import pytest
import torch
import os
import sys

# Ensure backend root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.architectures import FoundationVisionEncoder, GraphTransformer, TemporalTransformer, CrossAttentionFusion
from app.training.model_wrapper import AgriPulseTrainingWrapper
from app.training.dataset import AgriPulseDataset

@pytest.fixture
def device():
    return torch.device("cpu")

def test_vision_encoder(device):
    model = FoundationVisionEncoder(feature_dim=64).to(device)
    # Dummy image batch (2, 3, 224, 224)
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    out = model(dummy_input)
    
    assert out.shape == (2, 64), "Vision encoder output shape mismatch."

def test_graph_transformer(device):
    model = GraphTransformer(in_channels=64, out_channels=64, num_heads=4).to(device)
    # 5 nodes, feature dim 64
    x = torch.randn(5, 64, device=device)
    # Dummy adjacency (identity + some connections)
    adj = torch.eye(5, device=device)
    
    out = model(x, adj)
    assert out.shape == (5, 64), "Graph transformer output shape mismatch."

def test_temporal_transformer(device):
    model = TemporalTransformer(input_dim=64, num_heads=4, num_layers=2).to(device)
    # Batch 2, sequence length 3, feature dim 64
    x = torch.randn(2, 3, 64, device=device)
    
    out = model(x)
    assert out.shape == (2, 64), "Temporal transformer output shape mismatch."

def test_training_wrapper_forward(device):
    model = AgriPulseTrainingWrapper(feature_dim=64, num_classes=5).to(device)
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    
    logits, severity = model(dummy_input)
    assert logits.shape == (2, 5), "Wrapper classification logits shape mismatch."
    assert severity.shape == (2, 1), "Wrapper severity output shape mismatch."

def test_dataset_initialization():
    dummy_dirs = {
        "paddy_doctor": "/tmp/nonexistent/paddy",
    }
    dataset = AgriPulseDataset(data_dirs=dummy_dirs, is_train=True)
    # Dataset should initialize gracefully without crashing even if folder missing
    assert hasattr(dataset, 'samples'), "Dataset did not initialize samples list."
    assert isinstance(dataset.samples, list), "Samples is not a list."
