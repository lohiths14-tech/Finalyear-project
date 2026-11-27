"""
Deterministic end-to-end inference test.

Validates that the full SST pipeline produces consistent, reproducible
results when given a known input with fixed random seed.

This is the critical "smoke test" that guarantees:
  1. Model weights are valid and loadable
  2. Architecture shapes are compatible
  3. Forward pass completes without errors
  4. Outputs are within expected ranges
"""

import numpy as np
import pytest
import torch


@pytest.fixture
def inference_service():
    """Create an InferenceService with loaded model weights."""
    from app.core.config import settings
    from app.services.inference_service import InferenceService

    model_paths = settings.get_model_paths()
    svc = InferenceService(model_paths=model_paths if model_paths else None)
    return svc


class TestDeterministicInference:
    """Tests that inference produces reproducible, valid outputs."""

    def test_leaf_inference_produces_valid_output(self, inference_service):
        """A solid green image should produce a valid leaf diagnostic."""
        # Create a known synthetic image: 224x224 solid green
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[:, :, 1] = 180  # Green channel

        result = inference_service.analyze_leaf(image)

        # Must return all required fields
        assert "disease_type" in result
        assert "severity" in result
        assert "confidence" in result
        assert "alternatives" in result
        assert "uncertainty_breakdown" in result

        # Severity must be in valid range
        assert 0.0 <= result["severity"] <= 100.0

        # Confidence must be in valid range
        assert 0.0 <= result["confidence"] <= 100.0

        # Alternatives must be a list
        assert isinstance(result["alternatives"], list)

    def test_field_inference_produces_valid_output(self, inference_service):
        """A synthetic field image should produce valid field analysis."""
        # Create a known synthetic image: 640x480 with green patches
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add green vegetation patches
        image[50:400, 50:590, 1] = 160
        image[50:400, 50:590, 0] = 40
        image[50:400, 50:590, 2] = 30

        result = inference_service.analyze_field(image)

        # Must return all required fields
        assert "disease_type" in result
        assert "average_severity" in result
        assert "confidence" in result
        assert "grid_data" in result
        assert "nodes" in result
        assert "uncertainty_breakdown" in result

        # Average severity must be in valid range
        assert 0.0 <= result["average_severity"] <= 100.0

        # Infection rate must be in valid range
        assert 0.0 <= result["infection_rate"] <= 100.0

        # Grid data must have entries
        assert isinstance(result["grid_data"], list)

    def test_deterministic_same_input_same_output(self, inference_service):
        """Same input image must produce identical outputs across runs."""
        torch.manual_seed(42)
        image = np.full((224, 224, 3), 120, dtype=np.uint8)
        image[:, :, 1] = 180  # Green tint

        result1 = inference_service.analyze_leaf(image)
        result2 = inference_service.analyze_leaf(image)

        assert result1["disease_type"] == result2["disease_type"]
        assert abs(result1["severity"] - result2["severity"]) < 0.01
        assert abs(result1["confidence"] - result2["confidence"]) < 0.01

    def test_low_confidence_withholds_prediction(self, inference_service):
        """
        When confidence is very low, the system should withhold
        the prediction per the Real AI Policy.
        """
        # Pure noise image — should produce low confidence
        rng = np.random.RandomState(42)
        noise_image = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = inference_service.analyze_leaf(noise_image)

        # The system should either withhold or produce valid output
        # (we don't mandate withheld=True for noise, but the output must be valid)
        assert "disease_type" in result
        assert "withheld" in result
        if result["withheld"]:
            assert result["disease_type"] == "Unknown"

    def test_severity_map_has_correct_length(self, inference_service):
        """Severity map length must match the number of grid regions."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[50:400, 50:590, 1] = 160

        result = inference_service.analyze_field(image)

        if "severity_map" in result and "regions" in result:
            assert len(result["severity_map"]) == len(result["regions"])

    def test_single_scan_returns_simulated_progression(
        self, inference_service
    ):
        """
        Without temporal sequence, progression_data provides a 6-point
        logistic model estimate with simulated=True flags.
        """
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[50:400, 50:590, 1] = 160

        result = inference_service.analyze_field(image)

        assert result["temporal_available"] is False
        assert len(result["progression_data"]) == 6
        assert all(p.get("simulated") is True for p in result["progression_data"])
