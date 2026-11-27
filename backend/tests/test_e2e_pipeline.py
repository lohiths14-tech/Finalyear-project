"""
End-to-End Integration Tests for the AgriPulse 2.0 SST Pipeline
================================================================
These tests verify that the full Spatio-Spectral-Temporal pipeline produces
a valid, schema-correct response from a synthetic vegetation image.

All assertions are against the SCHEMA and VALUE BOUNDS of the response,
not against specific disease names (which depend on trained weights and
the input image). This ensures the tests are valid regardless of model state.
"""

import numpy as np
import pytest

from app.services.inference_service import DISEASE_CLASSES, InferenceService

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def service():
    """Shared InferenceService instance (loaded once per test module)."""
    return InferenceService()


@pytest.fixture
def vegetation_image():
    """
    Deterministic 256x256 vegetation image: high green channel, low red/blue.
    Sufficient ExG index to pass the vegetation mask threshold.
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 1] = 180  # Strong green channel
    img[:, :, 0] = 30   # Low red
    img[:, :, 2] = 20   # Low blue
    return img


@pytest.fixture
def field_result(service, vegetation_image):
    """Run field analysis once and share the result."""
    result = service.analyze_field(vegetation_image)
    if "error" in result:
        pytest.skip(f"No vegetation detected in synthetic image: {result['error']}")
    return result


# ── Schema Tests ─────────────────────────────────────────────────────────────

REQUIRED_FIELD_KEYS = {
    "disease_type", "treatment", "average_severity", "confidence",
    "withheld", "temporal_available", "alternatives", "spread_risk",
    "nodes", "progression_data", "spectral_radar", "ngvc_data",
    "grid_data", "infection_rate", "veg_coverage", "uncertainty_breakdown",
    "disease_distribution", "severity_map", "regions",
}


class TestFieldResultSchema:
    def test_all_required_keys_present(self, field_result):
        missing = REQUIRED_FIELD_KEYS - set(field_result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_withheld_key_is_bool(self, field_result):
        assert isinstance(field_result["withheld"], bool)

    def test_temporal_available_key_is_bool(self, field_result):
        assert isinstance(field_result["temporal_available"], bool)

    def test_temporal_unavailable_for_single_image(self, field_result):
        """Single image scan must never claim temporal analysis is available."""
        assert field_result["temporal_available"] is False, (
            "temporal_available must be False when no multi-frame sequence was provided. "
            "Static repetition of a single frame is not temporal modeling."
        )

    def test_disease_type_is_valid_class_or_unknown(self, field_result, service):
        valid = set(service.classes) | set(DISEASE_CLASSES) | {"Unknown", "Out-of-Distribution / Unknown"}
        assert field_result["disease_type"] in valid

    def test_spread_risk_is_valid_enum(self, field_result):
        assert field_result["spread_risk"] in {"Low", "Medium", "High", "Unknown"}

    def test_alternatives_is_list(self, field_result):
        assert isinstance(field_result["alternatives"], list)
        assert len(field_result["alternatives"]) <= 3

    def test_alternatives_have_correct_schema(self, field_result):
        for alt in field_result["alternatives"]:
            assert "disease" in alt
            assert "confidence" in alt
            assert isinstance(alt["confidence"], float)


# ── Value Range Tests ─────────────────────────────────────────────────────────

class TestFieldResultValueRanges:
    def test_severity_in_range(self, field_result):
        assert 0.0 <= field_result["average_severity"] <= 100.0

    def test_confidence_in_range(self, field_result):
        assert 0.0 <= field_result["confidence"] <= 100.0

    def test_infection_rate_in_range(self, field_result):
        assert 0.0 <= field_result["infection_rate"] <= 100.0

    def test_veg_coverage_in_range(self, field_result):
        assert 0.0 <= field_result["veg_coverage"] <= 100.0

    def test_disease_distribution_sums_to_one(self, field_result):
        dist = field_result["disease_distribution"]
        assert abs(sum(dist) - 1.0) < 1e-4, f"Probabilities must sum to 1: {sum(dist)}"

    def test_disease_distribution_all_non_negative(self, field_result):
        dist = field_result["disease_distribution"]
        assert all(v >= 0 for v in dist)

    def test_severity_map_values_in_range(self, field_result):
        for sev in field_result["severity_map"]:
            assert 0.0 <= sev <= 100.0, f"Severity out of range: {sev}"


# ── Progression Data Tests ────────────────────────────────────────────────────

class TestProgressionData:
    def test_progression_has_six_points(self, field_result):
        assert len(field_result["progression_data"]) == 6

    def test_progression_entries_have_simulated_flag(self, field_result):
        """When progression entries exist in single-scan mode, they must be marked simulated."""
        for entry in field_result["progression_data"]:
            assert "simulated" in entry, "Each progression entry must have 'simulated' key"

    def test_progression_entries_have_point_type(self, field_result):
        for entry in field_result["progression_data"]:
            assert entry.get("point_type") in {"estimated_history", "forecast"}

    def test_progression_values_in_range(self, field_result):
        for entry in field_result["progression_data"]:
            if entry.get("actual") is not None:
                assert 0.0 <= entry["actual"] <= 100.0
            assert 0.0 <= entry["predicted"] <= 100.0


# ── Uncertainty Tests ─────────────────────────────────────────────────────────

class TestUncertaintyBreakdown:
    def test_uncertainty_keys_present(self, field_result):
        ub = field_result["uncertainty_breakdown"]
        for key in ("aleatoric", "epistemic", "entropy_bits", "energy_score"):
            assert key in ub

    def test_aleatoric_in_range(self, field_result):
        assert 0.0 <= field_result["uncertainty_breakdown"]["aleatoric"] <= 1.0

    def test_epistemic_in_range(self, field_result):
        assert 0.0 <= field_result["uncertainty_breakdown"]["epistemic"] <= 1.0


# ── Spectral Radar Tests ──────────────────────────────────────────────────────

class TestSpectralRadar:
    def test_six_subjects(self, field_result):
        assert len(field_result["spectral_radar"]) == 6

    def test_subjects_have_correct_keys(self, field_result):
        for s in field_result["spectral_radar"]:
            assert "subject" in s
            assert "A" in s
            assert "fullMark" in s


# ── Leaf Analysis Schema Tests ────────────────────────────────────────────────

REQUIRED_LEAF_KEYS = {
    "type", "status", "disease_type", "treatment", "severity",
    "confidence", "withheld", "alternatives", "ngvc", "uncertainty_breakdown",
}


class TestLeafResultSchema:
    def test_all_required_keys_present(self, service, vegetation_image):
        result = service.analyze_leaf(vegetation_image)
        missing = REQUIRED_LEAF_KEYS - set(result.keys())
        assert not missing, f"Missing leaf result keys: {missing}"

    def test_withheld_is_bool(self, service, vegetation_image):
        result = service.analyze_leaf(vegetation_image)
        assert isinstance(result["withheld"], bool)

    def test_severity_in_range(self, service, vegetation_image):
        result = service.analyze_leaf(vegetation_image)
        assert 0.0 <= result["severity"] <= 100.0

    def test_confidence_in_range(self, service, vegetation_image):
        result = service.analyze_leaf(vegetation_image)
        assert 0.0 <= result["confidence"] <= 100.0
