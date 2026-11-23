"""
Temporal Honesty Tests — Real AI Policy Compliance
===================================================
These tests specifically verify that the AgriPulse temporal analysis branch
obeys the Real AI Policy:

  1. No single-frame temporal fabrication: when temporal_sequence is None or
     has only 1 frame, temporal_available must be False and the LSTM must NOT
     be invoked with a repeated static frame.
  2. Real multi-frame sequences correctly enable temporal analysis.
  3. The temporal_available flag propagates correctly in the API response.

These tests are separate from test_inference.py to make policy violations
immediately obvious to code reviewers.
"""

import numpy as np
import pytest

from app.services.inference_service import InferenceService


def make_green_frame(h: int = 128, w: int = 128) -> np.ndarray:
    """Create a deterministic vegetation frame (high ExG index)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = 180  # Strong green
    img[:, :, 0] = 25
    img[:, :, 2] = 15
    return img


@pytest.fixture(scope="module")
def service():
    return InferenceService()


@pytest.fixture
def green_frame():
    return make_green_frame()


# ── Core Policy: No Temporal Fabrication ─────────────────────────────────────

class TestTemporalHonestyPolicy:
    def test_no_sequence_returns_temporal_unavailable(self, service, green_frame):
        """
        POLICY: When no temporal_sequence is provided (single scan),
        temporal_available must be False. The system must not fabricate
        temporal dynamics by repeating the single frame.
        """
        result = service.analyze_field(green_frame, temporal_sequence=None)
        if "error" in result:
            pytest.skip("No vegetation detected in synthetic frame")
        assert "temporal_available" in result, (
            "temporal_available key missing from response"
        )
        assert result["temporal_available"] is False, (
            "POLICY VIOLATION: temporal_available is True with no real scan history. "
            "Static frame repetition is not temporal analysis."
        )

    def test_single_frame_sequence_returns_temporal_unavailable(self, service, green_frame):
        """
        POLICY: A sequence of exactly 1 frame is NOT a temporal sequence.
        temporal_available must be False.
        """
        result = service.analyze_field(green_frame, temporal_sequence=[green_frame])
        if "error" in result:
            pytest.skip("No vegetation detected")
        assert result["temporal_available"] is False, (
            "POLICY VIOLATION: A single-frame list is not a real temporal sequence. "
            "temporal_available must be False."
        )

    def test_multi_frame_sequence_enables_temporal(self, service, green_frame):
        """
        When 3 distinct frames are provided, temporal analysis is enabled
        and temporal_available must be True.
        NOTE: This test may only pass when temporal weights are loaded.
        """
        # Create 3 slightly different frames to simulate a real sequence
        frames = [
            make_green_frame(),  # Frame 1
            make_green_frame(),  # Frame 2
            make_green_frame(),  # Frame 3
        ]
        # Slightly vary each frame to simulate real multi-day imagery
        frames[1][:, :, 1] = 160  # Slightly less green (disease progression)
        frames[2][:, :, 1] = 140  # More degraded

        result = service.analyze_field(green_frame, temporal_sequence=frames)
        if "error" in result:
            pytest.skip("No vegetation detected")

        # Only assert True if temporal model weights are loaded
        assert "temporal_available" in result, "temporal_available key must always be present"
        if "temporal" in service.loaded_models:
            assert result["temporal_available"] is True, (
                "With 3 real frames and loaded temporal weights, temporal_available should be True"
            )

    def test_temporal_available_is_always_in_response(self, service, green_frame):
        """temporal_available must always be in the response, never omitted."""
        for seq in [None, [], [green_frame]]:
            result = service.analyze_field(green_frame, temporal_sequence=seq)
            if "error" not in result:
                assert "temporal_available" in result


# ── Simulated Progression Flags ───────────────────────────────────────────────

class TestProgressionHonesty:
    def test_progression_data_flags_simulated(self, service, green_frame):
        """
        POLICY: All progression points in single-scan mode must carry
        simulated=True so the UI can show a disclaimer.
        """
        result = service.analyze_field(green_frame)
        if "error" in result:
            pytest.skip("No vegetation detected")
        for i, entry in enumerate(result["progression_data"]):
            assert entry.get("simulated") is True, (
                f"Progression point {i} missing simulated=True flag. "
                "All single-scan progression points are synthetic estimates."
            )

    def test_progression_point_type_is_labeled(self, service, green_frame):
        """Each progression entry must declare its point_type."""
        result = service.analyze_field(green_frame)
        if "error" in result:
            pytest.skip("No vegetation detected")
        valid_types = {"estimated_history", "forecast"}
        for entry in result["progression_data"]:
            assert entry.get("point_type") in valid_types, (
                f"Invalid point_type: {entry.get('point_type')}"
            )


# ── Confidence Withholding ────────────────────────────────────────────────────

class TestConfidenceWithholding:
    def test_withheld_key_always_present_in_field_result(self, service, green_frame):
        """withheld must always be in the response."""
        result = service.analyze_field(green_frame)
        if "error" not in result:
            assert "withheld" in result

    def test_withheld_key_always_present_in_leaf_result(self, service, green_frame):
        """withheld must always be in the leaf response."""
        result = service.analyze_leaf(green_frame)
        assert "withheld" in result

    def test_withheld_is_bool(self, service, green_frame):
        result = service.analyze_field(green_frame)
        if "error" not in result:
            assert isinstance(result["withheld"], bool)

    def test_unknown_disease_when_withheld(self, service, green_frame):
        """When withheld=True, disease_type must be Unknown."""
        result = service.analyze_field(green_frame)
        if "error" not in result and result["withheld"]:
            assert result["disease_type"] == "Unknown", (
                "When prediction is withheld (low confidence), disease_type must be Unknown"
            )
