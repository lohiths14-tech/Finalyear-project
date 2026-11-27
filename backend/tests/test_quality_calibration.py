import numpy as np

from app.quality.calibration import (
    ConfidenceCalibrator,
    IsotonicCalibrator,
    OODDetector,
    ReliabilityScorer,
    TemperatureScaler,
)


def test_temperature_scaler():
    scaler = TemperatureScaler(temperature=2.0)
    logits = np.array([[2.0, 1.0, 0.1]])

    # Unscaled softmax
    scaled_1 = TemperatureScaler(1.0).calibrate(logits)

    # Scaled softmax
    scaled_2 = scaler.calibrate(logits)

    # Higher temperature should soften the probabilities
    assert scaled_2.max() < scaled_1.max()

def test_isotonic_calibrator():
    calibrator = IsotonicCalibrator()
    # High confidence for correct, low for incorrect
    raw_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])
    labels = np.array([0, 0, 1, 1])

    calibrator.fit(raw_probs, labels, num_bins=5)
    assert calibrator.fitted

    calibrated = calibrator.calibrate(raw_probs)
    # The output should remain a valid probability distribution
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert calibrated.shape == raw_probs.shape

def test_ood_detector():
    detector = OODDetector(msp_threshold=0.6, method="msp")

    # High confidence (In-Distribution)
    logits_id = np.array([5.0, 1.0, 0.5])
    is_ood_id, score_id, reason_id = detector.is_ood(logits_id)
    assert not is_ood_id

    # Uniform/Low confidence (OOD)
    logits_ood = np.array([1.0, 1.1, 1.0])
    is_ood, score, reason = detector.is_ood(logits_ood)
    assert is_ood
    assert reason.startswith("low_max_prob")

    # Test entropy method
    detector_entropy = OODDetector(entropy_threshold=1.0, method="entropy")
    is_ood_ent, _, _ = detector_entropy.is_ood(logits_ood)
    assert is_ood_ent

def test_reliability_scorer():
    scorer = ReliabilityScorer(
        confidence_threshold=0.45,
        quality_threshold=0.55,
        ood_threshold=0.50
    )

    # Good scenario
    score_good = scorer.score(
        calibrated_confidence=0.8,
        quality_iqi=0.8,
        ood_score=0.9
    )
    assert score_good.is_reliable
    assert score_good.rejection_reason is None

    # Bad quality scenario
    score_bad = scorer.score(
        calibrated_confidence=0.8,
        quality_iqi=0.4,
        ood_score=0.9
    )
    assert not score_bad.is_reliable
    assert score_bad.rejection_reason == "poor_quality"

def test_confidence_calibrator():
    calibrator = ConfidenceCalibrator(min_calibration_samples=5)
    logits = np.random.randn(10, 3)
    labels = np.random.randint(0, 3, size=10)

    # Should fit both temperature and isotonic because n > min_samples
    calibrator.fit(logits, labels)
    assert calibrator.fitted_temperature > 0

    result = calibrator.calibrate(np.array([[2.0, 0.5, -1.0]]))
    assert result.calibrated_probs.shape == (1, 3)
    assert result.method in ["hybrid", "temperature"]
