import numpy as np

from app.quality.image_quality import (
    FarmerFeedbackGenerator,
    ImageQualityAssessor,
    QualityScores,
)


def test_quality_scores():
    qs = QualityScores(
        blur=0.8,
        brightness=0.7,
        noise=0.9,
        contrast=0.8,
        resolution=1.0,
        shadow_ratio=0.7,
        vegetation_coverage=0.6
    )
    assert qs.is_acceptable
    assert 0.0 < qs.overall <= 1.0

def test_image_quality_assessor():
    assessor = ImageQualityAssessor(field_mode=False)
    # Create a dummy image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    scores = assessor.assess(image)
    assert isinstance(scores, QualityScores)

    # Random noise usually scores low on noise factor but high on contrast
    assert 0.0 <= scores.overall <= 1.0

def test_farmer_feedback_generator():
    scores = QualityScores(
        blur=0.2, # blurry
        brightness=0.4,
        noise=0.4,
        contrast=0.4,
        resolution=0.4,
        shadow_ratio=0.4,
        vegetation_coverage=0.4
    )

    fb = FarmerFeedbackGenerator.generate_quality_feedback(scores)
    assert "blurry" in fb["issues"]
    assert not fb["proceed_to_inference"]

    comb_fb = FarmerFeedbackGenerator.combine_feedback(
        quality=scores.to_dict(),
        inference_confidence=90.0,
        disease_label="Rice Blast"
    )

    assert "blurry" in comb_fb["quality"]["issues"]
    assert comb_fb["confidence"]["level"] == "high"
