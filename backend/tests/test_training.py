import numpy as np

from app.training.active_learning import (
    ActiveLearningLoop,
    PseudoLabeler,
    SampleCandidate,
    SampleSelector,
)

# Mock augmentation just to satisfy test requirements if it's there
try:
    from app.training.augmentation import CompositeAugmenter
    _ = CompositeAugmenter
except ImportError:
    pass

def test_sample_selector_entropy():
    selector = SampleSelector(strategy="entropy")
    candidates = [
        SampleCandidate(ngvc=np.zeros(128), image_id="1", tile_id=0, entropy=1.0, margin=0.1),
        SampleCandidate(ngvc=np.zeros(128), image_id="2", tile_id=0, entropy=0.5, margin=0.5)
    ]

    selected = selector.select(candidates, budget=1)
    assert len(selected) == 1
    assert selected[0].image_id == "1"

def test_pseudo_labeler():
    labeler = PseudoLabeler(confidence_threshold=0.85, entropy_threshold=0.4)
    # High confidence (margin > 0.85 => 1 - margin > 0.85 -> margin < 0.15 ? wait, 1.0 - margin > conf)
    # Actually, 1.0 - margin >= conf => margin <= 1.0 - conf
    # E.g., confidence_threshold = 0.85 => margin <= 0.15
    candidates = [
        SampleCandidate(ngvc=np.zeros(128), image_id="1", tile_id=0, entropy=0.3, margin=0.1, pseudo_label=1, pseudo_confidence=0.9),
        SampleCandidate(ngvc=np.zeros(128), image_id="2", tile_id=0, entropy=0.9, margin=0.5, pseudo_label=2, pseudo_confidence=0.6)
    ]

    accepted = labeler.generate_pseudo_labels(candidates)
    assert len(accepted) == 1
    assert accepted[0].image_id == "1"
    assert accepted[0].source == "pseudo"

def test_active_learning_loop():
    loop = ActiveLearningLoop(model_fn=lambda: None, get_dataloader_fn=lambda: None)

    ngvc = np.random.rand(128)
    logits = np.array([2.0, -1.0, -1.0])

    loop.add_inference_result(ngvc, logits, "img_1", 0)
    assert loop.stats["total_candidates"] == 1

    selected = loop.select_samples(budget=1)
    assert len(selected) == 1
    assert loop.stats["selected_for_labeling"] == 1
