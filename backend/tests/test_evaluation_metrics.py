import numpy as np
import pytest
from app.evaluation.metrics import (
    BinaryMetrics,
    ModelEvaluator,
    _dice_coefficient,
    _boundary_f1
)

def test_binary_metrics():
    bm = BinaryMetrics(tp=50, tn=40, fp=10, fn=5)
    assert bm.precision == 50 / 60
    assert bm.recall == 50 / 55
    assert bm.specificity == 40 / 50
    assert bm.accuracy == 90 / 105
    assert bm.iou == 50 / 65

def test_segmentation_helpers():
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[1, 0], [0, 0]])
    # Intersection = 1, sum(pred)=2, sum(gt)=1
    assert _dice_coefficient(pred, gt) == pytest.approx((2 * 1) / (2 + 1))
    
    # Boundary F1 and Hausdorff
    b_f1 = _boundary_f1(pred, gt, dilation=1)
    assert b_f1 > 0.0

def test_model_evaluator_classification():
    evaluator = ModelEvaluator(classes=["A", "B"])
    evaluator.register_classification(0, 0, 10.0, 10.0, np.array([0.9, 0.1]))
    evaluator.register_classification(1, 0, 10.0, 10.0, np.array([0.4, 0.6]))
    evaluator.register_classification(1, 1, 15.0, 15.0, np.array([0.2, 0.8]))
    
    cls_metrics = evaluator.compute_classification()
    assert cls_metrics.accuracy == pytest.approx(2 / 3)
    assert "A" in cls_metrics.per_class
    assert cls_metrics.per_class["A"].support == 2

def test_model_evaluator_regression():
    evaluator = ModelEvaluator(classes=["A", "B"], disease_threshold=15.0)
    evaluator.register_classification(0, 0, 10.0, 12.0)
    evaluator.register_classification(1, 1, 20.0, 25.0)
    
    reg_metrics = evaluator.compute_severity_regression()
    assert reg_metrics.mae == pytest.approx(3.5)  # (2 + 5) / 2 = 3.5
    assert reg_metrics.rmse == pytest.approx(np.sqrt((4 + 25) / 2))
    assert reg_metrics.mae_infected_only == pytest.approx(5.0)

def test_model_evaluator_segmentation():
    evaluator = ModelEvaluator(classes=["A"])
    pred1 = np.array([[1, 1], [0, 0]])
    gt1 = np.array([[1, 0], [0, 0]])
    evaluator.register_segmentation(pred1, gt1, class_idx=0, confidence=0.9)
    
    pred2 = np.array([[1, 1], [1, 1]])
    gt2 = np.array([[1, 1], [1, 1]])
    evaluator.register_segmentation(pred2, gt2, class_idx=0, confidence=0.95)
    
    seg_metrics = evaluator.compute_segmentation()
    assert seg_metrics.iou_mean > 0.0
    assert seg_metrics.dice_mean > 0.0

def test_model_evaluator_full_report():
    evaluator = ModelEvaluator(classes=["A", "B"])
    evaluator.register_classification(0, 0, 10.0, 10.0)
    
    report = evaluator.full_report()
    assert "classification" in report
    assert "segmentation" in report
    assert "severity_regression" in report
    assert "spatial_consistency" in report
