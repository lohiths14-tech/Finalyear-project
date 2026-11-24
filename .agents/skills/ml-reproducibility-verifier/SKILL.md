---
name: ml-reproducibility-verifier
description: >-
  Verifies machine learning inference pipelines against the strict zero-dummy-data policy (AGENTS.md).
  Audits confidence calibration, out-of-distribution (OOD) detection, regression heads, and field spatial analysis.
---

# ML Reproducibility & Scientific Verification

Use this skill to audit machine learning models, inference pipelines, and diagnostic algorithms in AgriPulse 2.0.

## Non-Negotiable Real AI Policy
As defined in `AGENTS.md`:
1. **No dummy data**: No `np.random` or `torch.randn` simulation for predictions.
2. **No placeholder models**: Every model referenced must execute a genuine forward pass.
3. **Calibrated Confidence**: Confidence scores must originate from isotonic/temperature calibration.
4. **OOD Rejection**: Out-of-Distribution images must be detected and rejected before classification.
5. **Accurate Terminology**: Field analysis must be described as **adaptive grid-based field analysis** (dense patch reasoning), NOT discrete object detection.

## Verification Checklist
- Run inference on the 4 benchmark samples:
  - `sample_healthy.jpg` -> Confirms baseline negative control.
  - `sample_blast.jpg` -> Validates severe fungal lesion recognition.
  - `sample_brown_spot.jpg` -> Validates discrete fungal pathology.
  - `sample_ood_trap.jpg` -> Validates OOD rejection.
- Check calibration output: ECE and Brier score must be within documented thresholds.
- Verify latency: High-Performance CPU inference must complete within expected runtime bounds.
