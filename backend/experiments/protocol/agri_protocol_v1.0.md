# AgriVision 2.0 Complete Experimental Protocol (v1.0)

> [!IMPORTANT]
> **Protocol Freeze**: No edits to this experimental protocol are permitted after E1 begins, unless discovering a genuine bug. Changing success criteria, thresholds, or evaluation methodology after observing results invalidates the experiment.

## 0. Version History
| Version | Date | Changes |
| :--- | :--- | :--- |
| **v1.0** | Current | Initial protocol. Established evaluation framework, predefined targets, and ablation progression. |

## 1. Predefined Success & Evaluation Criteria
These thresholds and rules define what constitutes a successful, production-ready system *before* any training results are observed.

### 1.1 Primary vs Secondary Endpoints
- **Primary Endpoint**: **Test Macro F1** (Used for model selection and determining overall "best" architecture).
- **Secondary Endpoints**: ECE, Brier Score, Latency, OOD Detection Rate (Used to evaluate deployment viability).

### 1.2 Metric Targets
| Metric | Target | Rationale |
| :--- | :--- | :--- |
| **Macro F1** | $\ge$ 0.90 | Ensures balanced accuracy across minority disease classes. |
| **Expected Calibration Error (ECE)** | $\le$ 0.05 | Guarantees confidence scores actually reflect ground-truth probability. |
| **Brier Score** | $\le$ 0.10 | Low mean squared difference between predicted probabilities and labels. |
| **OOD Detection Rate** | $\ge$ 90% | Crucial for real-world robustness when encountering non-plant images. |
| **CPU Inference Latency** | $\le$ 300 ms/img | Required for on-device or lightweight server deployment. |
| **GPU Inference Latency** | $\le$ 50 ms/img | Required for high-throughput batch processing. |

---

### 1.3 Acceptable Regression & Deployment Rules
A model is only eligible for deployment if it meets the primary endpoint (Macro F1 target) AND the strict deployment constraints (Latencies).

**Final Deployment Selection**:
1. E4 (Full Architecture) is only selected if the absolute $\Delta$ in Macro F1 vs E1 is $\ge$ **0.5%**, AND the associated latency hit is $\le$ **25%**. 
2. If the performance gap is within this acceptable regression margin (e.g. +0.2% F1 but massive complexity), **the simpler architecture MUST be deployed.**
3. Calibration improvements (ECE/Brier) may justify the more complex model ONLY if all strict deployment targets ($\le$ 300ms CPU) are met.

### 1.4 Stopping Criteria
An experiment is officially concluded and logged when:
1. Early stopping is triggered (Validation Macro-F1 does not improve for 15 epochs).
2. Maximum epochs (50) are reached.
3. Training diverges (NaN loss) - *Requires immediate diagnosis.*
4. Hardware interruption - *Must be marked invalid and fully rerun from checkpoint or scratch.*

### 1.5 Dataset Inclusion & Exclusion Criteria
**Inclusion**: Minimum image resolution of 224x224, RGB only, standard formats (JPG/PNG), verified disease labels.
**Exclusion**: Corrupt images, duplicate files, unknown/unmapped labels. *(Handled automatically by `dataset.py` pipeline).*

---

## 2. Evaluation Protocol
The following protocol guarantees fair and reproducible comparisons.

| Item | Value |
| :--- | :--- |
| **Dataset Version** | *Pending Load* (e.g., v2.1) |
| **Train/Val/Test Split** | 80/10/10 (stratified) |
| **Random Seeds** | 42, 123, 999 |
| **Number of Runs** | 3 per experiment (if compute allows, else 1) |
| **Model Selection** | Best validation Macro F1 |
| **Test Set Usage** | Evaluated exactly once after model selection is locked |
| **Confidence Threshold** | Optimized exclusively on Validation set. (Not hardcoded to 0.5 without validation). |
| **Qualitative Archive** | A fixed set of N evaluation images must be saved with Input, Prediction, Confidence, Heatmap, and Ground Truth for visual E1-E4 comparison. |

---

## 3. Threats to Validity
| Threat | Mitigation Strategy |
| :--- | :--- |
| **Limited Field Diversity** | Use multiple public datasets (PlantDoc, Paddy Doctor, PlantVillage) and a separate real-field dataset. |
| **Class Imbalance** | Strict stratified split; evaluate primarily on Macro F1 rather than raw accuracy. |
| **Random Initialization** | Where computationally feasible, execute multiple runs using seeds `42`, `123`, `999`. |
| **Dataset Shift** | Specifically evaluate OOD detection capabilities and robustness on wild images. |
| **Hardware Dependence** | Report CPU and GPU metrics separately; reset CUDA memory peaks before tests. |

---

## 4. Chronological Experiment Log
| Date | Experiment | Outcome | Notes |
| :--- | :--- | :--- | :--- |
| *YYYY-MM-DD* | E1 Baseline | *Pending* | *Pending* |
| *YYYY-MM-DD* | E2 Spatial | *Pending* | *Pending* |
| *YYYY-MM-DD* | E3 Temporal | *Pending* | *Pending* |
| *YYYY-MM-DD* | E4 Full | *Pending* | *Pending* |

---

## 5. Experimental Results
*(Note: If multiple runs are performed, metrics will be reported as Mean $\pm$ Std. Otherwise, single run values are reported).*

### Training & Validation Metrics
| Experiment | Best Epoch | Train Loss | Val Loss | Test Accuracy | Test Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 Baseline** *(DINOv2 + Linear)* | - | - | - | - | - |
| **E2 Spatial** *(DINOv2 + Graph)* | - | - | - | - | - |
| **E3 Temporal** *(DINOv2 + Time)* | - | - | - | - | - |
| **E4 Full** *(Graph+Time+Fusion)* | - | - | - | - | - |

### Statistical Significance & Deltas (vs Baseline)
| Comparison | $\Delta$ Macro F1 | $\Delta$ CPU Latency | $\Delta$ Peak VRAM |
| :--- | :--- | :--- | :--- |
| **E2 − E1** | - | - | - |
| **E3 − E1** | - | - | - |
| **E4 − E1** | - | - | - |

---

## 6. Deployment Metrics & Efficiency

### Model Characteristics
| Experiment | Total Parameters | Trainable Parameters | ONNX Size |
| :--- | :--- | :--- | :--- |
| **E1 Baseline** | - | - | - |
| **E2 Spatial** | - | - | - |
| **E3 Temporal** | - | - | - |
| **E4 Full** | - | - | - |

### Runtime Performance
| Experiment | CPU Latency | GPU Latency | Peak RAM | Peak VRAM | Eval FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 Baseline** | - | - | - | - | - |
| **E2 Spatial** | - | - | - | - | - |
| **E3 Temporal** | - | - | - | - | - |
| **E4 Full** | - | - | - | - | - |

### Training Efficiency
| Experiment | Total Train Time | Time per Epoch | Train FPS |
| :--- | :--- | :--- | :--- |
| **E1 Baseline** | - | - | - |
| **E2 Spatial** | - | - | - |
| **E3 Temporal** | - | - | - |
| **E4 Full** | - | - | - |

---

## 7. Experiment Analysis & Failure Diagnosis

### E1: Baseline (DINOv2 + Linear Head)
#### Categorized Failure Causes
| Category | Count | Example |
| :--- | :--- | :--- |
| Poor illumination | 0 | - |
| Motion blur | 0 | - |
| Heavy occlusion | 0 | - |
| Early-stage disease | 0 | - |
| Similar disease symptoms | 0 | - |

- **Most Confused Pairs**: *Pending*
- **Lowest Recall Classes**: *Pending*
- **Architecture Decision**: *Establish Baseline.*

### E2: Spatial Ablation (DINOv2 + Graph Transformer)
- *(Failure Category Table Pending)*
- **Most Confused Pairs**: *Pending*
- **Lowest Recall Classes**: *Pending*
- **Observation**: *Pending (e.g., Macro F1 improved by X%, latency increased by Y%)*
- **Architecture Decision**: *Pending*

### E3: Temporal Ablation (DINOv2 + Temporal Transformer)
- *(Failure Category Table Pending)*
- **Most Confused Pairs**: *Pending*
- **Lowest Recall Classes**: *Pending*
- **Observation**: *Pending*
- **Architecture Decision**: *Pending*

### E4: Full AgriVision (Graph + Temporal + Fusion)
- *(Failure Category Table Pending)*
- **Most Confused Pairs**: *Pending*
- **Lowest Recall Classes**: *Pending*
- **Observation**: *Pending*
- **Architecture Decision**: *Pending*

---

## 8. Final Architecture Summary
*To be filled after E4 is completed, providing a straightforward, evidence-based justification for the final deployed system.*

| Component | Keep? | Reason |
| :--- | :--- | :--- |
| **DINOv2 Backbone** | ⏳ | Baseline reference architecture. Awaiting evidence. |
| **Graph Transformer** | ⏳ | *Awaiting evidence from E2* |
| **Temporal Transformer** | ⏳ | *Awaiting evidence from E3* |
| **Cross-Attention Fusion** | ⏳ | *Awaiting evidence from E4* |

---

## 9. Reproducibility Checklist
*(Check off before concluding the evaluation phase)*
- [ ] Dataset version recorded
- [ ] Configuration frozen across E1-E4
- [ ] Random seeds logged
- [ ] Training code version (git hash) recorded
- [ ] Checkpoint archived
- [ ] ONNX model archived
- [ ] Experiment metadata JSON archived
- [ ] Dataset audit JSON archived

