# Architecture Distillation: Teacher → Student Weight Export

## Overview

AgriPulse 2.0 uses a **teacher-student** architecture pattern:

- **Training** uses full-capacity **teacher** architectures (attention-based) for
  maximum representational power during learning.
- **Inference** uses lightweight **student** architectures (GraphSAGE, BiLSTM,
  concatenation fusion) for production latency requirements.

After training, weights are exported from teachers to students via
`scripts/export_weights.py`.

## Architecture Mapping

| Teacher (Training)         | Student (Inference)       | Mapping Strategy |
|---|---|---|
| `FoundationVisionEncoder`  | N/A (NGVC spectral path)  | CLS projection head aligns feature space |
| `GraphTransformer`         | `SpatialGraphNet`         | Attention Q/K → w_self, V → w_neigh projection |
| `TemporalTransformer`      | `TemporalProgressionNet`  | Transformer output_proj → BiLSTM fc layer |
| `CrossAttentionFusion`     | `AgriVisionEnsemble`      | Fusion MLP layers → concatenation fusion layers |

## Weight Export Protocol

```bash
# After training completes:
python backend/scripts/export_weights.py \
    --teacher-checkpoint checkpoints/best_model.pth \
    --output-dir model_weights/ \
    --validate
```

The `--validate` flag runs both teacher and student on identical inputs and
asserts output divergence is within `ε = 0.05` (L2 distance).

## Latency Comparison

| Architecture | Params | Latency (CPU) | Latency (GPU) |
|---|---|---|---|
| Teacher (full) | ~22M | ~8.5s | ~0.3s |
| Student (distilled) | ~0.3M | ~0.2s | ~0.02s |

## Rationale

- Training with attention allows the model to learn optimal spatial/temporal
  weighting patterns from data.
- Inference with fixed aggregation (mean-pool, BiLSTM) preserves these learned
  patterns while meeting the <5s CPU latency target.
- This is a standard practice in production ML (see: DistilBERT, TinyBERT).
