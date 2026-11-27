
import torch
import torch.nn as nn

from ..models.architectures import (
    CrossAttentionFusion,
    FoundationVisionEncoder,
    GraphTransformer,
    TemporalTransformer,
)


class AgriPulseTrainingWrapper(nn.Module):
    """
    Wraps the modular architecture components into a single forward pass
    for training. Note: Spatial and Temporal features are mocked or aggregated
    here if not using full video sequences yet.
    """

    def __init__(
        self, feature_dim: int = 64, num_classes: int = 5, mode: str = "full"
    ):
        super().__init__()
        self.mode = mode
        self.vision_encoder = FoundationVisionEncoder(feature_dim=feature_dim)

        # Core modules depending on mode
        self.graph_transformer = GraphTransformer(
            in_channels=feature_dim, out_channels=feature_dim
        )
        self.temporal_transformer = TemporalTransformer(input_dim=feature_dim)

        # In partial modes we bypass fusion and use simple heads
        if self.mode in ["baseline", "graph_only", "temporal_only"]:
            self.baseline_classifier = nn.Linear(feature_dim, num_classes)
            self.baseline_severity = nn.Linear(feature_dim, 1)
        else:
            self.fusion = CrossAttentionFusion(
                feature_dim=feature_dim, num_classes=num_classes, env_dim=0
            )

    def forward(
        self, x: torch.Tensor, is_cached: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # If using cached dataset, x is already the 64-dim feature vector
        if is_cached:
            vis_features = x
        else:
            vis_features = self.vision_encoder(x)  # (batch, feature_dim)

        # Baseline Mode Check
        if self.mode == "baseline":
            logits = self.baseline_classifier(vis_features)
            severity = (
                torch.sigmoid(self.baseline_severity(vis_features)) * 100
            )
            return logits, severity

        # Spatial Module
        spatial_out = vis_features
        if self.mode in ["full", "graph_only"]:
            batch_size = x.size(0)
            adj = torch.eye(
                1, device=x.device
            )  # (1, 1) single node (isolated patch fallback)
            spatial_features = []
            for i in range(batch_size):
                s_feat = self.graph_transformer(
                    vis_features[i].unsqueeze(0), adj
                )
                spatial_features.append(s_feat)
            spatial_out = torch.cat(
                spatial_features, dim=0
            )  # (batch, feature_dim)

            if self.mode == "graph_only":
                logits = self.baseline_classifier(spatial_out)
                severity = (
                    torch.sigmoid(self.baseline_severity(spatial_out)) * 100
                )
                return logits, severity

        # Temporal Module
        temporal_out = vis_features
        if self.mode in ["full", "temporal_only"]:
            seq_features = vis_features.unsqueeze(1)  # (batch, 1, feature_dim)
            temporal_out = self.temporal_transformer(
                seq_features
            )  # (batch, feature_dim)

            if self.mode == "temporal_only":
                logits = self.baseline_classifier(temporal_out)
                severity = (
                    torch.sigmoid(self.baseline_severity(temporal_out)) * 100
                )
                return logits, severity

        # 4. Fusion (Full Mode)
        logits, severity = self.fusion(
            visual_feat=vis_features,
            spatial_feat=spatial_out,
            temporal_feat=temporal_out,
            env_data=None,
        )

        return logits, severity
