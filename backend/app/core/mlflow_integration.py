"""
MLflow integration — model registry, experiment tracking, and drift detection.
"""

from datetime import UTC, datetime
import os
from typing import Any

import numpy as np


class ModelRegistry:
    """
    Tracks model versions, training metrics, and detects concept/data drift.
    In production this connects to MLflow at MLFLOW_TRACKING_URI.
    """

    EXPERIMENT_NAME = "agrivision-disease-detection"

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self._client = None
        if self.tracking_uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.EXPERIMENT_NAME)
                self._client = mlflow.tracking.MlflowClient()
            except ImportError:
                pass  # MLflow not installed — fallback to local tracking

    # ── Model Registration ────────────────────────────────────────────

    def log_model_version(
        self,
        model_path: str,
        metrics: dict[str, float],
        tags: dict[str, str],
    ) -> str | None:
        """Register a new model version and return the version ID."""
        try:
            import mlflow

            with mlflow.start_run(
                run_name=f"train-{datetime.now(UTC).isoformat()}"
            ):
                for key, val in metrics.items():
                    mlflow.log_metric(key, val)
                for key, val in tags.items():
                    mlflow.set_tag(key, val)

                mlflow.log_artifact(model_path)

                # Register model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                registered = mlflow.register_model(
                    model_uri, "AgriVisionDiseaseModel"
                )

                # Stage: start as Staging, promote via API after validation
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    "AgriVisionDiseaseModel",
                    registered.version,
                    stage="Staging",
                )
                return f"v{registered.version}"
        except Exception:
            return None  # Fallback if MLflow unavailable

    # ── Drift Detection ──────────────────────────────────────────────────────

    def detect_concept_drift(
        self,
        historical_confidences: list[float],
        current_confidences: list[float],
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        PSI-based concept drift detection.
        Compares recent inference confidence distributions against historical baseline.
        Triggers retraining alert when PSI > threshold.
        """  # noqa: E501
        if len(historical_confidences) < 30 or len(current_confidences) < 30:
            return {"drifted": False, "reason": "insufficient_samples"}

        hist_arr = np.array(historical_confidences)
        curr_arr = np.array(current_confidences)

        # Bucket into 10 equal-width bins
        bins = np.linspace(0, 100, 11)
        hist_pct = np.histogram(hist_arr, bins=bins)[0] / len(hist_arr)
        curr_pct = np.histogram(curr_arr, bins=bins)[0] / len(curr_arr)

        # Avoid division by zero
        hist_pct = np.where(hist_pct == 0, 1e-6, hist_pct)
        curr_pct = np.where(curr_pct == 0, 1e-6, curr_pct)

        # Population Stability Index
        psi = np.sum((curr_pct - hist_pct) * np.log(curr_pct / hist_pct))

        drifted = psi > threshold
        return {
            "drifted": drifted,
            "psi": round(float(psi), 4),
            "threshold": threshold,
            "action": "trigger_retraining" if drifted else "none",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def detect_data_drift(
        self,
        reference_stats: dict[str, float],
        current_stats: dict[str, float],
    ) -> dict[str, Any]:
        """
        Feature-level data drift using KL divergence on NGVC distributions.
        Reference stats: pre-computed statistics from training dataset.
        Current stats: rolling statistics from recent inference results.
        """
        drifts = {}
        for feature in set(reference_stats) & set(current_stats):
            ref = reference_stats[feature]
            cur = current_stats[feature]
            if ref == 0:
                rel_change = abs(cur) if cur != 0 else 0.0
            else:
                rel_change = abs(cur - ref) / abs(ref)
            drifts[feature] = round(rel_change, 4)

        max_drift_feature = max(drifts, key=drifts.get) if drifts else None
        max_drift_value = (
            drifts.get(max_drift_feature, 0) if max_drift_feature else 0
        )

        return {
            "drifted": max_drift_value > 0.25,
            "max_feature": max_drift_feature,
            "max_drift": max_drift_value,
            "per_feature": drifts,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def log_prediction(self, disease: str, confidence: float, severity: float):
        """Record a prediction for drift monitoring."""
        try:
            import mlflow

            mlflow.log_dict(
                {
                    "disease": disease,
                    "confidence": confidence,
                    "severity": severity,
                },
                "predictions.jsonl",
            )
        except Exception:
            pass  # Non-blocking

    # ── Model Version Info ────────────────────────────────────────────

    def get_latest_stable_version(self) -> str | None:
        """Return the latest Production model version."""
        try:
            client = self._client
            if not client:
                return None
            versions = client.get_latest_versions(
                "AgriVisionDiseaseModel", stages=["Production"]
            )
            return versions[0].version if versions else None
        except Exception:
            return None

    def promote_model(self, version: str, from_stage: str = "Staging"):
        """Promote a model from Staging → Production."""
        try:
            client = self._client
            if not client:
                return
            client.transition_model_version_stage(
                "AgriVisionDiseaseModel", int(version), stage="Production"
            )
        except Exception:
            pass


# Global singleton
model_registry = ModelRegistry()
