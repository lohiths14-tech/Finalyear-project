from __future__ import annotations

import base64
import json
import logging
import math
import random
from typing import Any

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import torch

from ..core.config import settings
from ..models.architectures import (
    AgriVisionEnsemble,
    NGVCNet,
    SpatialGraphNet,
    TemporalProgressionNet,
)
from ..quality.calibration import OODDetector
from ..quality.explainability import AgriExplainer
from ..utils.image_processing import AgriImageProcessor

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
# Default fallback classes if no classes.json is found in weights dir
DISEASE_CLASSES = [
    "Healthy",
    "Rice Blast (Magnaporthe oryzae)",
    "Brown Spot (Bipolaris oryzae)",
    "Leaf Scald (Gerlachia oryzae)",
    "Bacterial Blight (Xanthomonas oryzicola)",
]

TREATMENT_MAP = {
    "Healthy": "No treatment required. Continue routine monitoring.",
    "Rice Blast (Magnaporthe oryzae)": "Tricyclazole 75% WP @ 0.6 g/L every 14 days.",  # noqa: E501
    "Brown Spot (Bipolaris oryzae)": "Propiconazole 25% EC @ 1 ml/L at boot-leaf stage.",  # noqa: E501
    "Leaf Scald (Gerlachia oryzae)": "Remove infected debris; apply Mancozeb 75% WP @ 2 g/L.",  # noqa: E501
    "Bacterial Blight (Xanthomonas oryzicola)": "Copper Oxychloride 50% WP @ 3 g/L; avoid overhead irrigation.",  # noqa: E501
}

# NOTE: These labels are proxy names derived from learned neural feature
# magnitudes, NOT direct spectral band measurements. They represent
# feature partitions of a 64-dim embedding space, named for interpretability.
SPECTRAL_SUBJECTS = [
    "Chlorophyll (proxy)",
    "Nitrogen (proxy)",
    "Water Stress (proxy)",
    "Structure (proxy)",
    "Biomass (proxy)",
    "Pathogen (proxy)",
]

# Maximum patches to process per image — prevents OOM on very large images
MAX_PATCHES = 256


class InferenceService:
    def __init__(self, model_paths: dict[str, str] | None = None):
        self.processor = AgriImageProcessor()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.loaded_models = set()
        self.ood_detector = OODDetector(
            msp_threshold=0.25,
            entropy_threshold=1.55,
            energy_threshold=-2.0,
            method="msp",
        )

        self.classes = list(DISEASE_CLASSES)
        self.treatments = dict(TREATMENT_MAP)
        self.trained_on_real_data = False

        # Load classes and treatments from resolved model dir
        classes_path = settings.resolve_model_base_dir() / "classes.json"
        if classes_path.exists():
            try:
                with open(classes_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if "classes" in data:
                        self.classes = data["classes"]
                    if "treatments" in data:
                        self.treatments = data["treatments"]
                    self.trained_on_real_data = data.get(
                        "trained_on_real_data", False
                    )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to load dynamic classes: %s", e
                )

        # Initialize model architectures
        self.spectral_net = NGVCNet(input_dim=128, feature_dim=64).to(
            self.device
        )
        self.explainer = AgriExplainer(model=self.spectral_net)
        self.spatial_net = SpatialGraphNet(in_channels=64, out_channels=64).to(
            self.device
        )
        self.temporal_net = TemporalProgressionNet(input_dim=64).to(
            self.device
        )
        self.meta_learner = AgriVisionEnsemble(
            feature_dim=64, num_classes=len(self.classes)
        ).to(self.device)

        # Load pre-trained weights if provided or discovered
        if model_paths is None:
            model_paths = settings.get_model_paths()

        if model_paths:
            self._load_weights(model_paths)
        else:
            from ..core.logger import (
                logger,
            )  # deferred to avoid circular import

            logger.warning(
                "⚠️  Models running on RANDOM INIT — non-deterministic. "
                "Run scripts/train_paddy_doctor.py to produce predictions."
            )

        # Initialize ONNX Runtime sessions (preferring dynamic INT8 quantized)
        self.ort_sessions: dict[str, Any] = {}
        self.use_onnx = False
        self._init_onnx_sessions()

        self._set_eval()

    def _init_onnx_sessions(self):
        """Loads ONNX Runtime sessions (INT8 quantized preferred) for minimal memory footprint."""
        try:
            import onnxruntime as ort

            onnx_dir = settings.resolve_model_base_dir() / "onnx"
            if not onnx_dir.exists():
                return

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            candidates = {
                "spectral": ["spectral_net_quant.onnx", "spectral_net.onnx"],
                "spatial": ["spatial_net_quant.onnx", "spatial_net.onnx"],
                "temporal": ["temporal_net_quant.onnx", "temporal_net.onnx"],
                "ensemble": ["ensemble_quant.onnx", "ensemble.onnx"],
            }

            for key, filenames in candidates.items():
                for fname in filenames:
                    fpath = onnx_dir / fname
                    if fpath.exists():
                        try:
                            self.ort_sessions[key] = ort.InferenceSession(
                                str(fpath),
                                opts,
                                providers=["CPUExecutionProvider"],
                            )
                            logging.getLogger(__name__).info(
                                "Loaded ONNX session for %s (%s)", key, fname
                            )
                            break
                        except Exception as e:
                            logging.getLogger(__name__).warning(
                                "Failed to initialize ONNX %s: %s", fpath, e
                            )

            if self.ort_sessions:
                self.use_onnx = True
                logging.getLogger(__name__).info(
                    "ONNX Runtime acceleration active: %s",
                    list(self.ort_sessions.keys()),
                )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "ONNX initialization skipped: %s", e
            )

    def _load_weights(self, model_paths: dict[str, str]):
        """Safely load serialized weights with map_location."""
        loaders = {
            "spectral": self.spectral_net,
            "spatial": self.spatial_net,
            "temporal": self.temporal_net,
            "ensemble": self.meta_learner,
        }
        for key, model in loaders.items():
            path = model_paths.get(key)
            if path:
                state = torch.load(
                    path, map_location=self.device, weights_only=True
                )
                model.load_state_dict(state)
                self.loaded_models.add(key)

    def _set_eval(self):
        self.spectral_net.eval()
        self.spatial_net.eval()
        self.temporal_net.eval()
        self.meta_learner.eval()

    # ------------------------------------------------------------------
    def _extract_grid_features(
            self,
            processed_img: np.ndarray,
            mask: np.ndarray,
            h: int,
            w: int):
        # Adaptive grid resolution (~20x20 uniform tiles)
        grid_cols = 20
        grid_rows = max(10, min(30, int(grid_cols * (h / w))))
        patch_w = max(32, w // grid_cols)
        patch_h = max(32, h // grid_rows)

        regions = []
        node_features = []
        ngvc_accumulator = np.zeros(128, dtype=np.float32)

        for r_idx in range(grid_rows):
            y = r_idx * patch_h
            ph = min(patch_h, h - y)
            for c_idx in range(grid_cols):
                x = c_idx * patch_w
                pw = min(patch_w, w - x)

                if ph <= 0 or pw <= 0:
                    continue

                patch_mask = mask[y: y + ph, x: x + pw]
                if patch_mask.size == 0:
                    continue

                veg_ratio = (
                    float((patch_mask > 0).mean())
                    if patch_mask.size > 0
                    else 0.0
                )
                if veg_ratio < 0.10:  # Skip non-vegetation background
                    continue

                veg_pixels = int((patch_mask > 0).sum())
                region = processed_img[y: y + ph, x: x + pw]
                if region.size == 0:
                    continue

                # Spectral analysis (NGVC)
                ngvc = self.processor.extract_ngvc(region, bins=128)
                ngvc_accumulator += ngvc

                if "spectral" in self.ort_sessions:
                    ngvc_np = ngvc.reshape(1, -1).astype(np.float32)
                    feat_np = self.ort_sessions["spectral"].run(
                        None, {"spectral_hist": ngvc_np}
                    )[0]
                    feat = torch.from_numpy(feat_np).to(self.device)
                else:
                    ngvc_tensor = torch.FloatTensor(
                        ngvc).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.spectral_net(ngvc_tensor)

                node_features.append(feat)
                regions.append(
                    {
                        "grid_r": r_idx,
                        "grid_c": c_idx,
                        "x": x,
                        "y": y,
                        "w": pw,
                        "h": ph,
                        "veg_pixels": veg_pixels,
                    }
                )

        # Fallback: if all patches were skipped, take the full image as 1 patch
        if not node_features and h > 0 and w > 0:
            ngvc = self.processor.extract_ngvc(processed_img, bins=128)
            ngvc_accumulator = ngvc.copy()
            if "spectral" in self.ort_sessions:
                ngvc_np = ngvc.reshape(1, -1).astype(np.float32)
                feat_np = self.ort_sessions["spectral"].run(
                    None, {"spectral_hist": ngvc_np}
                )[0]
                feat = torch.from_numpy(feat_np).to(self.device)
            else:
                ngvc_tensor = torch.FloatTensor(ngvc).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.spectral_net(ngvc_tensor)
            node_features.append(feat)
            veg_px = (
                int((mask > 0).sum()) if mask.size > 0 else (h * w)
            )
            regions.append(
                {
                    "grid_r": 0,
                    "grid_c": 0,
                    "x": 0,
                    "y": 0,
                    "w": w,
                    "h": h,
                    "veg_pixels": veg_px,
                }
            )

        return (
            regions,
            node_features,
            ngvc_accumulator,
            patch_w,
            grid_rows,
            grid_cols,
        )

    def _apply_temporal_analysis(
            self,
            temporal_sequence,
            spatial_feats,
            num_nodes) -> tuple:
        """Applies temporal analysis ONLY when sequence is available."""
        if (
            ("temporal" in self.loaded_models or "temporal" in self.ort_sessions)
            and temporal_sequence
            and len(temporal_sequence) > 1
        ):
            seq_feats = []
            for frame in temporal_sequence[-5:]:  # Last 5 frames
                f_proc, _ = self.processor.remove_soil_and_shadow(frame)
                f_ngvc = self.processor.extract_ngvc(f_proc, bins=128)
                if "spectral" in self.ort_sessions:
                    f_np = f_ngvc.reshape(1, -1).astype(np.float32)
                    feat_np = self.ort_sessions["spectral"].run(
                        None, {"spectral_hist": f_np}
                    )[0]
                    seq_feats.append(torch.from_numpy(feat_np).to(self.device))
                else:
                    f_t = torch.FloatTensor(f_ngvc).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        seq_feats.append(self.spectral_net(f_t))

            # Stack sequence: (1, seq_len, 64) -> broadcast to (N, seq_len, 64)
            seq_tensor = torch.stack(seq_feats, dim=1)
            temporal_input = seq_tensor.expand(num_nodes, -1, -1)
            if "temporal" in self.ort_sessions:
                temp_np = self.ort_sessions["temporal"].run(
                    None, {"temporal_sequence": temporal_input.cpu().numpy()}
                )[0]
                return torch.from_numpy(temp_np).to(self.device), True

            with torch.no_grad():
                return self.temporal_net(temporal_input), True

        # No real multi-frame history → zero vector, do not fabricate
        return torch.zeros(num_nodes, 64, device=self.device), False

    # ------------------------------------------------------------------
    # Field-level analysis (full SST pipeline)
    # ------------------------------------------------------------------
    def analyze_field(  # noqa: C901
        self,
        image: np.ndarray,
        temporal_sequence: list[np.ndarray] | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Full Spatio-Spectral-Temporal field pipeline."""
        if callable(progress_callback):
            progress_callback({
                "stage": "preprocess",
                "percent": 12,
                "message": "Segmenting canopy vegetation and filtering soil background"
            })
        h, w, _ = image.shape

        # 1. Preprocess
        processed_img, mask = self.processor.remove_soil_and_shadow(image)

        if callable(progress_callback):
            progress_callback({
                "stage": "spectral",
                "percent": 30,
                "message": "Initializing adaptive grid & extracting spectral NGVC signatures"
            })

        # 2 & 3. Grid segmentation & Spectral analysis
        (
            regions,
            node_features,
            ngvc_accumulator,
            patch_size,
            grid_rows,
            grid_cols,
        ) = self._extract_grid_features(processed_img, mask, h, w)

        if not node_features:
            return {
                "status": "insufficient data",
                "error": "No vegetation detected in the provided image.",
            }

        # --- MAX_PATCHES guard: subsample uniformly if too many patches ---
        if len(node_features) > MAX_PATCHES:
            indices = sorted(
                random.sample(range(len(node_features)), MAX_PATCHES)
            )
            node_features = [node_features[i] for i in indices]
            regions = [regions[i] for i in indices]

        node_features_t = torch.cat(node_features, dim=0)  # (N, 64)
        num_nodes = len(regions)

        if callable(progress_callback):
            progress_callback({
                "stage": "spatial",
                "percent": 65,
                "message": f"GraphSAGE spatial aggregation over {num_nodes} patch adjacency graph"
            })

        # 4. Spatial analysis — O(N) vectorized adjacency
        positions = np.array(
            [[r["x"], r["y"]] for r in regions], dtype=np.float32
        )
        dist_matrix = cdist(positions, positions)
        adj_np = (dist_matrix < patch_size * 2.0).astype(np.float32)
        np.fill_diagonal(adj_np, 0.0)  # No self-loops

        if "spatial" in self.ort_sessions:
            spatial_feats_np = self.ort_sessions["spatial"].run(
                None,
                {
                    "node_features": node_features_t.cpu().numpy(),
                    "adjacency_matrix": adj_np,
                },
            )[0]
            spatial_feats = torch.from_numpy(spatial_feats_np).to(self.device)
        elif "spatial" in self.loaded_models:
            adj = torch.tensor(adj_np, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                spatial_feats = self.spatial_net(
                    node_features_t, adj
                )  # (N, 64)
        else:
            # Stub if spatial weights not trained/loaded
            spatial_feats = torch.zeros(num_nodes, 64, device=self.device)

        # 5. Temporal analysis — runs when multi-frame sequence available
        if callable(progress_callback):
            progress_callback({
                "stage": "temporal",
                "percent": 82,
                "message": "Assessing temporal sequence progression"
            })

        temporal_feats, temporal_available = self._apply_temporal_analysis(
            temporal_sequence, spatial_feats, num_nodes
        )

        # 6. Meta-Learner (Ensemble)
        if callable(progress_callback):
            progress_callback({
                "stage": "ensemble",
                "percent": 92,
                "message": "Executing cross-modal ensemble fusion & calibrated classification"
            })

        with torch.no_grad():
            if "ensemble" in self.ort_sessions:
                logits_np, sev_np = self.ort_sessions["ensemble"].run(
                    None,
                    {
                        "spectral_feat": node_features_t.cpu().numpy(),
                        "spatial_feat": spatial_feats.cpu().numpy(),
                        "temporal_feat": temporal_feats.cpu().numpy(),
                    },
                )
                logits = torch.from_numpy(logits_np).to(self.device)
                severity = torch.from_numpy(sev_np).to(self.device)
            else:
                logits, severity = self.meta_learner(
                    node_features_t, spatial_feats, temporal_feats
                )
            probs = torch.softmax(logits, dim=1)
            predictions = probs.mean(dim=0)
            pred_class = int(torch.argmax(predictions).item())
            confidence = float(predictions.max().item() * 100)

            # Top-3 class alternatives
            top3 = torch.topk(predictions, min(3, len(self.classes)))
            alternatives = [
                {
                    "disease": self.classes[i],
                    "confidence": round(float(predictions[i].item() * 100), 2),
                }
                for i in top3.indices
            ]

            # Uncertainty Decomposition (Aleatoric vs. Epistemic)
            eps = 1e-7
            k_classes = max(2, len(self.classes))
            shannon_entropy = -float(
                torch.sum(predictions * torch.log2(predictions + eps)).item()
            )
            max_entropy = math.log2(k_classes)
            norm_aleatoric = float(
                np.clip(shannon_entropy / max_entropy, 0.0, 1.0)
            )

            # 2. Epistemic Uncertainty: Free energy score
            temperature = 1.42
            energy_score = -temperature * float(
                torch.logsumexp(
                    logits.mean(dim=0) / temperature, dim=0
                ).item()
            )
            norm_epistemic = float(
                np.clip((energy_score + 10.0) / 10.0, 0.0, 1.0)
            )

            uncertainty_breakdown = {
                "aleatoric": round(norm_aleatoric, 4),
                "epistemic": round(norm_epistemic, 4),
                "entropy_bits": round(shannon_entropy, 3),
                "energy_score": round(energy_score, 3),
                "calibration_temperature": temperature,
                "interpretation": (
                    "High diagnostic certainty within data manifold."
                    if (norm_aleatoric < 0.4 and norm_epistemic < 0.5)
                    else "Moderate ambiguity; inspect field patches."
                ),
            }

        # Determine healthy reference class index
        healthy_idx = 0
        for idx, name in enumerate(self.classes):
            if name.lower() in ["normal", "healthy"]:
                healthy_idx = idx
                break

        # Node-level predictions and infection mapping
        node_pred_indices = [
            int(torch.argmax(probs[i]).item()) for i in range(num_nodes)
        ]
        infected_nodes = sum(
            1 for idx in node_pred_indices if idx != healthy_idx
        )
        infection_rate = (
            (infected_nodes / num_nodes) * 100.0 if num_nodes > 0 else 0.0
        )

        # If spatial grid detected significant infection, ensure dominant disease reflects the infected patches
        if pred_class == healthy_idx and infection_rate > 15.0:
            infected_counts: dict[int, int] = {}
            for idx in node_pred_indices:
                if idx != healthy_idx:
                    infected_counts[idx] = infected_counts.get(idx, 0) + 1
            if infected_counts:
                pred_class = max(infected_counts, key=infected_counts.get)
                confidence = float(predictions[pred_class].item() * 100)

        node_severities = []
        for i in range(num_nodes):
            p_idx = node_pred_indices[i]
            if p_idx == healthy_idx:
                sev_val = float(
                    np.clip(float(severity[i].item()) * 0.1, 0.0, 10.0)
                )
            else:
                p_prob = float(probs[i][p_idx].item())
                sev_val = float(np.clip(30.0 + p_prob * 65.0, 30.0, 95.0))
            node_severities.append(round(sev_val, 2))

        avg_severity = (
            float(np.mean(node_severities)) if node_severities else 0.0
        )
        total_veg_pixels = sum(r["veg_pixels"] for r in regions)
        veg_coverage = (total_veg_pixels / (h * w)) * 100

        disease_label = self.classes[pred_class]

        # OOD Detection Guard
        logits_mean_np = logits.mean(dim=0).cpu().numpy()
        is_ood, ood_score, ood_reason = self.ood_detector.is_ood(
            logits_mean_np
        )

        withheld = (confidence < 25.0) or is_ood
        if is_ood:
            disease_label = "Unknown"
            treatment = (
                f"OOD sample detected ({ood_reason}). "
                "The image does not match in-distribution pathology patterns."
            )
            spread_risk = "Unknown"
        elif withheld:
            disease_label = "Unknown"
            treatment = (
                "The system cannot confidently identify symptoms in this "
                "scan (confidence < 25%). Ensure standard daylight illumination and canopy focus."
            )
            spread_risk = "Unknown"
        else:
            base_treatment = self.treatments.get(
                disease_label, "Monitor field closely."
            )
            if confidence < 45.0:
                alt_names = [a["disease"].split(" (")[0] for a in alternatives[1:3] if a["disease"] != disease_label]
                alt_text = f", possibly {', '.join(alt_names)}" if alt_names else ""
                treatment = (
                    f"Calibrated diagnosis: {disease_label.split(' (')[0]} ({confidence:.1f}% confidence{alt_text}). "
                    f"Field patch variation detected. Confirm via close-up leaf scan or ground spot check before chemical application. "
                    f"ICAR Guideline: {base_treatment}"
                )
            else:
                treatment = base_treatment
            spread_risk = (
                "High"
                if (avg_severity > 40 or infection_rate > 35)
                else "Low"
            )

        # Per-node structured outputs for spatial map
        nodes = [
            {
                "id": i,
                "x": reg["x"],
                "y": reg["y"],
                "w": reg["w"],
                "h": reg["h"],
                "severity": node_severities[i],
                "prediction": self.classes[node_pred_indices[i]],
            }
            for i, reg in enumerate(regions)
        ]

        grid_data = [
            {
                "x_percent": round(r["x"] / w * 100, 2),
                "y_percent": round(r["y"] / h * 100, 2),
                "w_percent": round(r["w"] / w * 100, 2),
                "h_percent": round(r["h"] / h * 100, 2),
                "severity": node_severities[i],
                "disease": self.classes[node_pred_indices[i]],
            }
            for i, r in enumerate(regions)
        ]

        # NGVC field-average for spectral chart
        total_ngvc = ngvc_accumulator / (len(regions) + 1e-6)
        ngvc_data = [
            {"intensity": int(i * 255 / 128), "value": round(float(v), 6)}
            for i, v in enumerate(total_ngvc)
            if i % 3 == 0
        ]

        # Spatio-Temporal Progression: single-scan logistic growth estimate
        progression_data = self._build_progression(avg_severity)
        spectral_radar = self._build_spectral_radar(spatial_feats)

        # Continuous High-Resolution Gaussian Heatmap Generation
        heatmap_base64 = None
        try:

            # 1. Build dense 2D grid of severity values
            grid_sev = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            grid_count = np.zeros((grid_rows, grid_cols), dtype=np.float32)

            for i, reg in enumerate(regions):
                r_i = reg.get("grid_r", 0)
                c_i = reg.get("grid_c", 0)
                grid_sev[r_i, c_i] += float(severity[i].item())
                grid_count[r_i, c_i] += 1.0

            # Normalize cells
            valid_mask = grid_count > 0
            grid_sev[valid_mask] /= grid_count[valid_mask]

            # Fill unpopulated background cells with median
            if valid_mask.any():
                fill_val = float(np.median(grid_sev[valid_mask]))
                grid_sev[~valid_mask] = fill_val

            # 2. Continuous bicubic interpolation
            dense_map = cv2.resize(
                grid_sev, (w, h), interpolation=cv2.INTER_CUBIC
            )

            # 3. Resolution-adaptive Gaussian smoothing
            ksize = max(21, int(max(h, w) * 0.04) | 1)
            heatmap_smooth = cv2.GaussianBlur(dense_map, (ksize, ksize), 0)

            # 4. Map severity to 0-255 range
            norm = np.clip(
                (heatmap_smooth / 100.0) * 255.0, 0, 255
            ).astype(np.uint8)

            # 5. Apply TURBO colormap
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            rgba = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)

            # 6. Canopy-Aware Alpha Mask
            alpha_ch = (
                norm.astype(np.float32) * (200.0 / 255.0)
            ).astype(np.uint8)
            dilated_mask = cv2.dilate(
                mask, np.ones((7, 7), np.uint8), iterations=1
            )
            alpha_ch[dilated_mask == 0] = 0
            rgba[:, :, 3] = alpha_ch

            success, buf = cv2.imencode(".png", rgba)
            if success:
                heatmap_base64 = base64.b64encode(
                    buf.tobytes()
                ).decode("utf-8")
        except Exception as _exc:
            logging.getLogger(__name__).warning(
                "heatmap_base64 generation error: %s", _exc
            )
            heatmap_base64 = None

        if callable(progress_callback):
            progress_callback({
                "stage": "complete",
                "percent": 100,
                "message": "Field intelligence telemetry computed successfully"
            })

        return {
            "disease_type": disease_label,
            "treatment": treatment,
            "average_severity": round(avg_severity, 2),
            "confidence": round(confidence, 2),
            "withheld": withheld,
            "temporal_available": temporal_available,
            "alternatives": alternatives,
            "spread_risk": spread_risk,
            "nodes": nodes,
            "progression_data": progression_data,
            "spectral_radar": spectral_radar,
            "ngvc_data": ngvc_data,
            "grid_data": grid_data,
            "infection_rate": round(infection_rate, 2),
            "percentage_infected_area": round(infection_rate, 2),
            "percentage_healthy_area": round(
                max(0.0, 100.0 - infection_rate), 2
            ),
            "veg_coverage": round(veg_coverage, 2),
            "heatmap_base64": heatmap_base64,
            "uncertainty_breakdown": uncertainty_breakdown,
            "ood": {
                "is_ood": is_ood,
                "ood_score": round(float(ood_score), 3),
                "reason": ood_reason,
            },
            # Raw data for downstream consumers
            "disease_distribution": predictions.cpu().numpy().tolist(),
            "severity_map": severity.cpu().numpy().flatten().tolist(),
            "regions": regions,
        }

    # ------------------------------------------------------------------
    # Leaf-level analysis (spectral only)
    # ------------------------------------------------------------------
    def analyze_leaf(self, image: np.ndarray) -> dict[str, Any]:
        """Micro-level leaf diagnostic using spectral-only inference."""
        processed_img, mask = self.processor.remove_soil_and_shadow(image)
        ngvc = self.processor.extract_ngvc(processed_img, bins=128)

        if "spectral" in self.ort_sessions:
            ngvc_np = ngvc.reshape(1, -1).astype(np.float32)
            feat_np = self.ort_sessions["spectral"].run(
                None, {"spectral_hist": ngvc_np}
            )[0]
            feat = torch.from_numpy(feat_np).to(self.device)
        else:
            ngvc_tensor = torch.FloatTensor(ngvc).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.spectral_net(ngvc_tensor)

        v_b_size = feat.size(0)
        with torch.no_grad():
            if "ensemble" in self.ort_sessions:
                zero_np = np.zeros((v_b_size, 64), dtype=np.float32)
                logits_np, sev_np = self.ort_sessions["ensemble"].run(
                    None,
                    {
                        "spectral_feat": feat.cpu().numpy(),
                        "spatial_feat": zero_np,
                        "temporal_feat": zero_np,
                    },
                )
                logits = torch.from_numpy(logits_np).to(self.device)
                severity = torch.from_numpy(sev_np).to(self.device)
            else:
                logits, severity = self.meta_learner(
                    feat,
                    torch.zeros(
                        v_b_size, 64, device=self.device
                    ),  # No spatial context for single leaf
                    torch.zeros(
                        v_b_size, 64, device=self.device
                    ),  # No temporal context
                )
            probs = torch.softmax(logits, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs.max().item() * 100)
            sev = float(severity.item())

            # Top 3 predictions
            top3 = torch.topk(probs, min(3, len(self.classes)))
            alternatives = [
                {
                    "disease": self.classes[i],
                    "confidence": round(float(probs[0][i].item() * 100), 2),
                }
                for i in top3.indices[0]
            ]

            # Uncertainty Decomposition for leaf
            eps = 1e-7
            leaf_probs = probs[0]
            shannon_entropy = -float(
                torch.sum(leaf_probs * torch.log2(leaf_probs + eps)).item()
            )
            max_entropy = math.log2(max(2, len(self.classes)))
            norm_aleatoric = float(
                np.clip(shannon_entropy / max_entropy, 0.0, 1.0)
            )
            temperature = 1.42
            energy_score = -temperature * float(
                torch.logsumexp(logits[0] / temperature, dim=0).item()
            )
            norm_epistemic = float(
                np.clip((energy_score + 10.0) / 10.0, 0.0, 1.0)
            )

            uncertainty_breakdown = {
                "aleatoric": round(norm_aleatoric, 4),
                "epistemic": round(norm_epistemic, 4),
                "entropy_bits": round(shannon_entropy, 3),
                "energy_score": round(energy_score, 3),
                "calibration_temperature": temperature,
            }

        disease_label = self.classes[pred_class]

        # OOD Detection Guard
        logits_np = logits[0].cpu().numpy()
        is_ood, ood_score, ood_reason = self.ood_detector.is_ood(logits_np)

        withheld = (confidence < 25.0) or is_ood
        if is_ood:
            disease_label = "Out-of-Distribution / Unknown"
            status = "insufficient data"
            treatment = (
                f"Out-of-distribution sample detected ({ood_reason}). "
                "Please provide a clear, standard paddy crop leaf image."
            )
        elif withheld:
            disease_label = "Unknown"
            status = "insufficient data"
            treatment = (
                "The model cannot confidently identify disease symptoms (confidence < 25%). "
                "Please retry with a clear, well-lit leaf image."
            )
        else:
            # Health status: healthy/normal class
            healthy_idx = 0
            for idx, name in enumerate(self.classes):
                if name.lower() in ["normal", "healthy"]:
                    healthy_idx = idx
                    break
            is_healthy = pred_class == healthy_idx
            status = "Healthy" if is_healthy else "Disease Detected"
            base_treatment = self.treatments.get(
                disease_label, "Monitor closely."
            )
            if confidence < 45.0:
                alt_names = [a["disease"].split(" (")[0] for a in alternatives[1:3] if a["disease"] != disease_label]
                alt_text = f", possibly {', '.join(alt_names)}" if alt_names else ""
                treatment = (
                    f"Calibrated diagnosis: {disease_label.split(' (')[0]} ({confidence:.1f}% confidence{alt_text}). "
                    f"Leaf symptom expression is mild or intermediate. Standard guidance: {base_treatment}"
                )
            else:
                treatment = base_treatment

        return {
            "type": "Leaf Diagnostic",
            "status": status,
            "disease_type": disease_label,
            "treatment": treatment,
            "severity": round(sev, 2),
            "confidence": round(confidence, 2),
            "withheld": withheld,
            "alternatives": alternatives,
            "ngvc": ngvc.tolist(),
            "uncertainty_breakdown": uncertainty_breakdown,
            "ood": {
                "is_ood": is_ood,
                "ood_score": round(float(ood_score), 3),
                "reason": ood_reason,
            },
        }

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_progression(current_severity: float) -> list[dict]:
        """Build 6-point progression chart using logistic growth model."""
        k = 0.3  # Logistic growth rate constant
        history = [
            current_severity * math.exp(-k * (4 - t)) for t in range(1, 5)
        ]
        forecast = [current_severity * math.exp(k * t) for t in range(1, 3)]
        forecast = [min(v, 100.0) for v in forecast]
        combined = history + forecast
        return [
            {
                "day": f"Day {i + 1}",
                "actual": round(combined[i], 1) if i < 4 else None,
                "predicted": round(combined[i], 1),
                "simulated": True,  # Always True in single-scan mode
                "point_type": "estimated_history" if i < 4 else "forecast",
            }
            for i in range(6)
        ]

    @staticmethod
    def _build_spectral_radar(spatial_feats: torch.Tensor) -> list[dict]:
        """Derive spectral fingerprint radar from spatial features."""
        feat_mean = spatial_feats.mean(dim=0).cpu().numpy()
        # Map 64 feature dims to 6 spectral proxies via even partitioning
        chunk = len(feat_mean) // 6
        subjects = SPECTRAL_SUBJECTS
        return [
            {
                "subject": subjects[i],
                "A": round(
                    float(
                        np.abs(feat_mean[i * chunk: (i + 1) * chunk]).mean()
                    )
                    * 100,
                    1,
                ),
                "B": round(
                    100
                    - float(
                        np.abs(feat_mean[i * chunk: (i + 1) * chunk]).mean()
                    )
                    * 80,
                    1,
                ),
                "fullMark": 150,
            }
            for i in range(6)
        ]
