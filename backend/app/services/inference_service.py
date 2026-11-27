from __future__ import annotations

import base64
import json
import logging
import math
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
# Canonical Paddy Doctor 10-class vision labels (exact order matching mobilenet_v2_paddy.pt / ONNX)
PADDY_VISION_CLASSES = [
    "Bacterial Blight (Xanthomonas oryzae)",
    "Bacterial Leaf Streak",
    "Bacterial Panicle Blight",
    "Rice Blast (Magnaporthe oryzae)",
    "Brown Spot (Bipolaris oryzae)",
    "Dead Heart (Stem Borer)",
    "Downy Mildew",
    "Rice Hispa",
    "Healthy",
    "Rice Tungro",
]

DISEASE_CLASSES = [
    "Healthy",
    "Rice Blast (Magnaporthe oryzae)",
    "Brown Spot (Bipolaris oryzae)",
    "Leaf Scald (Gerlachia oryzae)",
    "Bacterial Blight (Xanthomonas oryzae)",
    "Bacterial Blight (Xanthomonas oryzicola)",
    "Bacterial Leaf Streak",
    "Bacterial Panicle Blight",
    "Dead Heart (Stem Borer)",
    "Downy Mildew",
    "Rice Hispa",
    "Rice Tungro",
]

TREATMENT_MAP = {
    "Healthy": "Field health is optimal. Maintain balanced N-P-K fertilization and alternate wetting and drying irrigation. No fungicide application needed.",
    "Rice Blast (Magnaporthe oryzae)": "Apply Tricyclazole 75% WP @ 0.6 g/L or Isoprothiolane 40% EC @ 1.5 ml/L at early tillering and panicle emergence. Avoid excess nitrogen fertilizer.",
    "Brown Spot (Bipolaris oryzae)": "Foliar spray of Mancozeb 75% WP @ 2.0 g/L or Propiconazole 25% EC @ 1.0 ml/L at boot-leaf stage. Ensure adequate potassium and silicon nutrition.",
    "Leaf Scald (Gerlachia oryzae)": "Remove infected debris; apply Carbendazim 50% WP @ 1.0 g/L or Mancozeb 75% WP @ 2.0 g/L.",
    "Bacterial Blight (Xanthomonas oryzae)": "Spray Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline @ 0.1 g/L. Drain stagnant field water and delay nitrogen top-dressing.",
    "Bacterial Blight (Xanthomonas oryzicola)": "Spray Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline @ 0.1 g/L. Drain stagnant field water and avoid overhead irrigation.",
    "Bacterial Leaf Streak": "Apply Copper Hydroxide 53.8% DF @ 2.0 g/L or Validamycin 3% L @ 2.0 ml/L. Ensure balanced potassium fertilization.",
    "Bacterial Panicle Blight": "Apply Kasugamycin 3% SL @ 2.0 ml/L at booting and early heading stages. Use certified disease-free seed.",
    "Dead Heart (Stem Borer)": "Stem borer (Scirpophaga incertulas) damage. Apply Chlorantraniliprole 18.5% SC @ 0.3 ml/L or Cartap Hydrochloride 4G @ 25 kg/ha.",
    "Downy Mildew": "Spray Metalaxyl-M + Mancozeb 68% WP @ 2.5 g/L and drain stagnant water from low-lying pockets.",
    "Rice Hispa": "Spray Quinalphos 25% EC @ 2.0 ml/L or Chlorpyrifos 20% EC @ 2.5 ml/L; clear grassy bund weeds.",
    "Rice Tungro": "Control Green Leafhopper vector using Imidacloprid 17.8% SL @ 0.3 ml/L or Thiamethoxam 25% WG @ 0.2 g/L; rogue infected hills.",
}

# NOTE: These labels are proxy names derived from learned neural feature
# magnitudes, NOT direct spectral band measurements.
SPECTRAL_SUBJECTS = [
    "Chlorophyll (proxy)",
    "Nitrogen (proxy)",
    "Water Stress (proxy)",
    "Structure (proxy)",
    "Biomass (proxy)",
    "Pathogen (proxy)",
]

# Maximum patches to process per image
MAX_PATCHES = 144

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceService:
    def __init__(self, model_paths: dict[str, str] | None = None):
        self.processor = AgriImageProcessor()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.loaded_models = set()
        self.ood_detector = OODDetector(
            msp_threshold=0.25,
            entropy_threshold=1.95,
            energy_threshold=-0.5,
            method="msp",
        )

        self.classes = list(PADDY_VISION_CLASSES)
        self.legacy_classes = [
            "Healthy",
            "Rice Blast (Magnaporthe oryzae)",
            "Brown Spot (Bipolaris oryzae)",
            "Leaf Scald (Gerlachia oryzae)",
            "Bacterial Blight (Xanthomonas oryzicola)",
        ]
        self.treatments = dict(TREATMENT_MAP)
        self.trained_on_real_data = False
        self.calibration_temperature = 1.42

        # Load metadata from resolved model dir
        classes_path = settings.resolve_model_base_dir() / "classes.json"
        if classes_path.exists():
            try:
                with open(classes_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if "classes" in data and len(data["classes"]) == 5:
                        self.legacy_classes = data["classes"]
                    if "treatments" in data:
                        self.treatments.update(data["treatments"])
                    self.trained_on_real_data = data.get(
                        "trained_on_real_data", False
                    )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to load dynamic classes: %s", e
                )

        # Initialize auxiliary SST architectures
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
            feature_dim=64, num_classes=len(self.legacy_classes)
        ).to(self.device)

        # Real Vision CNN student (PyTorch fallback when ONNX unavailable)
        self.vision_student_pt = None

        # Load pre-trained weights if provided or discovered
        if model_paths is None:
            model_paths = settings.get_model_paths()

        if model_paths:
            self._load_weights(model_paths)
        else:
            from ..core.logger import logger

            logger.warning(
                "Models running without discovered .pt weights — checking ONNX sessions."
            )

        # Initialize ONNX Runtime sessions (preferring dynamic INT8 quantized)
        self.ort_sessions: dict[str, Any] = {}
        self.use_onnx = False
        self._init_onnx_sessions()

        # Verify vision student availability
        if "vision_student" in self.ort_sessions or self.vision_student_pt is not None:
            self.trained_on_real_data = True

        self._set_eval()

    def _init_onnx_sessions(self):
        """Loads ONNX Runtime sessions (INT8 quantized preferred) for minimal memory footprint."""
        try:
            import onnxruntime as ort

            onnx_dir = settings.resolve_model_base_dir() / "onnx"
            if not onnx_dir.exists():
                return

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            candidates = {
                "spectral": ["spectral_net_quant.onnx", "spectral_net.onnx"],
                "spatial": ["spatial_net_quant.onnx", "spatial_net.onnx"],
                "temporal": ["temporal_net_quant.onnx", "temporal_net.onnx"],
                "ensemble": ["ensemble_quant.onnx", "ensemble.onnx"],
                "vision_student": ["mobilenet_v2_quant.onnx", "mobilenet_v2.onnx"],
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

        mobilenet_path = model_paths.get("mobilenet")
        if mobilenet_path:
            try:
                import torchvision.models as tv_models

                m = tv_models.mobilenet_v2(num_classes=len(PADDY_VISION_CLASSES))
                state = torch.load(
                    mobilenet_path, map_location=self.device, weights_only=True
                )
                m.load_state_dict(state)
                m.to(self.device)
                m.eval()
                self.vision_student_pt = m
                self.loaded_models.add("mobilenet")
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Could not load PyTorch mobilenet_v2 weights: %s", e
                )

    def _set_eval(self):
        self.spectral_net.eval()
        self.spatial_net.eval()
        self.temporal_net.eval()
        self.meta_learner.eval()
        if self.vision_student_pt is not None:
            self.vision_student_pt.eval()

    # ------------------------------------------------------------------
    # Vision CNN Inference (MobileNetV2 trained on 10,407 Paddy Doctor images)
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_vision_batch(crops_bgr: list[np.ndarray]) -> np.ndarray:
        tensors = []
        for crop in crops_bgr:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            tensors.append(norm.transpose(2, 0, 1))
        return np.stack(tensors, axis=0).astype(np.float32)

    def _predict_vision_batch(
        self, crops_bgr: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Runs the real Paddy Doctor MobileNetV2 vision model (ONNX INT8 preferred,
        PyTorch FP32 fallback) on a batch of BGR crops.
        Returns (logits_tensor [B, C], class_names).
        """
        batch_np = self._preprocess_vision_batch(crops_bgr)

        if "vision_student" in self.ort_sessions:
            sess = self.ort_sessions["vision_student"]
            inp_name = sess.get_inputs()[0].name
            logits_np = sess.run(None, {inp_name: batch_np})[0]
            return torch.from_numpy(logits_np).to(self.device), PADDY_VISION_CLASSES

        if self.vision_student_pt is not None:
            with torch.no_grad():
                x_t = torch.from_numpy(batch_np).to(self.device)
                logits = self.vision_student_pt(x_t)
            return logits, PADDY_VISION_CLASSES

        raise RuntimeError("No trained vision_student model available")

    def _has_vision_model(self) -> bool:
        return (
            "vision_student" in self.ort_sessions
            or self.vision_student_pt is not None
        )

    # ------------------------------------------------------------------
    # Grid feature extraction
    # ------------------------------------------------------------------
    def _extract_grid_features(
        self,
        raw_img: np.ndarray,
        processed_img: np.ndarray,
        mask: np.ndarray,
        h: int,
        w: int,
    ):
        # Adaptive grid resolution (~12x12 uniform tiles for fast real CNN patch analysis)
        grid_cols = 12
        grid_rows = max(6, min(16, int(grid_cols * (h / max(w, 1)))))
        patch_w = max(32, w // grid_cols)
        patch_h = max(32, h // grid_rows)

        regions = []
        patch_crops = []
        node_features = []
        lesion_ratios = []
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

                veg_ratio = float((patch_mask > 0).mean())
                if veg_ratio < 0.08:  # Skip non-vegetation background
                    continue

                veg_pixels = int((patch_mask > 0).sum())
                raw_crop = raw_img[y: y + ph, x: x + pw]
                region = processed_img[y: y + ph, x: x + pw]
                if region.size == 0:
                    continue

                lesion_ratio = self.processor.measure_lesion_ratio(raw_crop, patch_mask)
                lesion_ratios.append(lesion_ratio)
                patch_crops.append(raw_crop)

                # Spectral NGVC features for radar/chart telemetry & SpatialGraphNet
                ngvc = self.processor.extract_ngvc(region, bins=128)
                ngvc_accumulator += ngvc

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

        return (
            regions,
            patch_crops,
            node_features,
            lesion_ratios,
            ngvc_accumulator,
            patch_w,
            grid_rows,
            grid_cols,
        )

    def _apply_temporal_analysis(
        self,
        temporal_sequence,
        spatial_feats,
        num_nodes,
    ) -> tuple:
        """Applies temporal analysis ONLY when sequence is available."""
        if (
            ("temporal" in self.loaded_models or "temporal" in self.ort_sessions)
            and temporal_sequence
            and len(temporal_sequence) > 1
        ):
            seq_feats = []
            for frame in temporal_sequence[-5:]:
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

            seq_tensor = torch.stack(seq_feats, dim=1)
            temporal_input = seq_tensor.expand(num_nodes, -1, -1)
            if "temporal" in self.ort_sessions:
                temp_np = self.ort_sessions["temporal"].run(
                    None, {"temporal_sequence": temporal_input.cpu().numpy()}
                )[0]
                return torch.from_numpy(temp_np).to(self.device), True

            with torch.no_grad():
                return self.temporal_net(temporal_input), True

        return torch.zeros(num_nodes, 64, device=self.device), False

    @staticmethod
    def _compute_morans_i(values: list[float], adj_np: np.ndarray) -> float:
        """Compute spatial Moran's I autocorrelation across grid patch severities."""
        n = len(values)
        if n < 3:
            return 0.0
        vals = np.asarray(values, dtype=np.float64)
        mean_v = vals.mean()
        diff = vals - mean_v
        denom = float(np.sum(diff ** 2))
        w_sum = float(adj_np.sum())
        if denom < 1e-8 or w_sum < 1e-8:
            return 0.0
        num = float(diff @ adj_np @ diff)
        morans_i = (n / w_sum) * (num / denom)
        return float(np.clip(morans_i, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Field-level analysis (full Vision + Spatial + Calibration pipeline)
    # ------------------------------------------------------------------
    def analyze_field(  # noqa: C901
        self,
        image: np.ndarray,
        temporal_sequence: list[np.ndarray] | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Full adaptive grid field pathology analysis pipeline."""
        if callable(progress_callback):
            progress_callback({
                "stage": "preprocess",
                "percent": 12,
                "message": "Segmenting canopy vegetation and filtering soil/non-plant background",
            })
        h, w, _ = image.shape

        # 1. Preprocess & real vegetation check
        processed_img, mask = self.processor.remove_soil_and_shadow(image)
        veg_coverage = float((mask > 0).mean() * 100.0)

        if veg_coverage < 3.0:
            return {
                "status": "insufficient data",
                "error": (
                    f"No crop vegetation detected (foliage coverage {veg_coverage:.1f}% < 3%). "
                    "Please upload a field or canopy photograph."
                ),
            }

        if callable(progress_callback):
            progress_callback({
                "stage": "spectral",
                "percent": 30,
                "message": "Segmenting adaptive grid & extracting patch visual/spectral features",
            })

        (
            regions,
            patch_crops,
            node_features,
            lesion_ratios,
            ngvc_accumulator,
            patch_size,
            grid_rows,
            grid_cols,
        ) = self._extract_grid_features(image, processed_img, mask, h, w)

        if not node_features:
            return {
                "status": "insufficient data",
                "error": "No vegetation patches detected in the provided image.",
            }

        # Subsample uniformly if exceeding MAX_PATCHES
        if len(node_features) > MAX_PATCHES:
            step = len(node_features) / MAX_PATCHES
            indices = [int(i * step) for i in range(MAX_PATCHES)]
            node_features = [node_features[i] for i in indices]
            regions = [regions[i] for i in indices]
            patch_crops = [patch_crops[i] for i in indices]
            lesion_ratios = [lesion_ratios[i] for i in indices]

        node_features_t = torch.cat(node_features, dim=0)
        num_nodes = len(regions)

        if callable(progress_callback):
            progress_callback({
                "stage": "spatial",
                "percent": 58,
                "message": f"Running Spatial Graph analysis across {num_nodes} vegetated patches",
            })

        # 2. Spatial adjacency graph
        positions = np.array([[r["x"], r["y"]] for r in regions], dtype=np.float32)
        dist_matrix = cdist(positions, positions)
        adj_np = (dist_matrix < max(patch_size * 2.0, 48.0)).astype(np.float32)
        np.fill_diagonal(adj_np, 0.0)

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
                spatial_feats = self.spatial_net(node_features_t, adj)
        else:
            spatial_feats = torch.zeros(num_nodes, 64, device=self.device)

        # 3. Temporal analysis (only real when temporal_sequence > 1)
        temporal_feats, temporal_available = self._apply_temporal_analysis(
            temporal_sequence, spatial_feats, num_nodes
        )

        # 4. Real Vision CNN patch classification (or auxiliary ensemble fallback if no vision weights)
        if callable(progress_callback):
            progress_callback({
                "stage": "ensemble",
                "percent": 82,
                "message": "Executing calibrated Paddy Doctor vision inference across canopy patches",
            })

        temperature = self.calibration_temperature
        if self._has_vision_model():
            raw_logits, active_classes = self._predict_vision_batch(patch_crops)
            # Also classify whole canopy for global context & fuse gently with spatial patch mean
            full_logits, _ = self._predict_vision_batch([image])
            logits = raw_logits
            patch_probs = torch.softmax(logits / temperature, dim=1)
            full_probs = torch.softmax(full_logits / temperature, dim=1)[0]
            predictions = 0.70 * patch_probs.mean(dim=0) + 0.30 * full_probs
            probs = patch_probs
        else:
            active_classes = self.legacy_classes
            with torch.no_grad():
                if "ensemble" in self.ort_sessions:
                    logits_np, _ = self.ort_sessions["ensemble"].run(
                        None,
                        {
                            "spectral_feat": node_features_t.cpu().numpy(),
                            "spatial_feat": spatial_feats.cpu().numpy(),
                            "temporal_feat": temporal_feats.cpu().numpy(),
                        },
                    )
                    logits = torch.from_numpy(logits_np).to(self.device)
                else:
                    logits, _ = self.meta_learner(
                        node_features_t, spatial_feats, temporal_feats
                    )
                probs = torch.softmax(logits / temperature, dim=1)
                predictions = probs.mean(dim=0)

        pred_class = int(torch.argmax(predictions).item())
        confidence = float(predictions.max().item() * 100.0)

        top3 = torch.topk(predictions, min(3, len(active_classes)))
        alternatives = [
            {
                "disease": active_classes[i],
                "confidence": round(float(predictions[i].item() * 100.0), 2),
            }
            for i in top3.indices
        ]

        # Uncertainty decomposition
        eps = 1e-7
        k_classes = max(2, len(active_classes))
        shannon_entropy = -float(
            torch.sum(predictions * torch.log2(predictions + eps)).item()
        )
        max_entropy = math.log2(k_classes)
        norm_aleatoric = float(np.clip(shannon_entropy / max_entropy, 0.0, 1.0))

        energy_score = -temperature * float(
            torch.logsumexp(logits.mean(dim=0) / temperature, dim=0).item()
        )
        norm_epistemic = float(np.clip((energy_score + 10.0) / 10.0, 0.0, 1.0))

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

        healthy_idx = 0
        for idx, name in enumerate(active_classes):
            if name.lower() in ["normal", "healthy"]:
                healthy_idx = idx
                break

        node_pred_indices = [
            int(torch.argmax(probs[i]).item()) for i in range(num_nodes)
        ]
        infected_nodes = sum(
            1 for idx in node_pred_indices if idx != healthy_idx
        )
        infection_rate = (
            (infected_nodes / num_nodes) * 100.0 if num_nodes > 0 else 0.0
        )

        # If spatial grid detected significant infection while global mean leaned Healthy, promote top disease
        if pred_class == healthy_idx and infection_rate > 25.0:
            infected_counts: dict[int, int] = {}
            for idx in node_pred_indices:
                if idx != healthy_idx:
                    infected_counts[idx] = infected_counts.get(idx, 0) + 1
            if infected_counts:
                pred_class = max(infected_counts, key=infected_counts.get)
                confidence = float(predictions[pred_class].item() * 100.0)

        # Physical lesion-ratio + calibrated disease probability severity per patch
        node_severities = []
        for i in range(num_nodes):
            p_idx = node_pred_indices[i]
            lesion_r = lesion_ratios[i]
            if p_idx == healthy_idx:
                sev_val = float(np.clip(lesion_r * 25.0, 0.0, 12.0))
            else:
                p_prob = float(probs[i][p_idx].item())
                sev_val = float(np.clip(20.0 + lesion_r * 45.0 + p_prob * 35.0, 15.0, 95.0))
            node_severities.append(round(sev_val, 2))

        avg_severity = float(np.mean(node_severities)) if node_severities else 0.0
        spatial_morans_i = self._compute_morans_i(node_severities, adj_np)

        disease_label = active_classes[pred_class]

        # OOD guard on mean logits
        logits_mean_np = logits.mean(dim=0).cpu().numpy()
        is_ood, ood_score, ood_reason = self.ood_detector.is_ood(logits_mean_np)

        withheld = (confidence < 25.0) or is_ood
        if is_ood:
            disease_label = "Unknown"
            treatment = (
                f"OOD sample detected ({ood_reason}). "
                "The image does not match in-distribution paddy pathology patterns."
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
                alt_names = [
                    a["disease"].split(" (")[0]
                    for a in alternatives[1:3]
                    if a["disease"] != disease_label
                ]
                alt_text = f", possibly {', '.join(alt_names)}" if alt_names else ""
                treatment = (
                    f"Calibrated diagnosis: {disease_label.split(' (')[0]} ({confidence:.1f}% confidence{alt_text}). "
                    f"Field patch variation detected. Confirm via close-up leaf scan before spraying. "
                    f"ICAR Guideline: {base_treatment}"
                )
            else:
                treatment = base_treatment
            spread_risk = (
                "High"
                if (avg_severity > 40 or infection_rate > 35)
                else "Low"
            )

        nodes = [
            {
                "id": i,
                "x": reg["x"],
                "y": reg["y"],
                "w": reg["w"],
                "h": reg["h"],
                "severity": node_severities[i],
                "prediction": active_classes[node_pred_indices[i]],
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
                "disease": active_classes[node_pred_indices[i]],
            }
            for i, r in enumerate(regions)
        ]

        # Clusters summary grouped by detected non-healthy diseases
        clusters = []
        disease_counts: dict[str, int] = {}
        for idx in node_pred_indices:
            if idx != healthy_idx:
                dname = active_classes[idx]
                disease_counts[dname] = disease_counts.get(dname, 0) + 1
        for cid, (dname, cnt) in enumerate(
            sorted(disease_counts.items(), key=lambda kv: -kv[1]), start=1
        ):
            clusters.append({
                "id": cid,
                "disease": dname,
                "coverage_pct": round((cnt / max(1, num_nodes)) * 100.0, 2),
            })

        total_ngvc = ngvc_accumulator / (len(regions) + 1e-6)
        ngvc_data = [
            {"intensity": int(i * 255 / 128), "value": round(float(v), 6)}
            for i, v in enumerate(total_ngvc)
            if i % 3 == 0
        ]

        progression_data = self._build_progression(avg_severity)
        spectral_radar = self._build_spectral_radar(spatial_feats)

        # Continuous High-Resolution Gaussian Heatmap Generation
        heatmap_base64 = None
        try:
            grid_sev = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            grid_count = np.zeros((grid_rows, grid_cols), dtype=np.float32)

            for i, reg in enumerate(regions):
                r_i = min(grid_rows - 1, max(0, int(reg.get("grid_r", 0))))
                c_i = min(grid_cols - 1, max(0, int(reg.get("grid_c", 0))))
                grid_sev[r_i, c_i] += float(node_severities[i])
                grid_count[r_i, c_i] += 1.0

            valid_mask = grid_count > 0
            grid_sev[valid_mask] /= grid_count[valid_mask]

            dense_map = np.clip(
                cv2.resize(grid_sev, (w, h), interpolation=cv2.INTER_CUBIC),
                0.0,
                100.0,
            )
            dense_valid = cv2.resize(
                valid_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )

            ksize = max(21, int(max(h, w) * 0.06) | 1)
            heatmap_smooth = np.clip(
                cv2.GaussianBlur(dense_map, (ksize, ksize), 0), 0.0, 100.0
            )
            support_smooth = np.clip(
                cv2.GaussianBlur(dense_valid, (ksize, ksize), 0), 0.0, 1.0
            )

            color_pos = np.clip(
                85.0 + (heatmap_smooth / 100.0) * 165.0, 0, 255
            ).astype(np.uint8)
            colored = cv2.applyColorMap(color_pos, cv2.COLORMAP_TURBO)
            rgba = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)

            dilated_mask = cv2.dilate(
                mask, np.ones((9, 9), np.uint8), iterations=1
            )
            canopy_presence = np.maximum(
                (dilated_mask > 0).astype(np.float32),
                (support_smooth > 0.2).astype(np.float32) * support_smooth,
            )
            sev_alpha = 65.0 + (heatmap_smooth / 100.0) * 160.0
            alpha_ch = np.clip(sev_alpha * canopy_presence, 0, 225).astype(np.uint8)
            rgba[:, :, 3] = alpha_ch

            success, buf = cv2.imencode(".png", rgba)
            if success:
                heatmap_base64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as _exc:
            logging.getLogger(__name__).warning(
                "heatmap_base64 generation error: %s", _exc
            )
            heatmap_base64 = None

        if callable(progress_callback):
            progress_callback({
                "stage": "complete",
                "percent": 100,
                "message": "Field intelligence telemetry computed successfully",
            })

        return {
            "disease_type": disease_label,
            "treatment": treatment,
            "average_severity": round(avg_severity, 2),
            "confidence": round(confidence, 2),
            "withheld": withheld,
            "temporal_available": temporal_available,
            "spatial_morans_i": round(spatial_morans_i, 4),
            "alternatives": alternatives,
            "spread_risk": spread_risk,
            "clusters": clusters,
            "nodes": nodes,
            "progression_data": progression_data,
            "spectral_radar": spectral_radar,
            "ngvc_data": ngvc_data,
            "grid_data": grid_data,
            "infection_rate": round(infection_rate, 2),
            "percentage_infected_area": round(infection_rate, 2),
            "percentage_healthy_area": round(max(0.0, 100.0 - infection_rate), 2),
            "veg_coverage": round(veg_coverage, 2),
            "heatmap_base64": heatmap_base64,
            "uncertainty_breakdown": uncertainty_breakdown,
            "ood": {
                "is_ood": is_ood,
                "ood_score": round(float(ood_score), 3),
                "reason": ood_reason,
            },
            "disease_distribution": predictions.cpu().numpy().tolist(),
            "severity_map": node_severities,
            "regions": regions,
        }

    # ------------------------------------------------------------------
    # Leaf-level analysis (real Vision CNN + foliage OOD gate)
    # ------------------------------------------------------------------
    def analyze_leaf(
        self, image: np.ndarray, assess_quality: bool = False
    ) -> dict[str, Any]:
        """Micro-level leaf diagnostic using calibrated Vision CNN + OOD vegetation guard."""
        processed_img, mask = self.processor.remove_soil_and_shadow(image)
        veg_ratio = float((mask > 0).mean())
        ngvc = self.processor.extract_ngvc(processed_img, bins=128)

        # Vegetation Presence Gate — rejects non-crop OOD images (e.g. metal/gears/text/walls)
        if veg_ratio < 0.04:
            msg = (
                f"Non-crop or out-of-distribution image detected (vegetation coverage {veg_ratio * 100:.1f}% < 4%). "
                "Please upload a clear paddy leaf photograph."
            )
            return {
                "type": "Leaf Diagnostic",
                "status": "insufficient data",
                "disease_type": "Unknown",
                "treatment": msg,
                "recommendation": msg,
                "severity": 0.0,
                "confidence": 0.0,
                "withheld": True,
                "alternatives": [],
                "ngvc": ngvc.tolist(),
                "uncertainty_breakdown": {
                    "aleatoric": 1.0,
                    "epistemic": 1.0,
                    "entropy_bits": 0.0,
                    "energy_score": 0.0,
                    "calibration_temperature": self.calibration_temperature,
                },
                "ood": {
                    "is_ood": True,
                    "ood_score": 1.0,
                    "reason": f"no_crop_foliage_detected ({veg_ratio * 100:.1f}% < 4.0%)",
                },
            }

        temperature = self.calibration_temperature
        if self._has_vision_model():
            logits, active_classes = self._predict_vision_batch([image])
        else:
            active_classes = self.legacy_classes
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
                    logits_np, _ = self.ort_sessions["ensemble"].run(
                        None,
                        {
                            "spectral_feat": feat.cpu().numpy(),
                            "spatial_feat": zero_np,
                            "temporal_feat": zero_np,
                        },
                    )
                    logits = torch.from_numpy(logits_np).to(self.device)
                else:
                    logits, _ = self.meta_learner(
                        feat,
                        torch.zeros(v_b_size, 64, device=self.device),
                        torch.zeros(v_b_size, 64, device=self.device),
                    )

        probs = torch.softmax(logits / temperature, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs.max().item() * 100.0)

        top3 = torch.topk(probs, min(3, len(active_classes)))
        alternatives = [
            {
                "disease": active_classes[i],
                "confidence": round(float(probs[0][i].item() * 100.0), 2),
            }
            for i in top3.indices[0]
        ]

        eps = 1e-7
        leaf_probs = probs[0]
        shannon_entropy = -float(
            torch.sum(leaf_probs * torch.log2(leaf_probs + eps)).item()
        )
        max_entropy = math.log2(max(2, len(active_classes)))
        norm_aleatoric = float(np.clip(shannon_entropy / max_entropy, 0.0, 1.0))
        energy_score = -temperature * float(
            torch.logsumexp(logits[0] / temperature, dim=0).item()
        )
        norm_epistemic = float(np.clip((energy_score + 10.0) / 10.0, 0.0, 1.0))

        uncertainty_breakdown = {
            "aleatoric": round(norm_aleatoric, 4),
            "epistemic": round(norm_epistemic, 4),
            "entropy_bits": round(shannon_entropy, 3),
            "energy_score": round(energy_score, 3),
            "calibration_temperature": temperature,
        }

        disease_label = active_classes[pred_class]

        # OOD Detection Guard
        logits_np = logits[0].cpu().numpy()
        is_ood, ood_score, ood_reason = self.ood_detector.is_ood(logits_np)

        lesion_ratio = self.processor.measure_lesion_ratio(image, mask)
        healthy_idx = 0
        for idx, name in enumerate(active_classes):
            if name.lower() in ["normal", "healthy"]:
                healthy_idx = idx
                break
        is_healthy = pred_class == healthy_idx

        if is_healthy:
            sev = float(np.clip(lesion_ratio * 25.0, 0.0, 12.0))
        else:
            sev = float(np.clip(20.0 + lesion_ratio * 45.0 + (confidence / 100.0) * 35.0, 15.0, 95.0))

        withheld = (confidence < 25.0) or is_ood
        if is_ood:
            disease_label = "Unknown"
            status = "insufficient data"
            sev = 0.0
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
            status = "Healthy" if is_healthy else "Disease Detected"
            base_treatment = self.treatments.get(disease_label, "Monitor closely.")
            if confidence < 45.0:
                alt_names = [
                    a["disease"].split(" (")[0]
                    for a in alternatives[1:3]
                    if a["disease"] != disease_label
                ]
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
            "recommendation": treatment,
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
        k = 0.3
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
                "simulated": True,
                "point_type": "estimated_history" if i < 4 else "forecast",
            }
            for i in range(6)
        ]

    @staticmethod
    def _build_spectral_radar(spatial_feats: torch.Tensor) -> list[dict]:
        """Derive spectral fingerprint radar from spatial features."""
        feat_mean = spatial_feats.mean(dim=0).cpu().numpy()
        chunk = len(feat_mean) // 6
        subjects = SPECTRAL_SUBJECTS
        return [
            {
                "subject": subjects[i],
                "A": round(
                    float(np.abs(feat_mean[i * chunk: (i + 1) * chunk]).mean()) * 100,
                    1,
                ),
                "B": round(
                    100
                    - float(np.abs(feat_mean[i * chunk: (i + 1) * chunk]).mean()) * 80,
                    1,
                ),
                "fullMark": 100,
            }
            for i in range(6)
        ]
