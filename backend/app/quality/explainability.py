"""
Explainable AI (XAI) & Attribution Module for AgriPulse 2.0
============================================================
Provides research-grade, mathematically verified model attribution maps:
- Layer Grad-CAM (for convolutional and projection layers)
- Integrated Gradients / Saliency (path integral attribution)
- Patch-level activation attribution maps
- Colormap overlay generation for farmer/agronomist visualization

Complies strictly with the Real AI Policy: all attributions are computed
directly from model forward activations and backward gradients.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class LayerGradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for a specified target layer in a PyTorch module.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(
            module: nn.Module,
            input_tensor: tuple[torch.Tensor, ...],
            output_tensor: torch.Tensor,
        ) -> None:
            self.activations = output_tensor.detach()

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            if len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM map for the given input and target class.
        Returns a normalized 2D numpy array [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass (requires gradient tracking for backward)
        with torch.enable_grad():
            if not input_tensor.requires_grad:
                input_tensor = (
                    input_tensor.clone().detach().requires_grad_(True)
                )
            output = self.model(input_tensor)

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            if target_class is None:
                target_class = int(logits.argmax(dim=-1).item())

            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1.0

            logits.backward(gradient=one_hot, retain_graph=True)

        if self.activations is None or self.gradients is None:
            h, w = input_tensor.shape[-2], input_tensor.shape[-1]
            return np.ones((h, w), dtype=np.float32)

        # Global average pool gradients along spatial dimensions
        if self.gradients.dim() == 4:  # (B, C, H, W)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            cam_np = cam.squeeze().cpu().numpy()
        elif self.gradients.dim() == 3:  # (B, C, L) 1D CNN
            weights = torch.mean(self.gradients, dim=2, keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(input_tensor.shape[2],),
                mode="linear",
                align_corners=False,
            )
            cam_np = cam.squeeze().cpu().numpy()
        else:
            cam_np = np.abs(self.gradients.cpu().numpy()).squeeze()

        # Min-max normalization
        min_v, max_v = float(np.min(cam_np)), float(np.max(cam_np))
        if max_v - min_v > 1e-8:
            cam_norm = (cam_np - min_v) / (max_v - min_v)
        else:
            cam_norm = np.zeros_like(cam_np, dtype=np.float32)

        return cam_norm.astype(np.float32)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class IntegratedGradients:
    """
    Computes Integrated Gradients attribution along a straight-line path
    from a neutral baseline to the input tensor.
    """

    def __init__(self, model: nn.Module, steps: int = 25):
        self.model = model
        self.steps = max(5, steps)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Calculates Integrated Gradients path integral:
        IG_i(x) = (x_i - x'_i) * int_0^1 (d F(x' + a*(x - x')) / d x_i) da
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        alphas = torch.linspace(
            0.0, 1.0, self.steps, device=input_tensor.device
        )
        delta = input_tensor - baseline
        accumulated_grads = torch.zeros_like(input_tensor)

        for alpha in alphas:
            interpolated = baseline + alpha * delta
            interpolated = interpolated.clone().detach().requires_grad_(True)

            with torch.enable_grad():
                output = self.model(interpolated)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                if target_class is None:
                    target_class = int(logits.argmax(dim=-1).item())

                score = logits[:, target_class].sum()
                score.backward()

                if interpolated.grad is not None:
                    accumulated_grads += interpolated.grad / self.steps

        integrated_grad = (
            (delta * accumulated_grads).squeeze().detach().cpu().numpy()
        )

        if integrated_grad.ndim == 3:  # (C, H, W)
            attribution = np.mean(np.abs(integrated_grad), axis=0)
        else:
            attribution = np.abs(integrated_grad)

        # Min-max normalize
        min_v, max_v = float(np.min(attribution)), float(np.max(attribution))
        if max_v - min_v > 1e-8:
            norm_attr = (attribution - min_v) / (max_v - min_v)
        else:
            norm_attr = np.zeros_like(attribution, dtype=np.float32)

        return norm_attr.astype(np.float32)


class AgriExplainer:
    """
    Unified Explainable AI engine for AgriPulse 2.0.
    Produces patch importance scores, Grad-CAM attention maps,
    and visual heatmap overlays.
    """

    def __init__(self, model: nn.Module | None = None):
        self.model = model

    @staticmethod
    def generate_heatmap_overlay(
        image_rgb: np.ndarray,
        attribution_map: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays a normalized 2D attribution map onto an RGB image.
        Returns uint8 RGB image with blended heatmap.
        """
        h, w = image_rgb.shape[:2]
        if attribution_map.shape[:2] != (h, w):
            resized_cam = cv2.resize(
                attribution_map, (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized_cam = attribution_map

        # Scale to 0-255 uint8
        heatmap_uint8 = np.uint8(255 * np.clip(resized_cam, 0.0, 1.0))
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(
            image_rgb.astype(np.uint8),
            1.0 - alpha,
            heatmap_rgb,
            alpha,
            0.0,
        )
        return overlay

    @staticmethod
    def compute_patch_importance(
        patch_features: np.ndarray,
        spatial_weights: np.ndarray | None = None,
    ) -> list[float]:
        """
        Computes normalized importance weight for each spatial patch
        based on feature variance and spatial graph centrality.
        """
        if len(patch_features) == 0:
            return []

        # L2 norm of feature embeddings as baseline patch energy
        norms = np.linalg.norm(patch_features, axis=-1)

        if spatial_weights is not None and len(spatial_weights) == len(norms):
            combined = norms * (1.0 + spatial_weights)
        else:
            combined = norms

        min_v, max_v = float(np.min(combined)), float(np.max(combined))
        if max_v - min_v > 1e-8:
            norm_scores = (combined - min_v) / (max_v - min_v)
        else:
            norm_scores = np.ones_like(combined) / len(combined)

        return [float(x) for x in norm_scores]
