import logging
import os

import torch

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Handles exporting the PyTorch model to ONNX or TorchScript
    for production deployment.
    """

    def __init__(self, export_dir: str):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export_onnx(
        self,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        filename: str = "model.onnx",
    ):
        path = os.path.join(self.export_dir, filename)
        model.eval()

        logger.info(f"Exporting model to ONNX format at {path}...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits", "severity"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                    "severity": {0: "batch_size"},
                },
            )
            logger.info("ONNX export successful.")
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")
