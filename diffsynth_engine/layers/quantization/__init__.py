"""Quantization layers for the Ascend NPU backend."""

from diffsynth_engine.layers.quantization.fp8_linear import (
    enable_fp8_linear,
    is_quantized,
    npu_quant_linear,
)

__all__ = ["enable_fp8_linear", "is_quantized", "npu_quant_linear"]
