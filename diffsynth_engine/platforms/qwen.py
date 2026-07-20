from __future__ import annotations

from typing import Any

import torch.nn as nn

from diffsynth_engine.configs import QuantizationConfig
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear

from .runtime import ModelRuntimeAdapter


class CudaQwenImageRuntimeAdapter(ModelRuntimeAdapter):
    @staticmethod
    def _quantization_config(config: Any) -> QuantizationConfig | None:
        return getattr(config, "quantization", None)

    def validate_config(self, config: Any) -> None:
        quantization = self._quantization_config(config)
        if quantization is None:
            return
        if quantization.backend == "mindie":
            raise ValueError("MindIE quantization requires an Ascend NPU device")
        if quantization.attention != "none":
            raise ValueError(
                "QuantizationConfig attention quantization is currently Ascend-only; "
                "use AttnImpl.FA3_FP8 or AttnImpl.AITER_FP8 on CUDA/ROCm"
            )
        if quantization.backend == "nunchaku":
            if not getattr(config, "use_nunchaku", False):
                raise ValueError("Nunchaku quantization requires a Nunchaku-packed Qwen checkpoint")
            return
        if quantization.linear == "int4":
            raise ValueError("INT4 Qwen execution on CUDA/ROCm requires a Nunchaku-packed checkpoint")

    def prepare_component(self, component_name: str, module: nn.Module, config: Any) -> nn.Module:
        if component_name != "dit" or getattr(config, "use_nunchaku", False):
            return module

        quantization = self._quantization_config(config)
        use_fp8_linear = getattr(config, "use_fp8_linear", False) if quantization is None else False
        if quantization is not None:
            use_fp8_linear = quantization.linear == "fp8" and quantization.backend in {"auto", "native"}
        if use_fp8_linear:
            enable_fp8_linear(module)
        return module


__all__ = ["CudaQwenImageRuntimeAdapter"]
