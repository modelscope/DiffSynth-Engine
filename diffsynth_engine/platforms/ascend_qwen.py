from __future__ import annotations

import importlib
from typing import Any

import torch.nn as nn

from diffsynth_engine.configs import QuantizationConfig

from .runtime import ModelRuntimeAdapter


class MindIEFP8AttentionProcessor:
    def __init__(self, fa_quant):
        self.fa_quant = fa_quant

    def __call__(self, query, key, value, *, attn_mask=None, **kwargs):
        if attn_mask is not None:
            raise RuntimeError("MindIE FP8 attention does not support Qwen entity attention masks")
        return self.fa_quant(query, key, value, layout="BSND")


class AscendQwenImageRuntimeAdapter(ModelRuntimeAdapter):
    def __init__(self, platform, model_family: str):
        super().__init__(platform, model_family)
        self._quantization_enabled = False

    @staticmethod
    def _quantization_config(config: Any) -> QuantizationConfig:
        quantization = getattr(config, "quantization", None)
        if quantization is None and getattr(config, "use_fp8_linear", False):
            quantization = QuantizationConfig(linear="fp8")
        return quantization or QuantizationConfig()

    @classmethod
    def _uses_quantization(cls, config: Any) -> bool:
        quantization = cls._quantization_config(config)
        return quantization.linear != "none" or quantization.attention != "none"

    def _supports(self, capability: str) -> bool:
        supports = getattr(self.platform, "supports", None)
        if callable(supports):
            return supports(capability)
        return bool(getattr(self.platform.capabilities(), capability))

    def validate_config(self, config: Any) -> None:
        if not self._supports("device"):
            raise RuntimeError(
                "Ascend device requested, but no available NPU was detected. Check torch_npu and CANN installation."
            )
        if getattr(config, "parallelism", 1) > 1:
            raise RuntimeError("Multi-NPU execution is not supported in the first Ascend release")

        attn_impl = getattr(getattr(config, "dit_attn_impl", "auto"), "value", "auto")
        if attn_impl == "mindie" and not self._supports("mindie_attention"):
            raise RuntimeError(
                "MindIE attention was explicitly requested, but the installed MindIE-SD package "
                "does not provide a compatible attention_forward implementation"
            )

        quantization = self._quantization_config(config)
        uses_quantization = self._uses_quantization(config)
        if quantization.backend == "nunchaku" or getattr(config, "use_nunchaku", False):
            raise RuntimeError(
                "Nunchaku SVDQ/AWQ checkpoints contain CUDA-specific packed weights and cannot run on Ascend. "
                "Use an unquantized Qwen-Image checkpoint with QuantizationConfig(backend='mindie', ...)."
            )
        if quantization.backend not in {"auto", "native", "mindie"}:
            raise ValueError(f"Unsupported Ascend quantization backend: {quantization.backend}")

        use_compile = getattr(config, "use_torch_compile", False)
        offload_mode = getattr(config, "offload_mode", None)
        if use_compile and offload_mode not in {None, "disable"}:
            raise ValueError("MindIE-SD compilation cannot be combined with CPU offload")
        if uses_quantization and offload_mode not in {None, "disable"}:
            raise ValueError("MindIE-SD native quantization cannot be combined with CPU offload")
        if uses_quantization and use_compile:
            raise ValueError("MindIE-SD native quantization and compilation are not supported together")

        if use_compile and not self._supports("mindie_compile"):
            raise RuntimeError("MindIE-SD compilation support is unavailable in the current Ascend environment")
        if quantization.linear == "fp8" and not self._supports("mindie_mxfp8_linear"):
            raise RuntimeError("MindIE-SD MXFP8 linear support is unavailable on this Ascend runtime")
        if quantization.linear == "int4" and not self._supports("mindie_w4a4_linear"):
            raise RuntimeError("MindIE-SD W4A4 linear support is unavailable on this Ascend runtime")
        if quantization.attention == "fp8" and not self._supports("mindie_fp8_attention"):
            raise RuntimeError("MindIE-SD FP8 attention support is unavailable on this Ascend runtime")

    def prepare_component(self, component_name: str, module: nn.Module, config: Any) -> nn.Module:
        if component_name != "dit" or not self._uses_quantization(config):
            return module

        quantization = self._quantization_config(config)
        quantization_module = importlib.import_module("mindiesd.quantization")
        quant_mode_module = importlib.import_module("mindiesd.quantization.mode")
        QuantAlgorithm = quant_mode_module.QuantAlgorithm

        if quantization.linear == "fp8":
            quant_type = QuantAlgorithm.W8A8_MXFP8
            fallback_layers = None
        elif quantization.linear == "int4":
            quant_type = QuantAlgorithm.W4A4_MXFP4_DYNAMIC
            fallback_layers = None
        else:
            quant_type = QuantAlgorithm.W8A8_DYNAMIC
            fallback_layers = {
                name: QuantAlgorithm.W16A16
                for name, layer in module.named_modules()
                if isinstance(layer, nn.Linear)
            }

        fa_quant_type = QuantAlgorithm.FP8_DYNAMIC if quantization.attention == "fp8" else None
        online_config = quantization_module.OnlineQuantConfig(
            quant_type=quant_type,
            fallback_layers=fallback_layers,
            fa_layers=("transformer_blocks.*.attn", "QwenDoubleStreamAttention") if fa_quant_type else None,
            fa_quant_type=fa_quant_type,
        )
        module = quantization_module.quantize(module, online_config=online_config, dtype=config.model_dtype)
        if fa_quant_type is not None:
            for layer in module.modules():
                set_processor = getattr(layer, "set_attention_processor", None)
                fa_quant = getattr(layer, "fa_quant", None)
                if callable(set_processor) and fa_quant is not None:
                    set_processor(MindIEFP8AttentionProcessor(fa_quant))
        self._quantization_enabled = True
        return module.eval()

    def before_denoise_step(self, step_index: int) -> None:
        if not self._quantization_enabled:
            return
        quantization_module = importlib.import_module("mindiesd.quantization")
        quantization_module.TimestepManager.set_timestep_idx(step_index)

    def validate_dynamic_weights(self, config: Any) -> None:
        if getattr(config, "use_torch_compile", False):
            raise ValueError("Dynamic LoRA/ControlNet loading is not supported with MindIE-SD compilation")
        if self._uses_quantization(config):
            raise ValueError("Dynamic LoRA/ControlNet loading is not supported with MindIE-SD native quantization")


__all__ = ["AscendQwenImageRuntimeAdapter", "MindIEFP8AttentionProcessor"]
