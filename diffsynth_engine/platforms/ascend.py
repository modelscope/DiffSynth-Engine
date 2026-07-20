from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn

from .base import PlatformBackend, PlatformCapabilities


def _import_torch_npu():
    try:
        return importlib.import_module("torch_npu")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Ascend device requested, but torch_npu is not installed. "
            "Install the torch_npu wheel matching the PyTorch and CANN versions."
        ) from exc


def _import_mindie_sd():
    try:
        return importlib.import_module("mindiesd")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "This Ascend feature requires MindIE-SD. Install a MindIE-SD 3.x wheel "
            "matching the current torch_npu and CANN versions."
        ) from exc


def _has_callable(obj: Any, name: str) -> bool:
    return callable(getattr(obj, name, None))


def _probe_npu_runtime(npu: Any) -> bool:
    try:
        probe = torch.zeros(1, device="npu:0")
        probe.add_(1)
        npu.synchronize()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _probe_ascend_device() -> bool:
    try:
        torch_npu = _import_torch_npu()
        npu = getattr(torch_npu, "npu", getattr(torch, "npu", None))
        device_available = bool(npu is not None and _has_callable(npu, "is_available") and npu.is_available())
    except Exception:
        return False

    if device_available:
        # is_available() can stay true while ACL initialization is failing.
        device_available = _probe_npu_runtime(npu)
    return device_available


@lru_cache(maxsize=1)
def _probe_mindie_installation() -> bool:
    if not _probe_ascend_device():
        return False

    try:
        _import_mindie_sd()
    except Exception:
        return False
    return True


def _feature_api_available(feature: str) -> bool:
    if feature == "mindie_attention":
        module = importlib.import_module("mindiesd.layers.flash_attn.attention_forward")
        return _has_callable(module, "attention_forward")
    if feature == "mindie_compile":
        module = importlib.import_module("mindiesd.compilation")
        return callable(getattr(module, "MindieSDBackend", None))

    quantization_module = importlib.import_module("mindiesd.quantization")
    quant_mode_module = importlib.import_module("mindiesd.quantization.mode")
    quant_algorithm = getattr(quant_mode_module, "QuantAlgorithm", None)
    if not all(
        (
            _has_callable(quantization_module, "quantize"),
            getattr(quantization_module, "OnlineQuantConfig", None) is not None,
            quant_algorithm is not None,
        )
    ):
        return False
    if feature == "mindie_mxfp8_linear":
        return hasattr(quant_algorithm, "W8A8_MXFP8")
    if feature == "mindie_w4a4_linear":
        return hasattr(quant_algorithm, "W4A4_MXFP4_DYNAMIC")
    if feature == "mindie_fp8_attention":
        quant_layer_module = importlib.import_module("mindiesd.quantization.layer")
        return hasattr(quant_algorithm, "FP8_DYNAMIC") and getattr(
            quant_layer_module, "FP8RotateQuantFA", None
        ) is not None
    raise ValueError(f"Unknown Ascend capability: {feature}")


def _tensor_probe_succeeded(output: torch.Tensor) -> bool:
    torch_npu = _import_torch_npu()
    torch_npu.npu.synchronize()
    return bool(torch.isfinite(output).all().cpu().item())


def _probe_mindie_attention_operation() -> bool:
    module = importlib.import_module("mindiesd.layers.flash_attn.attention_forward")
    query = torch.randn(1, 128, 8, 128, device="npu:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = module.attention_forward(
            query=query,
            key=query,
            value=query,
            attn_mask=None,
            scale=None,
            fused=True,
            head_first=False,
        )
    return output.shape == query.shape and _tensor_probe_succeeded(output)


def _probe_mindie_compile_operation() -> bool:
    compilation_module = importlib.import_module("mindiesd.compilation")

    def probe_fn(value):
        return torch.nn.functional.gelu(value + 1)

    compiled_fn = torch.compile(probe_fn, backend=compilation_module.MindieSDBackend(), fullgraph=False)
    value = torch.randn(8, 32, device="npu:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = compiled_fn(value)
    return output.shape == value.shape and _tensor_probe_succeeded(output)


def _probe_mindie_linear_operation(algorithm_name: str) -> bool:
    quantization_module = importlib.import_module("mindiesd.quantization")
    quant_mode_module = importlib.import_module("mindiesd.quantization.mode")
    algorithm = getattr(quant_mode_module.QuantAlgorithm, algorithm_name)
    model = nn.Sequential(nn.Linear(256, 256, device="npu:0", dtype=torch.bfloat16)).eval()
    model = quantization_module.quantize(
        model,
        online_config=quantization_module.OnlineQuantConfig(quant_type=algorithm),
        dtype=torch.bfloat16,
    )
    value = torch.randn(1, 256, device="npu:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = model(value)
    return output.shape == value.shape and _tensor_probe_succeeded(output)


def _probe_mindie_fp8_attention_operation() -> bool:
    quantization_module = importlib.import_module("mindiesd.quantization")
    quant_mode_module = importlib.import_module("mindiesd.quantization.mode")

    class ProbeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 128
            self.register_buffer("_device_anchor", torch.empty(0, device="npu:0"))

    model = nn.ModuleDict({"attn": ProbeAttention()}).eval()
    model = quantization_module.quantize(
        model,
        online_config=quantization_module.OnlineQuantConfig(
            quant_type=quant_mode_module.QuantAlgorithm.W8A8_DYNAMIC,
            fa_layers=("ProbeAttention",),
            fa_quant_type=quant_mode_module.QuantAlgorithm.FP8_DYNAMIC,
        ),
        dtype=torch.bfloat16,
    )
    query = torch.randn(1, 128, 8, 128, device="npu:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = model["attn"].fa_quant(query, query, query, layout="BSND")
    return output.shape == query.shape and _tensor_probe_succeeded(output)


_OPERATION_PROBES = {
    "mindie_attention": _probe_mindie_attention_operation,
    "mindie_compile": _probe_mindie_compile_operation,
    "mindie_mxfp8_linear": lambda: _probe_mindie_linear_operation("W8A8_MXFP8"),
    "mindie_w4a4_linear": lambda: _probe_mindie_linear_operation("W4A4_MXFP4_DYNAMIC"),
    "mindie_fp8_attention": _probe_mindie_fp8_attention_operation,
}


@lru_cache(maxsize=None)
def probe_ascend_feature(feature: str) -> bool:
    if feature == "device":
        return _probe_ascend_device()
    if feature == "mindie":
        return _probe_mindie_installation()
    if feature not in _OPERATION_PROBES:
        raise ValueError(f"Unknown Ascend capability: {feature}")
    if not _probe_mindie_installation():
        return False

    try:
        if not _feature_api_available(feature):
            return False
        return bool(_OPERATION_PROBES[feature]())
    except Exception:
        return False


def probe_ascend_capabilities() -> PlatformCapabilities:
    device = probe_ascend_feature("device")
    if not device:
        return PlatformCapabilities()
    mindie = probe_ascend_feature("mindie")
    if not mindie:
        return PlatformCapabilities(device=True)

    return PlatformCapabilities(
        device=True,
        mindie=True,
        mindie_attention=probe_ascend_feature("mindie_attention"),
        mindie_compile=probe_ascend_feature("mindie_compile"),
        mindie_mxfp8_linear=probe_ascend_feature("mindie_mxfp8_linear"),
        mindie_w4a4_linear=probe_ascend_feature("mindie_w4a4_linear"),
        mindie_fp8_attention=probe_ascend_feature("mindie_fp8_attention"),
    )


def reset_ascend_capability_cache() -> None:
    _probe_ascend_device.cache_clear()
    _probe_mindie_installation.cache_clear()
    probe_ascend_feature.cache_clear()


class AscendPlatform(PlatformBackend):
    name = "ascend"
    device_type = "npu"

    @classmethod
    def is_available(cls) -> bool:
        return probe_ascend_feature("device")

    @classmethod
    def normalize_device(cls, device: str | torch.device) -> torch.device:
        _import_torch_npu()
        return torch.device(device)

    @classmethod
    def set_device(cls, index: int | str | torch.device) -> None:
        torch_npu = _import_torch_npu()
        torch_npu.npu.set_device(index)

    @classmethod
    def synchronize(cls) -> None:
        torch_npu = _import_torch_npu()
        torch_npu.npu.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        torch_npu = _import_torch_npu()
        torch_npu.npu.empty_cache()

    @classmethod
    def pin_memory(cls, tensor: torch.Tensor) -> torch.Tensor:
        _import_torch_npu()
        try:
            return tensor.pin_memory(device="npu")
        except (RuntimeError, TypeError):
            # Pageable CPU memory is slower but remains correct on runtimes that
            # do not expose an NPU-specific pinned allocator.
            return tensor

    @classmethod
    def distributed_backend(cls) -> str:
        return "hccl"

    @classmethod
    def compile_backend(cls):
        if not cls.supports("mindie_compile"):
            raise RuntimeError(
                "MindIE-SD compilation was requested, but MindieSDBackend is unavailable "
                "in the installed MindIE-SD package."
            )
        compilation_module = importlib.import_module("mindiesd.compilation")
        return compilation_module.MindieSDBackend()

    @classmethod
    def capabilities(cls) -> PlatformCapabilities:
        return probe_ascend_capabilities()

    @classmethod
    def supports(cls, capability: str) -> bool:
        return probe_ascend_feature(capability)
