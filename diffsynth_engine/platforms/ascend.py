from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any

import torch

from .base import PlatformBackend, PlatformCapabilities
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


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




_OPERATION_PROBES = {
    "mindie_attention": _probe_mindie_attention_operation,
    "mindie_compile": _probe_mindie_compile_operation,
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
    )


def reset_ascend_capability_cache() -> None:
    _probe_ascend_device.cache_clear()
    _probe_mindie_installation.cache_clear()
    probe_ascend_feature.cache_clear()


class AscendPlatform(PlatformBackend):
    name = "ascend"
    device_type = "npu"

    # Ascend-specific tuning knobs (seeded from environment for now).
    op_fusion = os.environ.get("USE_MINDIESD_FUSE", "False").lower() == "true"
    fa_alltoall_overlap = int(os.environ.get("FA_ALLTOALL_OVERLAP", 1))
    fa_alltoall_cut = int(os.environ.get("FA_ALLTOALL_CUT", 1))

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
    def device_count(cls) -> int:
        _import_torch_npu()
        return torch.npu.device_count()

    @classmethod
    def get_device(cls, local_rank: int) -> torch.device:
        return torch.device(cls.device_type, local_rank)

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
    def compile_backend(cls) -> Any | None:
        if not cls.supports("mindie_compile"):
            logger.warning(
                "MindIE-SD compile backend is unavailable; falling back to default torch.compile backend"
            )
            return None
        from mindiesd.compilation import CompilationConfig, MindieSDBackend

        CompilationConfig.fusion_patterns.enable_fast_gelu = False
        return MindieSDBackend()

    @classmethod
    def capabilities(cls) -> PlatformCapabilities:
        return probe_ascend_capabilities()

    @classmethod
    def supports(cls, capability: str) -> bool:
        return probe_ascend_feature(capability)
