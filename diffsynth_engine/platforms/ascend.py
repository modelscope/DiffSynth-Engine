from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any

import torch

from diffsynth_engine.utils import logging

from .base import PlatformBackend

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


@lru_cache(maxsize=1)
def _probe_npu_runtime(npu: Any) -> bool:
    try:
        probe = torch.zeros(1, device="npu:0")
        probe.add_(1)
        npu.synchronize()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_mindie_sd_available() -> bool:
    if not AscendPlatform.is_available():
        return False
    try:
        _import_mindie_sd()
    except Exception:
        return False
    return True


def _probe_mindie_attention(attention_forward: Any) -> bool:
    try:
        query = torch.randn(1, 128, 8, 128, device="npu:0", dtype=torch.bfloat16)
        with torch.no_grad():
            output = attention_forward(
                query=query, key=query, value=query, attn_mask=None, scale=None, fused=True, head_first=False
            )
        if output.shape != query.shape:
            return False
        _import_torch_npu().npu.synchronize()
        return bool(torch.isfinite(output).all().cpu().item())
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_mindie_attention_available() -> bool:
    if not is_mindie_sd_available():
        return False
    try:
        module = importlib.import_module("mindiesd.layers.flash_attn.attention_forward")
        entry = getattr(module, "attention_forward", None)
        return callable(entry) and _probe_mindie_attention(entry)
    except Exception:
        return False


def _probe_mindie_compile(backend_class: Any) -> bool:
    try:

        def probe_fn(value):
            return torch.nn.functional.gelu(value + 1)

        compiled_fn = torch.compile(probe_fn, backend=backend_class(), fullgraph=False)
        value = torch.randn(8, 32, device="npu:0", dtype=torch.bfloat16)
        with torch.no_grad():
            output = compiled_fn(value)
        if output.shape != value.shape:
            return False
        _import_torch_npu().npu.synchronize()
        return bool(torch.isfinite(output).all().cpu().item())
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_mindie_compile_available() -> bool:
    if not is_mindie_sd_available():
        return False
    try:
        module = importlib.import_module("mindiesd.compilation")
        entry = getattr(module, "MindieSDBackend", None)
        return callable(entry) and _probe_mindie_compile(entry)
    except Exception:
        return False


class AscendPlatform(PlatformBackend):
    name = "ascend"
    device_type = "npu"

    # Ascend-specific tuning knobs (seeded from environment for now).
    op_fusion = os.environ.get("USE_MINDIESD_FUSE", "False").lower() == "true"
    fa_alltoall_overlap = int(os.environ.get("FA_ALLTOALL_OVERLAP", 1))
    fa_alltoall_cut = int(os.environ.get("FA_ALLTOALL_CUT", 1))

    @classmethod
    @lru_cache(maxsize=1)
    def is_available(cls) -> bool:
        try:
            torch_npu = _import_torch_npu()
            npu = getattr(torch_npu, "npu", getattr(torch, "npu", None))
            if npu is None or not callable(getattr(npu, "is_available", None)) or not npu.is_available():
                return False
            return _probe_npu_runtime(npu)
        except Exception:
            return False

    @classmethod
    def normalize_device(cls, device: str | torch.device) -> torch.device:
        _import_torch_npu()
        return torch.device(device)

    @classmethod
    def get_device(cls, local_rank: int) -> torch.device:
        return torch.device(cls.device_type, local_rank)

    @classmethod
    def set_device(cls, index: int | str | torch.device) -> None:
        torch_npu = _import_torch_npu()
        torch_npu.npu.set_device(index)

    @classmethod
    def device_count(cls) -> int:
        _import_torch_npu()
        return torch.npu.device_count()

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
        if not is_mindie_compile_available():
            logger.warning("MindIE-SD compile backend is unavailable; falling back to default torch.compile backend")
            return None
        from mindiesd.compilation import CompilationConfig, MindieSDBackend

        CompilationConfig.fusion_patterns.enable_fast_gelu = False
        return MindieSDBackend()
