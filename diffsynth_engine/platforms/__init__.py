from __future__ import annotations

from functools import lru_cache

import torch

from diffsynth_engine.registry import get_platform_class

from .base import PlatformBackend


class CPUPlatform(PlatformBackend):
    name = "cpu"
    device_type = "cpu"

    @classmethod
    def pin_memory(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @classmethod
    def distributed_backend(cls) -> str:
        raise NotImplementedError("Unsupported device type")


class CUDAPlatform(PlatformBackend):
    name = "cuda"
    device_type = "cuda"

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def get_device(cls, local_rank: int) -> torch.device:
        return torch.device(cls.device_type, local_rank)

    @classmethod
    def set_device(cls, index: int | str | torch.device) -> None:
        torch.cuda.set_device(index)

    @classmethod
    def device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        torch.cuda.empty_cache()

    @classmethod
    def distributed_backend(cls) -> str:
        return "nccl"


class ROCmPlatform(CUDAPlatform):
    name = "rocm"

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if not cls.is_available():
            return torch.float8_e4m3fn
        properties = torch.cuda.get_device_properties(0)
        if "gfx94" in properties.gcnArchName:
            return torch.float8_e4m3fnuz
        return torch.float8_e4m3fn


class MPSPlatform(PlatformBackend):
    name = "mps"
    device_type = "mps"

    @classmethod
    def is_available(cls) -> bool:
        return torch.backends.mps.is_available()

    @classmethod
    def synchronize(cls) -> None:
        torch.mps.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        torch.mps.empty_cache()

    @classmethod
    def pin_memory(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


_platform: type[PlatformBackend] | None = None


@lru_cache(maxsize=1)
def auto_detect_device() -> str:
    """Auto detect device in order of CUDA/ROCm, NPU, MPS, CPU."""
    for device_type in ("cuda", "npu", "mps", "cpu"):
        try:
            if get_platform_class(device_type).is_available():
                return device_type
        except Exception:
            continue
    raise RuntimeError("No available platform found.")


def get_device_type(device: str | torch.device | None = None) -> str:
    if device is None or (isinstance(device, str) and device.lower() in ("auto", "")):
        return auto_detect_device()
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0].lower()


def get_platform() -> type[PlatformBackend]:
    if _platform is None:
        raise RuntimeError("Platform must be initialized with set_platform() before use.")
    return _platform


def set_platform(device: str | torch.device | None = None):
    global _platform
    device_type = get_device_type(device)
    platform = get_platform_class(device_type)
    if not platform.is_available():
        raise RuntimeError(f"Device type {device_type!r} is not available.")
    if _platform is not None and _platform is not platform:
        raise RuntimeError(f"Platform is already set to {_platform.name!r}; cannot switch to {platform.name!r}.")
    _platform = platform


def is_npu_available() -> bool:
    return get_platform_class("npu").is_available()


def is_mindie_sd_available() -> bool:
    from .ascend import is_mindie_sd_available as ascend_mindie_sd_available

    return ascend_mindie_sd_available()


def get_device(local_rank: int) -> torch.device:
    return get_platform().get_device(local_rank)


def set_device(index: int | str | torch.device) -> None:
    get_platform().set_device(index)


def device_count() -> int:
    return get_platform().device_count()


def get_fp8_dtype() -> torch.dtype:
    return get_platform().fp8_dtype()


def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    return get_platform().pin_memory(tensor)


def get_torch_distributed_backend() -> str:
    return get_platform().distributed_backend()


def get_compile_kwargs() -> dict:
    return get_platform().compile_kwargs()


DTYPE_MAP: dict[str, torch.dtype] = {
    # Integer dtypes
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "bool": torch.bool,
    # Floating dtypes
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    # Complex dtypes
    "complex64": torch.complex64,
    "cfloat": torch.complex64,
    "complex128": torch.complex128,
    "cdouble": torch.complex128,
    # Quantized dtypes
    "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "float8_e5m2": getattr(torch, "float8_e5m2", None),
    "float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
    "float8_e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None),
    "float8_e8m0fnu": getattr(torch, "float8_e8m0fnu", None),
    "float4_e2m1fn_x2": getattr(torch, "float4_e2m1fn_x2", None),
}
DTYPE_MAP = {k: v for k, v in DTYPE_MAP.items() if v is not None}


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype)[6:]  # remove "torch." prefix


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]

    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype string: {dtype_str}.")

    return DTYPE_MAP[dtype_str]


__all__ = [
    "PlatformBackend",
    "CPUPlatform",
    "CUDAPlatform",
    "ROCmPlatform",
    "MPSPlatform",
    "auto_detect_device",
    "get_device_type",
    "get_platform",
    "set_platform",
    "is_npu_available",
    "is_mindie_sd_available",
    "get_device",
    "set_device",
    "device_count",
    "get_fp8_dtype",
    "pin_memory",
    "get_torch_distributed_backend",
    "get_compile_kwargs",
    "DTYPE_MAP",
    "torch_dtype_to_str",
    "str_to_torch_dtype",
]
