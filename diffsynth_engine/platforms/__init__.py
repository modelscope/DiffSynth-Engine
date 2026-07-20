from __future__ import annotations

import platform as host_platform
from typing import Type

import torch

from .ascend import (
    AscendPlatform,
    probe_ascend_capabilities,
    probe_ascend_feature,
    reset_ascend_capability_cache,
)
from .base import PlatformBackend, PlatformCapabilities


class CPUPlatform(PlatformBackend):
    name = "cpu"
    device_type = "cpu"


class CUDAPlatform(PlatformBackend):
    name = "cuda"
    device_type = "cuda"

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def set_device(cls, index: int | str | torch.device) -> None:
        torch.cuda.set_device(index)

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


_PLATFORM_REGISTRY: dict[str, Type[PlatformBackend]] = {
    "cpu": CPUPlatform,
    "cuda": ROCmPlatform if torch.version.hip else CUDAPlatform,
    "mps": MPSPlatform,
    "npu": AscendPlatform,
}


def register_platform(device_type: str, platform_cls: Type[PlatformBackend], *, overwrite: bool = False) -> None:
    if device_type in _PLATFORM_REGISTRY and not overwrite:
        raise ValueError(f"Platform for device type {device_type!r} is already registered")
    _PLATFORM_REGISTRY[device_type] = platform_cls


def get_device_type(device: str | torch.device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0].lower()


def resolve_platform(device: str | torch.device) -> Type[PlatformBackend]:
    device_type = get_device_type(device)
    try:
        return _PLATFORM_REGISTRY[device_type]
    except KeyError as exc:
        available = ", ".join(sorted(_PLATFORM_REGISTRY))
        raise ValueError(f"Unsupported device type {device_type!r}. Registered device types: {available}") from exc


def get_preferred_fp8_dtype(device: str | torch.device = "cuda") -> torch.dtype:
    platform_cls = resolve_platform(device)
    if platform_cls is ROCmPlatform and platform_cls.is_available():
        properties = torch.cuda.get_device_properties(0)
        if "gfx94" in properties.gcnArchName:
            return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def pin_memory(tensor: torch.Tensor, device: str | torch.device | None = None) -> torch.Tensor:
    if host_platform.system() != "Linux":
        return tensor
    platform_cls = resolve_platform(device or "cuda")
    return platform_cls.pin_memory(tensor)


__all__ = [
    "AscendPlatform",
    "CPUPlatform",
    "CUDAPlatform",
    "MPSPlatform",
    "PlatformBackend",
    "PlatformCapabilities",
    "ROCmPlatform",
    "get_device_type",
    "get_preferred_fp8_dtype",
    "pin_memory",
    "probe_ascend_capabilities",
    "probe_ascend_feature",
    "register_platform",
    "reset_ascend_capability_cache",
    "resolve_platform",
]
