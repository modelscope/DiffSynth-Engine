from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PlatformCapabilities:
    device: bool = False
    mindie: bool = False
    mindie_attention: bool = False
    mindie_compile: bool = False


class PlatformBackend(ABC):
    name = "unknown"
    device_type = "cpu"

    op_fusion = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def normalize_device(cls, device: str | torch.device) -> torch.device:
        return torch.device(device)

    @classmethod
    def get_device(cls, local_rank: int) -> torch.device:
        return torch.device(cls.device_type)

    @classmethod
    def set_device(cls, index: int | str | torch.device) -> None:
        return None

    @classmethod
    def device_count(cls) -> int:
        return 1

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn

    @classmethod
    def synchronize(cls) -> None:
        return None

    @classmethod
    def empty_cache(cls) -> None:
        return None

    @classmethod
    def pin_memory(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.pin_memory()

    @classmethod
    def distributed_backend(cls) -> str:
        return "gloo"

    @classmethod
    def compile_backend(cls) -> Any | None:
        return None

    @classmethod
    def compile_kwargs(cls) -> dict[str, Any]:
        backend = cls.compile_backend()
        return {} if backend is None else {"backend": backend}

    @classmethod
    def supports(cls, capability: str) -> bool:
        capabilities = cls.capabilities()
        if not hasattr(capabilities, capability):
            raise ValueError(f"Unknown platform capability: {capability}")
        return bool(getattr(capabilities, capability))

    @classmethod
    def capabilities(cls) -> PlatformCapabilities:
        return PlatformCapabilities(device=cls.is_available())
