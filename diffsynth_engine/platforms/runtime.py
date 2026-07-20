from __future__ import annotations

import importlib
from typing import Any, Type

import torch.nn as nn

from .base import PlatformBackend
from . import resolve_platform


class ModelRuntimeAdapter:
    def __init__(self, platform: Type[PlatformBackend], model_family: str):
        self.platform = platform
        self.model_family = model_family

    def validate_config(self, config: Any) -> None:
        return None

    def prepare_component(self, component_name: str, module: nn.Module, config: Any) -> nn.Module:
        return module

    def before_denoise_step(self, step_index: int) -> None:
        return None

    def compile_component(self, component_name: str, module: nn.Module) -> nn.Module:
        if hasattr(module, "compile_repeated_blocks"):
            module.compile_repeated_blocks(**self.platform.compile_kwargs())
        else:
            module.compile(**self.platform.compile_kwargs())
        return module

    def validate_dynamic_weights(self, config: Any) -> None:
        return None


class DefaultRuntimeAdapter(ModelRuntimeAdapter):
    pass


_RUNTIME_ADAPTERS: dict[tuple[str, str], Type[ModelRuntimeAdapter]] = {}
_LAZY_RUNTIME_ADAPTERS = {
    ("cuda", "qwen_image"): (
        "diffsynth_engine.platforms.qwen",
        "CudaQwenImageRuntimeAdapter",
    ),
    ("rocm", "qwen_image"): (
        "diffsynth_engine.platforms.qwen",
        "CudaQwenImageRuntimeAdapter",
    ),
    ("ascend", "qwen_image"): (
        "diffsynth_engine.platforms.ascend_qwen",
        "AscendQwenImageRuntimeAdapter",
    ),
}


def register_runtime_adapter(
    platform_name: str,
    model_family: str,
    adapter_cls: Type[ModelRuntimeAdapter],
    *,
    overwrite: bool = False,
) -> None:
    key = (platform_name, model_family)
    if key in _RUNTIME_ADAPTERS and not overwrite:
        raise ValueError(f"Runtime adapter for {key!r} is already registered")
    _RUNTIME_ADAPTERS[key] = adapter_cls


def _load_lazy_adapter(key: tuple[str, str]) -> None:
    if key not in _LAZY_RUNTIME_ADAPTERS or key in _RUNTIME_ADAPTERS:
        return
    module_name, class_name = _LAZY_RUNTIME_ADAPTERS[key]
    module = importlib.import_module(module_name)
    _RUNTIME_ADAPTERS[key] = getattr(module, class_name)


def get_runtime_adapter(device: str, model_family: str) -> ModelRuntimeAdapter:
    platform = resolve_platform(device)
    key = (platform.name, model_family)
    _load_lazy_adapter(key)
    adapter_cls = _RUNTIME_ADAPTERS.get(key, DefaultRuntimeAdapter)
    return adapter_cls(platform, model_family)


__all__ = [
    "DefaultRuntimeAdapter",
    "ModelRuntimeAdapter",
    "get_runtime_adapter",
    "register_runtime_adapter",
]
