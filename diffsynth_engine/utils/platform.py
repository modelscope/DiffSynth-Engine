# cross-platform definitions and utilities
import gc
import torch

from diffsynth_engine.platforms import (
    AscendPlatform,
    get_device_type as _get_device_type,
    get_preferred_fp8_dtype,
    pin_memory as _pin_memory,
    resolve_platform,
)


DTYPE_FP8 = get_preferred_fp8_dtype("cuda")


def empty_cache(device: str | torch.device | None = None):
    gc.collect()
    resolve_platform(get_device_type(device)).empty_cache()


def pin_memory(
    tensor: torch.Tensor,
    device: str | torch.device | None = None,
):
    return _pin_memory(tensor, device)


def is_npu_available() -> bool:
    return AscendPlatform.supports("device")


def is_mindie_sd_available() -> bool:
    return AscendPlatform.supports("mindie")


def get_device_type(device: str | torch.device | None = None) -> str:
    return _get_device_type(device)


def get_torch_distributed_backend(device: str | torch.device) -> str:
    return resolve_platform(device).distributed_backend()
