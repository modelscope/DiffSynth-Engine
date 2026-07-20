import gc
from typing import Optional

import torch

from diffsynth_engine.platforms import (
    AscendPlatform,
    get_device_type as _get_device_type,
    get_preferred_fp8_dtype,
    pin_memory as _pin_memory,
    resolve_platform,
)


DTYPE_FP8 = get_preferred_fp8_dtype("cuda")


def empty_cache(device: Optional[str | torch.device] = None):
    gc.collect()
    if device is not None:
        platform_cls = resolve_platform(device)
        if platform_cls.is_available():
            platform_cls.empty_cache()
        return

    # No-argument calls preserve the historical CUDA/MPS behavior. NPU callers
    # pass their explicit device so a coexisting NPU never affects CUDA cleanup.
    if torch.cuda.is_available():
        resolve_platform("cuda").empty_cache()
    if torch.backends.mps.is_available():
        resolve_platform("mps").empty_cache()


def pin_memory(tensor: torch.Tensor, device: Optional[str | torch.device] = None):
    return _pin_memory(tensor, device)


def is_npu_available() -> bool:
    return AscendPlatform.supports("device")


def is_mindie_sd_available() -> bool:
    return AscendPlatform.supports("mindie")


def get_device_type(device: str | torch.device) -> str:
    return _get_device_type(device)


def get_torch_distributed_backend(device: str | torch.device) -> str:
    return resolve_platform(device).distributed_backend()
