import torch


from diffsynth_engine.platforms import (
    AscendPlatform,
    current_platform,
)


def is_npu_available() -> bool:
    return AscendPlatform.is_available()


def is_mindie_sd_available() -> bool:
    return AscendPlatform.supports("mindie")


def get_device_type() -> str:
    return current_platform.device_type


def get_device(local_rank: int) -> torch.device:
    return current_platform.get_device(local_rank)


def get_torch_distributed_backend() -> str:
    return current_platform.distributed_backend()


def device_count() -> int:
    return current_platform.device_count()


def set_device(index: int | str | torch.device) -> None:
    current_platform.set_device(index)


def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    return current_platform.pin_memory(tensor)


def get_compile_kwargs() -> dict:
    return current_platform.compile_kwargs()


DTYPE_FP8 = current_platform.fp8_dtype()

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
