import torch
from functools import cache


def _is_cuda() -> bool:
    return torch.version.cuda is not None


def _is_rocm() -> bool:
    return torch.version.hip is not None


def _is_mps() -> bool:
    return torch.backends.mps.is_available()

@cache
def is_npu_available() -> bool:
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except ImportError:
        return False

@cache
def is_mindie_sd_available() -> bool:
    try:
        import mindiesd  # noqa: F401

        return is_npu_available()
    except ImportError:
        return False


def get_device(local_rank: int) -> torch.device:
    if is_npu_available():
        return torch.device("npu", local_rank)
    if _is_cuda() or _is_rocm():
        return torch.device("cuda", local_rank)
    if _is_mps():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_type() -> str:
    if is_npu_available():
        return "npu"
    if _is_cuda() or _is_rocm():
        return "cuda"
    if _is_mps():
        return "mps"
    else:
        return "cpu"


def get_torch_distributed_backend() -> str:
    if is_npu_available():
        return "hccl"
    if _is_cuda() or _is_rocm():
        return "nccl"
    if _is_mps():
        return "gloo"
    else:
        raise NotImplementedError("Unsupported device type")


DTYPE_FP8 = torch.float8_e4m3fnuz if _is_rocm() else torch.float8_e4m3fn

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
