import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


def _is_cuda() -> bool:
    # Historical: torch build has CUDA, not necessarily a visible GPU.
    return torch.version.cuda is not None


def _is_rocm() -> bool:
    return torch.version.hip is not None


def _is_mps() -> bool:
    return torch.backends.mps.is_available()


def _active_platform():
    """Resolve PlatformBackend for the process-preferred accelerator."""
    from diffsynth_engine.platforms import resolve_platform

    return resolve_platform(get_device_type())


def is_npu_available() -> bool:
    from diffsynth_engine.platforms import AscendPlatform

    return AscendPlatform.is_available()


def is_mindie_sd_available() -> bool:
    from diffsynth_engine.platforms import AscendPlatform

    return AscendPlatform.supports("mindie")


def get_device(local_rank: int) -> torch.device:
    if _is_cuda() or _is_rocm():
        return torch.device("cuda", local_rank)
    if is_npu_available():
        return torch.device("npu", local_rank)
    if _is_mps():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_type() -> str:
    """Preferred accelerator for this process (no-arg, v1 public API).

    Priority matches historical utils behavior: cuda/rocm build > npu > mps > cpu.
    Differs from ``platforms.auto_detect_device`` which uses ``is_available()``.
    """
    if _is_cuda() or _is_rocm():
        return "cuda"
    if is_npu_available():
        return "npu"
    if _is_mps():
        return "mps"
    return "cpu"


def get_torch_distributed_backend() -> str:
    device_type = get_device_type()
    if device_type == "cpu":
        raise NotImplementedError("Unsupported device type")
    return _active_platform().distributed_backend()


def device_count() -> int:
    return _active_platform().device_count()


def set_device(index: int | str | torch.device) -> None:
    """Bind the current process to a local device (cuda or npu)."""
    _active_platform().set_device(index)


def align_config_device(config_device: str | torch.device, target_type: str | None = None) -> str:
    """Rewrite historical CUDA placeholder to NPU when Ascend is the active accelerator.

    Leaves other placeholders alone (e.g. default ``cuda`` on a CPU laptop).
    Explicit ``npu`` on a non-NPU machine raises.
    """
    if target_type is None:
        target_type = get_device_type()
    device_str = str(config_device)
    current_type = device_str.split(":", 1)[0].lower()
    if current_type == target_type:
        return device_str
    if target_type == "npu" and current_type == "cuda":
        return "npu"
    if current_type == "npu" and target_type != "npu":
        raise RuntimeError(
            f"config.device={config_device!r} does not match available device_type={target_type!r}"
        )
    return device_str


def bind_rank_device(config_device: str | torch.device, local_rank: int) -> str:
    """Worker-only: bind config.device to this rank's local device (e.g. npu:0)."""
    device_type = str(config_device).split(":", 1)[0].lower()
    if device_type in ("cpu", "mps"):
        return str(config_device)
    return f"{device_type}:{local_rank}"


def get_compile_kwargs() -> dict:
    """Return kwargs for ``nn.Module.compile`` / ``torch.compile``.

    On Ascend with MindIE compile available, injects MindieSDBackend.
    Otherwise returns ``{}`` so the default inductor path is used.
    """
    if not is_npu_available():
        return {}

    from diffsynth_engine.platforms import AscendPlatform

    if not AscendPlatform.supports("mindie_compile"):
        logger.warning(
            "MindIE-SD compile backend is unavailable; falling back to default torch.compile backend"
        )
        return {}
    return AscendPlatform.compile_kwargs()


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
