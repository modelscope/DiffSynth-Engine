# cross-platform definitions and utilities
import torch
import gc
import platform


# data type
# AMD only supports float8_e4m3fnuz
# https://onnx.ai/onnx/technical/float8.html
if torch.version.hip and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName:
    DTYPE_FP8 = torch.float8_e4m3fnuz
else:
    DTYPE_FP8 = torch.float8_e4m3fn


# device backend
try:
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

IS_NPU = False
if _HAS_TORCH_NPU and hasattr(torch, "npu"):
    try:
        IS_NPU = torch.npu.is_available()
    except Exception:
        # a driver/CANN mismatch must degrade to "no NPU" instead of breaking every
        # module that imports this one
        IS_NPU = False


def empty_cache():
    gc.collect()
    if IS_NPU:
        torch.npu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()


def pin_memory(tensor: torch.Tensor):
    return tensor.pin_memory() if platform.system() == "Linux" else tensor
