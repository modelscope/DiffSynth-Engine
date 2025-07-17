# cross-platform definitions and utilities
import torch
import gc


# data type
# AMD only supports float8_e4m3fnuz
# https://onnx.ai/onnx/technical/float8.html
if torch.version.hip and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName:
    DTYPE_FP8 = torch.float8_e4m3fnuz
else:
    DTYPE_FP8 = torch.float8_e4m3fn


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
