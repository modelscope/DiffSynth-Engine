import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from diffsynth_engine.utils.constants import DTYPE_FP8


def enable_fp8_linear(module: nn.Module):
    _enable_fp8_linear(module)
    setattr(module, "fp8_linear_enabled", True)


def _enable_fp8_linear(module: nn.Module):
    if isinstance(module, nn.Linear) and torch.is_floating_point(module.weight.data):
        # avoid conversion for int weights like GGUF
        module.weight.data = module.weight.data.to(DTYPE_FP8)
    for submodule in module.children():
        _enable_fp8_linear(submodule)


@contextmanager
def fp8_inference(enabled=True):
    if not enabled:
        yield
        return

    origin_linear = F.linear

    def fp8_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = input.device
        origin_dtype = input.dtype
        scale_a = 1.0
        # For float8_e4m3fnuz, the maximum representable value is half of that of e4m3fn.
        # To avoid overflow and ensure numerical compatibility during FP8 computation,
        # we scale down the input by 2.0 in advance.
        # This scaling will be compensated later during the final result scaling.
        if DTYPE_FP8 == torch.float8_e4m3fnuz:
            scale_a = 2.0
            input = input / scale_a
        input = input.to(DTYPE_FP8)
        weight = weight.to(DTYPE_FP8)

        if len(input.shape) > 2:
            origin_shape = input.shape
            input = input.reshape(-1, origin_shape[-1])
            result = torch._scaled_mm(
                input,
                weight.T,
                scale_a=torch.tensor(scale_a).to(device=device),
                scale_b=torch.tensor(1.0).to(device=device),
                bias=bias,
                out_dtype=origin_dtype,
            )
            new_shape = origin_shape[:-1] + result.shape[-1:]
            result = result.reshape(new_shape)
        else:
            result = torch._scaled_mm(
                input,
                weight.T,
                scale_a=torch.tensor(scale_a).to(device=device),
                scale_b=torch.tensor(1.0).to(device=device),
                bias=bias,
                out_dtype=origin_dtype,
            )
        return result

    F.linear = fp8_linear
    yield
    F.linear = origin_linear
