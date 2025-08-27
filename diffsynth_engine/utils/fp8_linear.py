import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from diffsynth_engine.utils.platform import DTYPE_FP8


def enable_fp8_autocast(module: nn.Module, compute_dtype: torch.dtype = torch.bfloat16, use_fp8_linear: bool = False):
    if len(list(module.children())) == 0:
        if len(list(module.parameters())) > 0:
            add_fp8_autocast_hook(module, compute_dtype)
        return
    if len(list(module.parameters(recurse=False))) > 0:
        add_fp8_autocast_hook(module, compute_dtype)
    for submodule in module.children():
        if isinstance(submodule, nn.Linear) and use_fp8_linear:
            continue

        enable_fp8_autocast(submodule, compute_dtype, use_fp8_linear)


def add_fp8_autocast_hook(module: nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
    def _fp8_autocast_pre_hook(module: nn.Module, input_):
        for name, param in module.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                param.data = param.data.to(compute_dtype)
        new_inputs = []
        for x in input_:
            if isinstance(x, torch.Tensor) and x.dtype in [torch.float8_e4m3fn, torch.float16, torch.bfloat16]:
                new_inputs.append(x.to(compute_dtype))
            else:
                new_inputs.append(x)
        return tuple(new_inputs)

    def _fp8_autocast_hook(module: nn.Module, input_, output_):
        for name, param in module.named_parameters():
            if param.dtype == compute_dtype:
                param.data = param.data.to(torch.float8_e4m3fn)

    if getattr(module, "_fp8_autocast_enabled", False):
        return
    module.register_forward_pre_hook(_fp8_autocast_pre_hook)
    module.register_forward_hook(_fp8_autocast_hook)
    setattr(module, "_fp8_autocast_enabled", True)


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
        origin_shape = input.shape
        input = input.reshape(-1, origin_shape[-1])

        x_max = torch.max(torch.abs(input), dim=-1, keepdim=True).values
        fp8_max = 448.0
        # For float8_e4m3fnuz, the maximum representable value is half of that of e4m3fn.
        # To avoid overflow and ensure numerical compatibility during FP8 computation,
        # we scale down the input by 2.0 in advance.
        # This scaling will be compensated later during the final result scaling.
        if DTYPE_FP8 == torch.float8_e4m3fnuz:
            fp8_max = fp8_max / 2.0
        scale_a = torch.clamp(x_max / fp8_max, min=1.0).float().to(device=device)
        scale_b = torch.ones((weight.shape[0], 1)).float().to(device=device)
        input = input / scale_a
        input = input.to(DTYPE_FP8)
        weight = weight.to(DTYPE_FP8)

        result = torch._scaled_mm(
            input,
            weight.T,
            scale_a=scale_a,
            scale_b=scale_b.T,
            bias=bias,
            out_dtype=origin_dtype,
        )
        new_shape = origin_shape[:-1] + result.shape[-1:]
        result = result.reshape(new_shape)
        return result

    F.linear = fp8_linear
    yield
    F.linear = origin_linear
