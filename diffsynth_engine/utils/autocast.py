import torch
import torch.nn as nn


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
