import torch
import torch.nn as nn


def enable_sequential_cpu_offload(module: nn.Module, device: str = "cuda"):
    if len(list(module.children())) == 0:
        if len(list(module.parameters())) > 0:  # leaf module with parameters
            add_cpu_offload_hook(module, device)
        return
    if len(list(module.parameters(recurse=False))) > 0:  # module with direct parameters
        add_cpu_offload_hook(module, device, recurse=False)
    for submodule in module.children():
        enable_sequential_cpu_offload(submodule, device)


# TODO: supports module buffer
def add_cpu_offload_hook(module: nn.Module, device: str = "cuda", recurse: bool = True):
    def _forward_pre_hook(module: nn.Module, input):
        offload_params = {}
        for name, param in module.named_parameters(recurse=recurse):
            offload_params[name] = param.data
            param.data = param.data.to(device=device)
        setattr(module, "_offload_params", offload_params)
        return tuple(x.to(device=device) if isinstance(x, torch.Tensor) else x for x in input)

    def _forward_hook(module: nn.Module, input, output):
        offload_params = getattr(module, "_offload_params", {})
        for name, param in module.named_parameters(recurse=recurse):
            if name in offload_params:
                param.data = offload_params[name]

    if getattr(module, "_cpu_offload_enabled", False):
        return
    module.register_forward_pre_hook(_forward_pre_hook)
    module.register_forward_hook(_forward_hook)
    setattr(module, "_cpu_offload_enabled", True)
