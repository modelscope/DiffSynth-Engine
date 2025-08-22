import torch
import torch.nn as nn
from typing import Dict

from diffsynth_engine.utils.platform import pin_memory


def enable_sequential_cpu_offload(module: nn.Module, device: str = "cuda"):
    module = module.to("cpu")
    if len(list(module.children())) == 0:
        if len(list(module.parameters())) > 0 or len(list(module.buffers())) > 0:
            # leaf module with parameters or buffers
            add_cpu_offload_hook(module, device)
        return
    if len(list(module.parameters(recurse=False))) > 0 or len(list(module.buffers(recurse=False))):
        # module with direct parameters or buffers
        add_cpu_offload_hook(module, device, recurse=False)
    for submodule in module.children():
        enable_sequential_cpu_offload(submodule, device)


def add_cpu_offload_hook(module: nn.Module, device: str = "cuda", recurse: bool = True):
    def _forward_pre_hook(module: nn.Module, input_):
        offload_param_dict = getattr(module, "_offload_param_dict", {})
        if len(offload_param_dict) > 0:
            for name, param in module.named_parameters(recurse=recurse):
                param.data = param.data.to(device=device)
            for name, buffer in module.named_buffers(recurse=recurse):
                buffer.data = buffer.data.to(device=device)
            return tuple(x.to(device=device) if isinstance(x, torch.Tensor) else x for x in input_)
        for name, param in module.named_parameters(recurse=recurse):
            param.data = pin_memory(param.data)
            offload_param_dict[name] = param.data
            param.data = param.data.to(device=device)
        for name, buffer in module.named_buffers(recurse=recurse):
            buffer.data = pin_memory(buffer.data)
            offload_param_dict[name] = buffer.data
            buffer.data = buffer.data.to(device=device)
        setattr(module, "_offload_param_dict", offload_param_dict)
        return tuple(x.to(device=device) if isinstance(x, torch.Tensor) else x for x in input_)

    def _forward_hook(module: nn.Module, input_, output_):
        offload_param_dict = getattr(module, "_offload_param_dict", {})
        for name, param in module.named_parameters(recurse=recurse):
            if name in offload_param_dict:
                param.data = offload_param_dict[name]
        for name, buffer in module.named_buffers(recurse=recurse):
            if name in offload_param_dict:
                buffer.data = offload_param_dict[name]

    if getattr(module, "_cpu_offload_enabled", False):
        return
    module.register_forward_pre_hook(_forward_pre_hook)
    module.register_forward_hook(_forward_hook)
    setattr(module, "_cpu_offload_enabled", True)


def offload_model_to_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    module = module.to("cpu")
    offload_param_dict = {}
    for name, param in module.named_parameters(recurse=True):
        param.data = pin_memory(param.data)
        offload_param_dict[name] = param.data
    for name, buffer in module.named_buffers(recurse=True):
        buffer.data = pin_memory(buffer.data)
        offload_param_dict[name] = buffer.data
    return offload_param_dict


def restore_model_from_dict(module: nn.Module, offload_param_dict: Dict[str, torch.Tensor]):
    for name, param in module.named_parameters(recurse=True):
        if name in offload_param_dict:
            param.data = offload_param_dict[name]
    for name, buffer in module.named_buffers(recurse=True):
        if name in offload_param_dict:
            buffer.data = offload_param_dict[name]
