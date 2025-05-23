import torch
import torch.nn as nn
from contextlib import contextmanager

# mofified from transformers.modeling_utils
TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

_init_weights = True


@contextmanager
def no_init_weights():
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    global _init_weights
    old_init_weights = _init_weights

    def _skip_init(*args, **kwargs):
        pass

    _init_weights = False
    # Save the original initialization functions
    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        _init_weights = old_init_weights
        # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)


def zero_module(module: nn.Module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
