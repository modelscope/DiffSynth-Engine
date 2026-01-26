"""
Context manager to skip PyTorch weight initialization.

This is useful when loading models from state_dict, where initialization
would be wasteful as weights will be overwritten anyway.
"""
import torch
import torch.nn as nn
from typing import Literal
from contextlib import contextmanager


@contextmanager
def skip_init(mode: Literal["smart", "full", "minimal"] = "smart"):
    """
    Context manager that skips weight initialization for PyTorch modules.

    This works by temporarily replacing initialization functions with no-ops.
    Useful when you plan to load weights from a state_dict immediately after
    module creation, avoiding wasteful initialization computation.

    Args:
        mode: Skipping mode, one of:
            - "smart": Skip init functions only, allow reset_parameters (safer, default)
            - "full": Skip both init functions and reset_parameters (most aggressive)
            - "minimal": Only skip expensive init functions like kaiming/xavier

    Example:
        >>> # Smart mode (default) - works for most cases including RotaryEmbedding
        >>> with skip_init(mode="smart"):
        ...     model = MyModel(device="meta", dtype=torch.float32)
        >>> model.load_state_dict(state_dict, assign=True)

        >>> # Full mode - maximum speed but may break some modules
        >>> with skip_init(mode="full"):
        ...     model = MyModel(device="meta", dtype=torch.float32)
    """
    original_inits = {}

    if mode == "minimal":
        # Only skip expensive initializations
        init_functions = [
            (nn.init, 'xavier_uniform_'),
            (nn.init, 'xavier_normal_'),
            (nn.init, 'kaiming_uniform_'),
            (nn.init, 'kaiming_normal_'),
            (nn.init, 'orthogonal_'),
        ]
    else:
        # Skip all init functions
        init_functions = [
            (nn.init, 'uniform_'),
            (nn.init, 'normal_'),
            (nn.init, 'constant_'),
            (nn.init, 'ones_'),
            (nn.init, 'zeros_'),
            (nn.init, 'eye_'),
            (nn.init, 'dirac_'),
            (nn.init, 'xavier_uniform_'),
            (nn.init, 'xavier_normal_'),
            (nn.init, 'kaiming_uniform_'),
            (nn.init, 'kaiming_normal_'),
            (nn.init, 'trunc_normal_'),
            (nn.init, 'orthogonal_'),
            (nn.init, 'sparse_'),
        ]

    def noop_init(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return tensor

    try:
        # Patch init functions
        for module, func_name in init_functions:
            if hasattr(module, func_name):
                original_inits[(module, func_name)] = getattr(module, func_name)
                setattr(module, func_name, noop_init)

        # Optionally patch reset_parameters
        original_reset_params = {}
        if mode == "full":
            module_classes = [
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.Embedding,
                nn.EmbeddingBag,
            ]

            for module_cls in module_classes:
                if hasattr(module_cls, 'reset_parameters'):
                    original_reset_params[module_cls] = module_cls.reset_parameters
                    module_cls.reset_parameters = lambda self: None

        yield

    finally:
        # Restore init functions
        for (module, func_name), original_func in original_inits.items():
            setattr(module, func_name, original_func)

        # Restore reset_parameters
        for module_cls, original_method in original_reset_params.items():
            module_cls.reset_parameters = original_method


@contextmanager
def skip_init_on_meta(mode: Literal["smart", "full", "minimal"] = "smart"):
    """
    Enhanced version that combines skip_init with meta device initialization.

    This ensures modules are created on meta device without any weight
    initialization, providing maximum efficiency when loading from state_dict.

    Args:
        mode: Same as skip_init mode parameter

    Example:
        >>> with skip_init_on_meta():
        ...     model = MyModel(dtype=torch.float32)  # device defaults to "meta"
        >>> model.load_state_dict(state_dict, assign=True)
        >>> model.to(device="cuda", dtype=torch.float32)
    """
    original_device = None

    try:
        original_device = torch.get_default_device()
        torch.set_default_device('meta')

        with skip_init(mode=mode):
            yield

    finally:
        if original_device is not None:
            torch.set_default_device(original_device)
        else:
            torch.set_default_device(None)
