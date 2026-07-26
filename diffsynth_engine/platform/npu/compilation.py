import os
import torch

from diffsynth_engine.utils.platform import is_mindie_sd_available


def apply_mindie_sd_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Apply MindIE-SD torch.compile backend fusion patterns.

    Enables RMSNorm, RoPE, AdaLayerNorm, and MulAdd op fusion for NPU execution.
    FastGELU fusion is disabled (npu_fast_gelu was slower than native GELU in practice).
    On non-NPU devices or when mindiesd is unavailable, returns the model unchanged.

    Compilation strategy:
        If the model defines ``_repeated_blocks`` (e.g. ``["QwenImageTransformerBlock"]``),
        compiles only the repeated submodules individually to reduce compilation time
        and graph break risk. Otherwise falls back to full-model ``torch.compile``.
        Set ``DIFFSYNTH_DISABLE_COMPILE=1`` to disable.
    """
    if not is_mindie_sd_available() or os.environ.get("DIFFSYNTH_DISABLE_COMPILE", "0") == "1":
        return model

    from mindiesd.compilation import CompilationConfig, MindieSDBackend

    CompilationConfig.fusion_patterns.enable_fast_gelu = False

    repeated_blocks = getattr(model, "_repeated_blocks", None)
    if repeated_blocks:
        paths = [
            name
            for name, submodule in model.named_modules()
            if submodule.__class__.__name__ in repeated_blocks
        ]
        # Sort by depth in descending order (submodules before parent modules)
        paths.sort(key=lambda p: len(p.split(".")), reverse=True)

        backend = MindieSDBackend()
        for name in paths:
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            submodule = getattr(parent, child_name)
            setattr(parent, child_name, torch.compile(submodule, backend=backend))

        return model

    # Fall back to full-model compilation for models without `_repeated_blocks` defined.
    return torch.compile(model, backend=MindieSDBackend())
