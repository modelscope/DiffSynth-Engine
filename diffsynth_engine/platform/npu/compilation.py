import torch

from diffsynth_engine.utils.platform import is_mindie_sd_available


def apply_mindie_sd_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Apply MindIE-SD torch.compile backend fusion patterns.

    Enables RMSNorm, RoPE, AdaLayerNorm, and MulAdd op fusion for NPU execution.
    FastGELU fusion is disabled (npu_fast_gelu was slower than native GELU in practice).
    On non-NPU devices or when mindiesd is unavailable, returns the model unchanged.
    """
    if not is_mindie_sd_available():
        return model

    from mindiesd.compilation import CompilationConfig, MindieSDBackend

    CompilationConfig.fusion_patterns.enable_fast_gelu = False
    return torch.compile(model, backend=MindieSDBackend())
