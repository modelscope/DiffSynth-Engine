import torch

from diffsynth_engine.utils.platform import is_mindie_sd_available


def apply_mindie_sd_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Apply MindIE-SD torch.compile backend fusion patterns.

    Enables RMSNorm, RoPE, AdaLayerNorm, FastGELU, and MulAdd op fusion
    for NPU execution. On non-NPU devices or when mindiesd is unavailable,
    returns the model unchanged.
    """
    if not is_mindie_sd_available():
        return model

    from mindiesd.compilation import MindieSDBackend

    return torch.compile(model, backend=MindieSDBackend())
