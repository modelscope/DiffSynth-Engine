import os
import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffsynth_engine.utils.import_utils import is_npu_available

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    from mindiesd.layers import layernorm_scale_shift
except ImportError:
    layernorm_scale_shift = None


class RMSNorm(nn.Module):
    """NPU-optimized RMSNorm wrapper with fallback to diffusers implementation."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # Cache the fallback instance so forward() reuses the same weight
        # tensor. register_parameter is reference assignment (no copy), so
        # self.weight and self._fallback.weight share the same storage.
        # When a checkpoint writes to "weight", both paths see the update.
        fallback = DiffusersRMSNorm(hidden_size, eps)
        self.register_parameter("weight", fallback.weight)
        # Use object.__setattr__ to avoid registering _fallback as an
        # nn.Module submodule, which would add spurious keys to state_dict()
        # and break strict checkpoint loading.
        object.__setattr__(self, "_fallback", fallback)

    def forward(self, hidden_states):
        if is_npu_available() and torch_npu is not None and not os.environ.get("DISABLE_NPU_RMSNORM"):
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
        else:
            return self._fallback(hidden_states)


class AdaLayerNorm(nn.Module):
    """NPU-optimized AdaLayerNorm with fallback to original implementation.

    Performs: output = layernorm(x) * (1 + scale) + shift

    Args:
        layernorm: The underlying nn.LayerNorm module (elementwise_affine=False)
    """

    def __init__(self, layernorm: nn.LayerNorm):
        super().__init__()
        self.layernorm = layernorm

    def forward(self, hidden_states: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor, shape [B, S, H]
            scale: Scale parameter, shape [B, H] or [B, 1, H]
            shift: Shift parameter, shape [B, H] or [B, 1, H]

        Returns:
            layernorm(x) * (1 + scale) + shift
        """
        if is_npu_available() and layernorm_scale_shift is not None and not os.environ.get("DISABLE_NPU_ADALAYERNORM"):
            # NPU path: use MindIE-SD fused operator
            return layernorm_scale_shift(
                layernorm=self.layernorm,
                x=hidden_states,
                scale=scale,
                shift=shift,
                fused=True
            )
        else:
            # Fallback: original Python implementation
            normed = self.layernorm(hidden_states)
            # Handle [B, 1, H] -> [B, H] dimension
            if scale.dim() == 2:
                scale = scale.unsqueeze(1)
            if shift.dim() == 2:
                shift = shift.unsqueeze(1)
            return normed * (1 + scale) + shift