import torch.nn as nn
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffsynth_engine.utils.import_utils import is_npu_available

try:
    import torch_npu
except ImportError:
    torch_npu = None


class RMSNorm(nn.Module):
    """NPU-optimized RMSNorm wrapper with fallback to diffusers implementation."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        diffusers_norm = DiffusersRMSNorm(hidden_size, eps)
        # Use same weight as diffusers RMSNorm to match checkpoint keys
        self.register_parameter("weight", diffusers_norm.weight)

    def forward(self, hidden_states):
        if is_npu_available() and torch_npu is not None:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
        else:
            return DiffusersRMSNorm(self.hidden_size, self.eps)(hidden_states)