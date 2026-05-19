import torch.nn as nn
import torch.nn.functional as F
from diffsynth_engine.utils.import_utils import is_npu_available

try:
    import torch_npu
except ImportError:
    torch_npu = None


class _GELUProj(nn.Module):
    """Wrapper to match diffusers FeedForward GELU structure with internal proj.

    This wrapper holds the first Linear layer as .proj to match checkpoint keys.
    """

    def __init__(self, dim, inner_dim):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, x):
        return F.gelu(x, approximate="tanh")


class FastGELUMLP(nn.Module):
    """MLP with npu_fast_gelu on NPU, fallback to F.gelu on other devices.

    Functionally equivalent to diffusers.models.attention.FeedForward(
        dim=dim, dim_out=dim, activation_fn="gelu-approximate"
    )
    """

    def __init__(self, dim, dim_out=None, mult=4):
        """Initialize MLP.

        Args:
            dim: Input and output dimension
            dim_out: Output dimension, defaults to dim
            mult: inner_dim = dim * mult, defaults to 4
        """
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Match diffusers FeedForward structure: net[0]=GELU(proj), net[2]=output
        # net[1] is Dropout which is skipped in inference
        self.net = nn.ModuleList([
            _GELUProj(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out),
        ])

    def forward(self, hidden_states):
        """Forward pass.

        Args:
            hidden_states: Input tensor, shape [B, S, dim]

        Returns:
            Output tensor, shape [B, S, dim_out]
        """
        # net[0] = _GELUProj with internal proj (dim → inner_dim)
        hidden_states = self.net[0].proj(hidden_states)

        if is_npu_available() and torch_npu is not None:
            hidden_states = torch_npu.npu_fast_gelu(hidden_states)
        else:
            hidden_states = F.gelu(hidden_states, approximate="tanh")

        # net[2] = output Linear (inner_dim → dim_out)
        hidden_states = self.net[2](hidden_states)
        return hidden_states