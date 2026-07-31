import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth_engine.layers.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear, get_tp_size


class ColumnParallelGELU(nn.Module):
    """Column-parallel linear projection followed by GELU."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, gather_output=False)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class TPFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4,
        inner_dim: int | None = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()
        if activation_fn not in ("gelu", "gelu-approximate"):
            raise ValueError(f"Unsupported activation_fn={activation_fn!r}; supported: ['gelu', 'gelu-approximate']")
        approximate = "tanh" if activation_fn == "gelu-approximate" else "none"

        inner_dim = inner_dim if inner_dim is not None else int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        tp_size = get_tp_size()
        if inner_dim % tp_size != 0:
            raise ValueError(f"inner_dim ({inner_dim}) must be divisible by tp_size ({tp_size})")

        self.net = nn.ModuleList(
            [
                ColumnParallelGELU(
                    dim,
                    inner_dim,
                    approximate=approximate,
                    bias=True,
                ),
                nn.Dropout(dropout),
                RowParallelLinear(inner_dim, dim_out, bias=True, input_is_parallel=True),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
