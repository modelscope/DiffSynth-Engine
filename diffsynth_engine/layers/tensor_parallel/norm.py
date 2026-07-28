import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth_engine.layers.tensor_parallel.linear import get_tp_rank, get_tp_size, tp_all_reduce


class TensorParallelRMSNorm(nn.Module):
    """RMSNorm over a hidden dimension sharded across the tensor-parallel group."""

    def __init__(
        self,
        hidden_size: int,
        eps: float | None = None,
        elementwise_affine: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        tp_size = get_tp_size()
        if hidden_size % tp_size != 0:
            raise ValueError(
                f"TensorParallelRMSNorm: hidden_size ({hidden_size}) must be divisible by tp_size ({tp_size})"
            )

        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size // tp_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.tp_size = tp_size
        self.tp_rank = get_tp_rank()

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.hidden_size_per_partition, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.rms_norm(hidden_states, (self.hidden_size,), self.weight, self.eps)
        if hidden_states.shape[-1] != self.hidden_size_per_partition:
            raise ValueError(f"Expected last dimension {self.hidden_size_per_partition}, got {hidden_states.shape[-1]}")

        variance = hidden_states.float().pow(2).sum(dim=-1, keepdim=True)
        variance = tp_all_reduce(variance) / self.hidden_size
        eps = self.eps if self.eps is not None else torch.finfo(hidden_states.dtype).eps
        output = hidden_states * torch.rsqrt(variance + eps).to(hidden_states.dtype)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"hidden_size_per_partition={self.hidden_size_per_partition}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
