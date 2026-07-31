import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth_engine.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    is_tp_group_initialized,
)


def get_tp_size() -> int:
    return get_tensor_model_parallel_world_size() if is_tp_group_initialized() else 1


def get_tp_rank() -> int:
    return get_tensor_model_parallel_rank() if is_tp_group_initialized() else 0


@torch.compiler.disable
def tp_all_reduce(output: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(output)


@torch.compiler.disable
def tp_all_gather(output: torch.Tensor, dim: int) -> torch.Tensor:
    return get_tp_group().all_gather(output, dim=dim)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        tp_size = get_tp_size()
        if out_features % tp_size != 0:
            raise ValueError(
                f"ColumnParallelLinear: out_features ({out_features}) must be divisible by tp_size ({tp_size})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.out_features_per_partition = out_features // tp_size
        self.tp_size = tp_size
        self.tp_rank = get_tp_rank()

        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            output = tp_all_gather(output, dim=-1)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"out_per_partition={self.out_features_per_partition}, "
            f"bias={self.bias is not None}, gather_output={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        tp_size = get_tp_size()
        if in_features % tp_size != 0:
            raise ValueError(f"RowParallelLinear: in_features ({in_features}) must be divisible by tp_size ({tp_size})")

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.in_features_per_partition = in_features // tp_size
        self.tp_size = tp_size
        self.tp_rank = get_tp_rank()

        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.linear(hidden_states, self.weight, self.bias)

        if not self.input_is_parallel:
            hidden_states = hidden_states.chunk(self.tp_size, dim=-1)[self.tp_rank].contiguous()

        output = F.linear(hidden_states, self.weight, None)
        output = tp_all_reduce(output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_per_partition={self.in_features_per_partition}, "
            f"bias={self.bias is not None}, input_is_parallel={self.input_is_parallel}"
        )
