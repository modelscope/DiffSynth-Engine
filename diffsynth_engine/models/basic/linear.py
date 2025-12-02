import torch
import torch.nn as nn
from typing import Tuple

from diffsynth_engine.utils.platform import DTYPE_FP8, FP8_MAX


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_max = x.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = x_max / FP8_MAX
    x_scaled = x / scale
    return x_scaled, scale


def fp8_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, scaling: bool = True
) -> torch.Tensor:
    device = input.device
    origin_dtype = input.dtype
    origin_shape = input.shape
    input = input.reshape(-1, origin_shape[-1])
    out_features, _ = weight.shape

    if scaling:
        input, scale_a = per_token_cast_to_fp8(input)
        scale_b = torch.ones((out_features, 1), device=device)
    else:
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
    input = input.to(DTYPE_FP8)
    weight = weight.to(DTYPE_FP8)

    result = torch._scaled_mm(
        input,
        weight.T,
        scale_a=scale_a,
        scale_b=scale_b.T,
        bias=bias,
        out_dtype=origin_dtype,
    )
    new_shape = origin_shape[:-1] + result.shape[-1:]
    result = result.reshape(new_shape)
    return result


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        scaling: bool = True,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight.data = self.weight.data.to(DTYPE_FP8)
        self.scaling = scaling

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return fp8_linear(input, self.weight, self.bias, self.scaling)
