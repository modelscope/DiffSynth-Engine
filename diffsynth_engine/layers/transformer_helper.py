# SPDX-License-Identifier: Apache-2.0
"""Reusable transformer helper layers (fused RMSNorm for NPU, manual fallback elsewhere)."""

import torch
import torch.nn as nn

from diffsynth_engine.utils.platform import current_platform, is_mindie_sd_available


class RMSNorm(nn.Module):
    """RMSNorm over the last dim.

    API-compatible with `diffusers.models.normalization.RMSNorm`
    (dim, eps, elementwise_affine), so existing checkpoints (weight key) load
    unchanged. On Ascend with `current_platform.op_fusion` enabled the norm is
    fused into a single `torch_npu.npu_rms_norm` op; otherwise it falls back to the
    reference fp32 math so numerics match diffusers exactly.
    """

    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if (
            current_platform.op_fusion
            and is_mindie_sd_available()
            and self.elementwise_affine
            and x.device.type == "npu"
        ):
            import torch_npu

            return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]

        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"