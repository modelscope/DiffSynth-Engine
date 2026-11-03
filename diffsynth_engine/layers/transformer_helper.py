# SPDX-License-Identifier: Apache-2.0
"""Reusable transformer helper layers (fused RMSNorm for NPU, manual fallback elsewhere)."""

import torch
import torch.nn as nn

from diffsynth_engine.platforms.ops import fused_rms_norm


class RMSNorm(nn.Module):
    """RMSNorm over the last dim.

    API-compatible with `diffusers.models.normalization.RMSNorm`
    (dim, eps, elementwise_affine), so existing checkpoints (weight key) load
    unchanged. When `elementwise_affine` is enabled, delegates to
    `fused_rms_norm` which automatically dispatches to the platform-optimal
    implementation; otherwise falls back to reference fp32 math.
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
        if self.elementwise_affine:
            return fused_rms_norm(x, self.weight, self.eps)

        output = self._norm(x.float()).type_as(x)
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"