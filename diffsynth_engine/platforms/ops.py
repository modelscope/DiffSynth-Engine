# SPDX-License-Identifier: Apache-2.0
"""统一平台融合算子接口。

GPU 路径直通原逻辑，NPU 路径调用 mindiesd / torch_npu 融合算子。
调用方无需判断平台类型，直接调用本模块函数即可自动分发。
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from diffsynth_engine.platforms import current_platform
from diffsynth_engine.utils.platform import is_mindie_sd_available


def _is_npu_fused(x: torch.Tensor) -> bool:
    """判断是否走 NPU 融合路径。"""
    return current_platform.op_fusion and is_mindie_sd_available() and x.device.type == "npu"


# ---------------------------------------------------------------------------
# fused_rotary_embedding
# ---------------------------------------------------------------------------


def fused_rotary_embedding(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """平台无关 RoPE 旋转位置编码。

    - NPU (op_fusion + mindiesd): mindiesd.rotary_position_embedding 融合算子
    - GPU / 其他: 原始数学实现 (view_as_complex/polar 或 real-valued cos/sin)

    Args:
        x: 输入张量，形状 [B, S, H, D]。
        freqs_cis: 预计算频率张量。use_real=True 时为 (cos, sin) 元组；
                   use_real=False 时为复数张量 [S, D/2]。
        use_real: 是否使用实数形式的 cos/sin（适用于 flux/cogvideox 等）。
        use_real_unbind_dim: use_real=True 时，拆分维度 (-1 或 -2)。

    Returns:
        应用 RoPE 后的张量，形状与输入一致。
    """
    if use_real:
        # Real-valued cos/sin 模式 (flux, cogvideox, hunyuan-dit, stable audio, etc.)
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    else:
        # Complex-valued 模式 (qwen_image 等)
        if _is_npu_fused(x):
            from mindiesd import rotary_position_embedding

            # Cache expanded cos/sin on the tensor object itself.
            cached = getattr(freqs_cis, "_rope_expanded", None)
            if cached is None:
                cos = freqs_cis.real  # (s, d/2)
                sin = freqs_cis.imag
                cos = cos.reshape(1, -1, 1, cos.shape[-1])  # (1, S, 1, D/2)
                sin = sin.reshape(1, -1, 1, sin.shape[-1])
                cos = cos.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(start_dim=-2)  # (1, S, 1, D)
                sin = sin.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(start_dim=-2)
                cos, sin = cos.to(x.device), sin.to(x.device)
                cached = (cos, sin)
                freqs_cis._rope_expanded = cached
            cos, sin = cached
            return rotary_position_embedding(
                x,
                cos,
                sin,
                rotated_mode="rotated_interleaved",
                head_first=False,
                fused=True,
            )

        # GPU fallback: complex 乘法
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


# ---------------------------------------------------------------------------
# fused_layernorm_scale_shift
# ---------------------------------------------------------------------------


def fused_layernorm_scale_shift(
    norm_layer: nn.LayerNorm,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """融合 LayerNorm + Scale + Shift。

    - NPU (op_fusion + mindiesd): mindiesd.layernorm_scale_shift 融合算子
    - GPU / 其他: norm_layer(x) * (1 + scale) + shift

    Args:
        norm_layer: nn.LayerNorm 层实例。
        x: 输入张量。
        scale: 缩放因子 (对应 Ada modulate 中的 scale)。
        shift: 偏移量 (对应 Ada modulate 中的 shift)。

    Returns:
        融合归一化+调制后的张量。
    """
    if _is_npu_fused(x):
        from mindiesd import layernorm_scale_shift

        return layernorm_scale_shift(norm_layer, x, scale, shift, fused=True)

    # GPU fallback: 手动计算
    return norm_layer(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# fused_rms_norm
# ---------------------------------------------------------------------------


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """融合 RMSNorm。

    - NPU (op_fusion + mindiesd + elementwise_affine): torch_npu.npu_rms_norm
    - GPU / 其他: fp32 手动计算

    Args:
        x: 输入张量。
        weight: RMSNorm 权重参数。
        eps: 数值稳定性 epsilon。

    Returns:
        归一化后的张量。
    """
    if _is_npu_fused(x):
        import torch_npu

        return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]

    # GPU fallback: fp32 精度手动计算
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (output * weight).type_as(x)
