# SPDX-License-Identifier: Apache-2.0
"""Unit tests for diffsynth_engine.platforms.ops on CPU (GPU-fallback paths)."""

import math

import torch
import torch.nn as nn
import pytest

from diffsynth_engine.platforms.ops import (
    fused_layernorm_scale_shift,
    fused_rms_norm,
    fused_rotary_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, atol=1e-6, rtol=1e-6):
    """断言两个 tensor 在给定容差内相等。"""
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# fused_rms_norm tests
# ---------------------------------------------------------------------------


class TestFusedRmsNorm:
    """验证 fused_rms_norm GPU fallback 与手动计算一致。"""

    @pytest.mark.parametrize("shape", [(2, 8, 64), (1, 128), (4, 16, 32)])
    def test_output_matches_manual(self, shape):
        torch.manual_seed(42)
        x = torch.randn(*shape)
        weight = torch.randn(shape[-1])
        eps = 1e-6

        # 手动计算参考结果
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
        expected = (x_fp32 * rms * weight).to(x.dtype)

        result = fused_rms_norm(x, weight, eps)
        assert_tensor_equal(result, expected)

    def test_preserves_dtype_bfloat16(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, dtype=torch.bfloat16)

        result = fused_rms_norm(x, weight, eps=1e-6)
        assert result.dtype == torch.bfloat16

    def test_preserves_dtype_float16(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, dtype=torch.float16)
        weight = torch.randn(16, dtype=torch.float16)

        result = fused_rms_norm(x, weight, eps=1e-6)
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# fused_layernorm_scale_shift tests
# ---------------------------------------------------------------------------


class TestFusedLayernormScaleShift:
    """验证 fused_layernorm_scale_shift GPU fallback 与手动计算一致。"""

    @pytest.mark.parametrize("shape", [(2, 8, 64), (1, 4, 128)])
    def test_output_matches_manual(self, shape):
        torch.manual_seed(42)
        dim = shape[-1]
        norm = nn.LayerNorm(dim)
        x = torch.randn(*shape)
        scale = torch.randn(*shape)
        shift = torch.randn(*shape)

        # 手动参考: norm(x) * (1 + scale) + shift
        expected = norm(x) * (1 + scale) + shift

        result = fused_layernorm_scale_shift(norm, x, scale, shift)
        assert_tensor_equal(result, expected)

    def test_zero_scale_shift(self):
        """scale=0, shift=0 应等同于 norm(x)。"""
        torch.manual_seed(7)
        dim = 32
        norm = nn.LayerNorm(dim)
        x = torch.randn(2, 4, dim)
        scale = torch.zeros(2, 4, dim)
        shift = torch.zeros(2, 4, dim)

        expected = norm(x)
        result = fused_layernorm_scale_shift(norm, x, scale, shift)
        assert_tensor_equal(result, expected)


# ---------------------------------------------------------------------------
# fused_rotary_embedding tests
# ---------------------------------------------------------------------------


class TestFusedRotaryEmbedding:
    """验证 fused_rotary_embedding GPU fallback 与原始 apply_rotary_emb_qwen 一致。"""

    def _make_complex_freqs(self, seq_len: int, dim: int) -> torch.Tensor:
        """生成 complex 格式的 freqs_cis [S, D/2]。"""
        half_dim = dim // 2
        freqs = torch.randn(seq_len, half_dim)
        # 转为 complex: e^(i*theta) 形式
        angles = torch.randn(seq_len, half_dim)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        return freqs_cis

    def _make_real_freqs(self, num_positions: int, dim: int):
        """生成 real 格式的 (cos, sin) freqs [num_positions, D]。

        注意: use_real 路径中 cos[None, None] 形成 [1,1,N,D]，
        与 x [B,S,H,D] 做广播时 N 须等于 H。
        """
        cos = torch.randn(num_positions, dim)
        sin = torch.randn(num_positions, dim)
        return (cos, sin)

    def test_complex_mode_matches_reference(self):
        """use_real=False: 与 view_as_complex 参考实现一致。"""
        torch.manual_seed(42)
        B, S, H, D = 2, 8, 4, 64
        x = torch.randn(B, S, H, D)
        freqs_cis = self._make_complex_freqs(S, D)

        # 参考实现
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        fc = freqs_cis.unsqueeze(1)
        expected = torch.view_as_real(x_rotated * fc).flatten(3).type_as(x)

        result = fused_rotary_embedding(x, freqs_cis, use_real=False)
        assert_tensor_equal(result, expected)

    def test_real_mode_unbind_neg1(self):
        """use_real=True, use_real_unbind_dim=-1 与参考实现一致。"""
        torch.manual_seed(42)
        B, S, H, D = 2, 8, 4, 64
        x = torch.randn(B, S, H, D)
        # cos/sin 第一维须等于 H 以满足 cos[None,None] 与 x 的广播
        cos, sin = self._make_real_freqs(H, D)
        freqs_cis = (cos, sin)

        # 参考实现
        c = cos[None, None].to(x.device)
        s = sin[None, None].to(x.device)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        expected = (x.float() * c + x_rotated.float() * s).to(x.dtype)

        result = fused_rotary_embedding(x, freqs_cis, use_real=True, use_real_unbind_dim=-1)
        assert_tensor_equal(result, expected)

    def test_real_mode_unbind_neg2(self):
        """use_real=True, use_real_unbind_dim=-2 与参考实现一致。"""
        torch.manual_seed(42)
        B, S, H, D = 2, 8, 4, 64
        x = torch.randn(B, S, H, D)
        # cos/sin 第一维须等于 H 以满足 cos[None,None] 与 x 的广播
        cos, sin = self._make_real_freqs(H, D)
        freqs_cis = (cos, sin)

        # 参考实现
        c = cos[None, None].to(x.device)
        s = sin[None, None].to(x.device)
        x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
        x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        expected = (x.float() * c + x_rotated.float() * s).to(x.dtype)

        result = fused_rotary_embedding(x, freqs_cis, use_real=True, use_real_unbind_dim=-2)
        assert_tensor_equal(result, expected)

    def test_invalid_unbind_dim_raises(self):
        """无效的 use_real_unbind_dim 应抛出 ValueError。"""
        x = torch.randn(1, 4, 2, 8)
        freqs_cis = (torch.randn(4, 8), torch.randn(4, 8))

        with pytest.raises(ValueError, match="use_real_unbind_dim"):
            fused_rotary_embedding(x, freqs_cis, use_real=True, use_real_unbind_dim=0)

    def test_output_shape_preserved(self):
        """输出形状与输入一致。"""
        B, S, H, D = 1, 16, 8, 128
        x = torch.randn(B, S, H, D)
        freqs_cis = self._make_complex_freqs(S, D)

        result = fused_rotary_embedding(x, freqs_cis, use_real=False)
        assert result.shape == x.shape
