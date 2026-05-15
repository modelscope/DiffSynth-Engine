import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from diffsynth_engine.layers.norm import AdaLayerNorm


class TestAdaLayerNorm(unittest.TestCase):
    """Test AdaLayerNorm wrapper class"""

    def test_forward_with_2d_scale_shift(self):
        """Test AdaLayerNorm with [B, H] scale and shift"""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.randn(2, 64)
        shift = torch.randn(2, 64)

        output = adaln(hidden_states, scale, shift)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_forward_with_3d_scale_shift(self):
        """Test AdaLayerNorm with [B, 1, H] scale and shift"""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.randn(2, 1, 64)  # 3D
        shift = torch.randn(2, 1, 64)  # 3D

        output = adaln(hidden_states, scale, shift)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_forward_mixed_scale_shift(self):
        """Test AdaLayerNorm with [B, H] scale and [B, 1, H] shift"""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.randn(2, 64)  # 2D
        shift = torch.randn(2, 1, 64)  # 3D

        output = adaln(hidden_states, scale, shift)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_adalayernorm_vs_manual(self):
        """Test that AdaLayerNorm output matches manual implementation"""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.randn(2, 64)
        shift = torch.randn(2, 64)

        # Get output from AdaLayerNorm
        output = adaln(hidden_states, scale, shift)

        # Manual implementation for comparison
        normed = layernorm(hidden_states)
        scale_expanded = scale.unsqueeze(1)  # [B, H] -> [B, 1, H]
        shift_expanded = shift.unsqueeze(1)
        expected = normed * (1 + scale_expanded) + shift_expanded

        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_different_batch_size(self):
        """Test AdaLayerNorm with different batch sizes"""
        layernorm = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        for batch_size in [1, 4, 8]:
            hidden_states = torch.randn(batch_size, 32, 128)
            scale = torch.randn(batch_size, 128)
            shift = torch.randn(batch_size, 128)

            output = adaln(hidden_states, scale, shift)

            self.assertEqual(output.shape, hidden_states.shape)
            self.assertFalse(torch.isnan(output).any())

    # ----- Edge case tests -----

    def test_scale_negative_one(self):
        """scale=-1 → 1+scale=0 → output equals shift alone."""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = -torch.ones(2, 64)
        shift = torch.randn(2, 64)

        output = adaln(hidden_states, scale, shift)

        # normed * (1 + (-1)) + shift = 0 + shift = shift
        expected = shift.unsqueeze(1)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_zero_scale_and_shift(self):
        """scale=0, shift=0 → output = layernorm(x)."""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.zeros(2, 64)
        shift = torch.zeros(2, 64)

        output = adaln(hidden_states, scale, shift)
        expected = layernorm(hidden_states)

        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_eps_propagation(self):
        """Different layernorm eps values produce different outputs."""
        layernorm_small = nn.LayerNorm(64, elementwise_affine=False, eps=1e-8)
        layernorm_large = nn.LayerNorm(64, elementwise_affine=False, eps=1e-3)

        adaln_small = AdaLayerNorm(layernorm_small)
        adaln_large = AdaLayerNorm(layernorm_large)

        # Use a zero-mean input to make eps matter
        hidden_states = torch.zeros(2, 16, 64)
        hidden_states[0, 0, 0] = 1.0  # slight perturbation

        scale = torch.zeros(2, 64)
        shift = torch.zeros(2, 64)

        out_small = adaln_small(hidden_states, scale, shift)
        out_large = adaln_large(hidden_states, scale, shift)

        self.assertFalse(torch.allclose(out_small, out_large, atol=1e-5))

    def test_large_seq_len(self):
        """Works with large sequence length."""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 1024, 64)
        scale = torch.randn(2, 64)
        shift = torch.randn(2, 64)

        output = adaln(hidden_states, scale, shift)
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_dtype_preserved(self):
        """Output dtype matches input."""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            hidden_states = torch.randn(2, 16, 64, dtype=dtype)
            scale = torch.randn(2, 64, dtype=dtype)
            shift = torch.randn(2, 64, dtype=dtype)

            output = adaln(hidden_states, scale, shift)
            self.assertEqual(output.dtype, dtype)

    def test_hidden_dim_mismatch_raises(self):
        """LayerNorm hidden_size mismatch with scale/shift last dim → error."""
        layernorm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 32)
        scale = torch.randn(2, 64)  # Wrong dim
        shift = torch.randn(2, 64)  # Wrong dim

        with self.assertRaises(RuntimeError):
            adaln(hidden_states, scale, shift)

    # ----- NPU path mock tests -----

    @patch(
        "diffsynth_engine.layers.norm.is_npu_available", return_value=True
    )
    def test_npu_path_calls_layernorm_scale_shift(self, _mock_npu):
        """NPU path calls mindiesd layernorm_scale_shift with correct args."""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.randn(2, 64)
        shift = torch.randn(2, 64)

        with patch(
            "diffsynth_engine.layers.norm.layernorm_scale_shift",
            return_value=hidden_states.clone(),
            create=True,
        ) as mock_ls:
            output = adaln(hidden_states, scale, shift)

            mock_ls.assert_called_once()
            _, kwargs = mock_ls.call_args
            self.assertIs(kwargs["layernorm"], layernorm)
            self.assertTrue(kwargs["fused"])
            self.assertIs(kwargs["x"], hidden_states)
            self.assertIs(kwargs["scale"], scale)
            self.assertIs(kwargs["shift"], shift)

    @patch(
        "diffsynth_engine.layers.norm.is_npu_available", return_value=True
    )
    def test_npu_path_handles_layernorm_scale_shift_none(self, _mock_npu):
        """When layernorm_scale_shift is None, falls back to manual path."""
        layernorm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        adaln = AdaLayerNorm(layernorm)

        hidden_states = torch.randn(2, 16, 64)
        scale = torch.randn(2, 64)
        shift = torch.randn(2, 64)

        with patch(
            "diffsynth_engine.layers.norm.layernorm_scale_shift", None
        ):
            output = adaln(hidden_states, scale, shift)

        # Should use fallback path without crashing
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output).any())


if __name__ == "__main__":
    unittest.main()