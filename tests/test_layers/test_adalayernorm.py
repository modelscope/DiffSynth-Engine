import unittest

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


if __name__ == "__main__":
    unittest.main()