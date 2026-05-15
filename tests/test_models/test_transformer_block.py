import unittest
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from diffsynth_engine.models.qwen_image.transformer_qwenimage import (
    QwenImageTransformerBlock,
)
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.layers.attention import AttentionType


def _make_context(attn_type=None):
    return set_forward_context(attn_type=attn_type)


class TestGateBroadcastFix(unittest.TestCase):
    """Test that gate tensors are properly broadcast from [B, dim] to [B, 1, dim]."""

    def setUp(self):
        self.B, self.S, self.D = 4, 32, 64

    def test_chunk_produces_2d_gates(self):
        """mod_params.chunk(3, dim=-1) on [B, 3*dim] produces [B, dim] gates."""
        mod_params = torch.randn(self.B, 3 * self.D)
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        self.assertEqual(gate.shape, (self.B, self.D))
        self.assertEqual(gate.dim(), 2)

    def test_gate_unsqueeze_enables_broadcast(self):
        """gate [B, dim].unsqueeze(1) → [B, 1, dim] broadcasts with [B, S, dim]."""
        gate = torch.randn(self.B, self.D)
        output = torch.randn(self.B, self.S, self.D)
        gate_3d = gate.unsqueeze(1)
        self.assertEqual(gate_3d.shape, (self.B, 1, self.D))
        result = gate_3d * output
        self.assertEqual(result.shape, (self.B, self.S, self.D))

    def test_gate_without_unsqueeze_fails_for_batch_gt_1(self):
        """Without unsqueeze, [B, D] * [B, S, D] raises RuntimeError when B != S."""
        gate = torch.randn(2, 64)
        output = torch.randn(2, 16, 64)
        with self.assertRaises(RuntimeError):
            _ = gate * output

    def test_gate_without_unsqueeze_works_for_batch_1_only(self):
        """Without unsqueeze, [1, D] * [1, S, D] works (1 broadcasts to S)."""
        gate = torch.randn(1, 64)
        output = torch.randn(1, 16, 64)
        result = gate * output
        self.assertEqual(result.shape, (1, 16, 64))

    def test_all_four_gate_sites(self):
        """All 4 gate multiply sites require unsqueeze for correct broadcast."""
        gate = torch.randn(self.B, self.D)
        attn_out = torch.randn(self.B, self.S, self.D)
        gate_bc = gate.unsqueeze(1)
        self.assertEqual(gate_bc.shape, (self.B, 1, self.D))
        residual = torch.randn(self.B, self.S, self.D)
        new_state = residual + gate_bc * attn_out
        self.assertEqual(new_state.shape, (self.B, self.S, self.D))


class TestZeroCondTFix(unittest.TestCase):
    """Test that zero_cond_t chunk happens before img_mod_params computation."""

    def setUp(self):
        self.B, self.S, self.D = 2, 32, 64

    def test_chunk_reduces_batch_size(self):
        """torch.chunk(temb, 2) on [2*B, D] gives two [B, D] tensors."""
        B = 2
        temb = torch.randn(2 * B, self.D)
        chunks = torch.chunk(temb, 2, dim=0)
        self.assertEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c.shape, (B, self.D))

    def test_img_mod_uses_half_batch_after_chunk(self):
        """After zero_cond_t chunk, img_mod produces [B, 6*dim] not [2*B, 6*dim]."""
        B = 2
        temb = torch.randn(2 * B, self.D)
        temb_chunked = torch.chunk(temb, 2, dim=0)[0]
        img_mod = nn.Sequential(nn.SiLU(), nn.Linear(self.D, 6 * self.D))
        img_mod_params = img_mod(temb_chunked)
        self.assertEqual(img_mod_params.shape, (B, 6 * self.D))

    def test_img_mod_without_chunk_produces_double_batch(self):
        """Without chunk, img_mod produces [2*B, 6*dim] which crashes AdaLayerNorm."""
        B = 2
        temb = torch.randn(2 * B, self.D)
        img_mod = nn.Sequential(nn.SiLU(), nn.Linear(self.D, 6 * self.D))
        img_mod_params = img_mod(temb)
        self.assertEqual(img_mod_params.shape, (2 * B, 6 * self.D))


class TestTransformerBlockForward(unittest.TestCase):
    """Integration test for QwenImageTransformerBlock forward pass."""

    def setUp(self):
        self.dim = 64
        self.num_heads = 8
        self.head_dim = self.dim // self.num_heads
        self.B, self.S_img, self.S_txt = 2, 16, 8
        self.eps = 1e-6

    def _make_block(self, zero_cond_t=False):
        with _make_context(attn_type=AttentionType.SDPA):
            return QwenImageTransformerBlock(
                dim=self.dim,
                num_attention_heads=self.num_heads,
                attention_head_dim=self.head_dim,
                qk_norm="rms_norm",
                eps=self.eps,
                zero_cond_t=zero_cond_t,
            )

    @patch(
        "diffsynth_engine.models.qwen_image.transformer_qwenimage.QwenDoubleStreamAttention"
    )
    def test_forward_no_crash_zero_cond_t_false(self, mock_attn_cls):
        mock_attn = MagicMock()
        mock_attn.return_value = (
            torch.randn(self.B, self.S_img, self.dim),
            torch.randn(self.B, self.S_txt, self.dim),
        )
        mock_attn_cls.return_value = mock_attn

        block = self._make_block(zero_cond_t=False)
        block.attn = mock_attn

        hidden_states = torch.randn(self.B, self.S_img, self.dim)
        encoder_hidden_states = torch.randn(self.B, self.S_txt, self.dim)
        encoder_hidden_states_mask = torch.ones(self.B, self.S_txt, dtype=torch.bool)
        temb = torch.randn(self.B, self.dim)

        with _make_context(attn_type=AttentionType.SDPA):
            txt_out, img_out = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
            )

        self.assertEqual(txt_out.shape, encoder_hidden_states.shape)
        self.assertEqual(img_out.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(txt_out).any())
        self.assertFalse(torch.isnan(img_out).any())

    @patch(
        "diffsynth_engine.models.qwen_image.transformer_qwenimage.QwenDoubleStreamAttention"
    )
    def test_forward_no_crash_zero_cond_t_true(self, mock_attn_cls):
        mock_attn = MagicMock()
        mock_attn.return_value = (
            torch.randn(self.B, self.S_img, self.dim),
            torch.randn(self.B, self.S_txt, self.dim),
        )
        mock_attn_cls.return_value = mock_attn

        block = self._make_block(zero_cond_t=True)
        block.attn = mock_attn

        hidden_states = torch.randn(self.B, self.S_img, self.dim)
        encoder_hidden_states = torch.randn(self.B, self.S_txt, self.dim)
        encoder_hidden_states_mask = torch.ones(self.B, self.S_txt, dtype=torch.bool)
        temb = torch.randn(2 * self.B, self.dim)

        with _make_context(attn_type=AttentionType.SDPA):
            txt_out, img_out = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
            )

        self.assertEqual(txt_out.shape, encoder_hidden_states.shape)
        self.assertEqual(img_out.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(txt_out).any())
        self.assertFalse(torch.isnan(img_out).any())

    @patch(
        "diffsynth_engine.models.qwen_image.transformer_qwenimage.QwenDoubleStreamAttention"
    )
    def test_forward_preserves_residual_connection(self, mock_attn_cls):
        mock_attn = MagicMock()
        mock_attn.return_value = (
            torch.randn(self.B, self.S_img, self.dim) * 0.1,
            torch.randn(self.B, self.S_txt, self.dim) * 0.1,
        )
        mock_attn_cls.return_value = mock_attn

        block = self._make_block(zero_cond_t=False)
        block.attn = mock_attn

        hidden_states = torch.randn(self.B, self.S_img, self.dim)
        encoder_hidden_states = torch.randn(self.B, self.S_txt, self.dim)
        encoder_hidden_states_mask = torch.ones(self.B, self.S_txt, dtype=torch.bool)
        temb = torch.randn(self.B, self.dim)

        with _make_context(attn_type=AttentionType.SDPA):
            txt_out, img_out = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
            )

        self.assertFalse(torch.equal(img_out, hidden_states))
        self.assertFalse(torch.equal(txt_out, encoder_hidden_states))

    @patch(
        "diffsynth_engine.models.qwen_image.transformer_qwenimage.QwenDoubleStreamAttention"
    )
    def test_forward_fp16_clip_applied(self, mock_attn_cls):
        """fp16 tensors pass through block without overflow/NaN."""
        mock_attn = MagicMock()
        mock_attn.return_value = (
            torch.randn(self.B, self.S_img, self.dim, dtype=torch.float16),
            torch.randn(self.B, self.S_txt, self.dim, dtype=torch.float16),
        )
        mock_attn_cls.return_value = mock_attn

        block = self._make_block(zero_cond_t=False)
        block.attn = mock_attn

        hidden_states = torch.randn(self.B, self.S_img, self.dim, dtype=torch.float16)
        encoder_hidden_states = torch.randn(
            self.B, self.S_txt, self.dim, dtype=torch.float16
        )
        encoder_hidden_states_mask = torch.ones(
            self.B, self.S_txt, dtype=torch.bool
        )
        temb = torch.randn(self.B, self.dim)

        with _make_context(attn_type=AttentionType.SDPA):
            txt_out, img_out = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
            )

        self.assertFalse(torch.isinf(txt_out).any())
        self.assertFalse(torch.isinf(img_out).any())

    def test_adalayernorm_integration(self):
        """Block uses AdaLayerNorm instances (not raw nn.LayerNorm)."""
        block = self._make_block(zero_cond_t=False)

        from diffsynth_engine.layers.norm import AdaLayerNorm

        self.assertIsInstance(block.img_norm1, AdaLayerNorm)
        self.assertIsInstance(block.img_norm2, AdaLayerNorm)
        self.assertIsInstance(block.txt_norm1, AdaLayerNorm)
        self.assertIsInstance(block.txt_norm2, AdaLayerNorm)

    def test_fast_gelumlp_integration(self):
        """Block uses FastGELUMLP instances (not diffusers FeedForward)."""
        block = self._make_block(zero_cond_t=False)

        from diffsynth_engine.layers.mlp import FastGELUMLP

        self.assertIsInstance(block.img_mlp, FastGELUMLP)
        self.assertIsInstance(block.txt_mlp, FastGELUMLP)

    def test_mod_params_dimension(self):
        """img_mod and txt_mod output [B, 6*dim]."""
        block = self._make_block(zero_cond_t=False)
        temb = torch.randn(self.B, self.dim)

        img_mod = block.img_mod(temb)
        txt_mod = block.txt_mod(temb)

        self.assertEqual(img_mod.shape, (self.B, 6 * self.dim))
        self.assertEqual(txt_mod.shape, (self.B, 6 * self.dim))


class TestModulatePreserved(unittest.TestCase):
    """Test that _modulate method is preserved for zero_cond_t CFG support."""

    def setUp(self):
        self.B, self.S, self.D = 2, 32, 64

    def test_modulate_without_index(self):
        """_modulate without index works correctly."""
        with _make_context(attn_type=AttentionType.SDPA):
            block = QwenImageTransformerBlock(
                dim=self.D,
                num_attention_heads=8,
                attention_head_dim=8,
                eps=1e-6,
                zero_cond_t=False,
            )
        x = torch.randn(self.B, self.S, self.D)
        mod_params = torch.randn(self.B, 3 * self.D)

        modulated, gate = block._modulate(x, mod_params)

        self.assertEqual(modulated.shape, x.shape)
        self.assertEqual(gate.shape, (self.B, 1, self.D))

        shift, scale, gate_raw = mod_params.chunk(3, dim=-1)
        expected = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        self.assertTrue(torch.allclose(modulated, expected, atol=1e-5))

    def test_modulate_with_index(self):
        """_modulate with index uses per-token conditional gating."""
        with _make_context(attn_type=AttentionType.SDPA):
            block = QwenImageTransformerBlock(
                dim=self.D,
                num_attention_heads=8,
                attention_head_dim=8,
                eps=1e-6,
                zero_cond_t=True,
            )
        x = torch.randn(self.B, self.S, self.D)
        mod_params = torch.randn(2 * self.B, 3 * self.D)
        index = torch.zeros(self.B, self.S, dtype=torch.long)

        modulated, gate = block._modulate(x, mod_params, index=index)

        self.assertEqual(modulated.shape, x.shape)
        self.assertEqual(gate.shape, (self.B, self.S, self.D))


if __name__ == "__main__":
    unittest.main()