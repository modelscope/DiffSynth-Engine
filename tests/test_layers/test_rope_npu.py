import sys
import unittest
from unittest.mock import patch, MagicMock

import torch

from diffsynth_engine.models.qwen_image.transformer_qwenimage import apply_rotary_emb_qwen


class TestApplyRotaryEmbQwen(unittest.TestCase):
    """Test RoPE function with NPU and fallback paths."""

    def setUp(self):
        self.B, self.S, self.H, self.D = 2, 32, 8, 64
        self.x = torch.randn(self.B, self.S, self.H, self.D)

    def _make_freqs_cis_real(self):
        cos = torch.randn(self.S, self.D)
        sin = torch.randn(self.S, self.D)
        return (cos, sin)

    def _make_freqs_cis_complex(self):
        freqs_cis = torch.randn(self.S, self.D // 2).to(torch.complex64)
        return freqs_cis

    def test_cos_sin_broadcast_shape(self):
        """cos/sin are broadcast from [S, D] to [1, S, 1, D] to match [B, S, H, D]."""
        freqs_cis = self._make_freqs_cis_real()
        cos, sin = freqs_cis

        # cos[None, :, None, :] from [S, D] → [1, S, 1, D]
        # S in the test is self.S = 32, NOT self.S_img (which doesn't exist)
        cos_bc = cos[None, :, None, :]
        self.assertEqual(cos_bc.shape, (1, self.S, 1, self.D))

    def test_use_real_unbind_minus1_fallback(self):
        """use_real_unbind_dim=-1 path produces valid output (non-NPU)."""
        freqs_cis = self._make_freqs_cis_real()
        with patch(
            "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
            return_value=False,
        ):
            out = apply_rotary_emb_qwen(
                self.x, freqs_cis, use_real=True, use_real_unbind_dim=-1
            )
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_use_real_unbind_minus2_fallback(self):
        """use_real_unbind_dim=-2 path produces valid output (non-NPU)."""
        freqs_cis = self._make_freqs_cis_real()
        with patch(
            "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
            return_value=False,
        ):
            out = apply_rotary_emb_qwen(
                self.x, freqs_cis, use_real=True, use_real_unbind_dim=-2
            )
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_use_real_fallback_output_matches_reference(self):
        """Fallback output matches original pre-fix implementation."""
        freqs_cis = self._make_freqs_cis_real()
        cos, sin = freqs_cis

        with patch(
            "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
            return_value=False,
        ):
            out = apply_rotary_emb_qwen(
                self.x, freqs_cis, use_real=True, use_real_unbind_dim=-1
            )

        # Reference: the new code does cos[None, :, None, :] (correct)
        cos_bc = cos[None, :, None, :].to(self.x.device)
        sin_bc = sin[None, :, None, :].to(self.x.device)
        x_real, x_imag = self.x.reshape(*self.x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        expected = (self.x.float() * cos_bc + x_rotated.float() * sin_bc).to(self.x.dtype)

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_use_real_invalid_unbind_dim(self):
        """use_real_unbind_dim not -1 or -2 → ValueError."""
        freqs_cis = self._make_freqs_cis_real()
        with patch(
            "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
            return_value=False,
        ):
            with self.assertRaises(ValueError) as ctx:
                apply_rotary_emb_qwen(
                    self.x, freqs_cis, use_real=True, use_real_unbind_dim=0
                )
            self.assertIn("use_real_unbind_dim must be -1 or -2", str(ctx.exception))

    def test_use_complex_fallback(self):
        """use_real=False path produces valid output."""
        freqs_cis = self._make_freqs_cis_complex()
        out = apply_rotary_emb_qwen(self.x, freqs_cis, use_real=False)
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_use_complex_fallback_matches_reference(self):
        """use_real=False output matches original implementation."""
        freqs_cis = self._make_freqs_cis_complex()
        out = apply_rotary_emb_qwen(self.x, freqs_cis, use_real=False)

        x_rotated = torch.view_as_complex(
            self.x.float().reshape(*self.x.shape[:-1], -1, 2)
        )
        freqs_cis_bc = freqs_cis.unsqueeze(1)
        expected = torch.view_as_real(x_rotated * freqs_cis_bc).flatten(3)
        expected = expected.type_as(self.x)

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_npu_real_path_calls_rotary_position_embedding(self):
        """NPU use_real=True path calls mindiesd rotary_position_embedding."""
        import types

        fake_mindiesd = types.ModuleType("mindiesd")
        fake_layers = types.ModuleType("mindiesd.layers")
        fake_rope = types.ModuleType("mindiesd.layers.rope")
        fake_mindiesd.layers = fake_layers
        fake_layers.rope = fake_rope
        for mod in [fake_mindiesd, fake_layers, fake_rope]:
            mod.__path__ = []
            mod.__file__ = f"<fake:{mod.__name__}>"

        mock_rope = MagicMock(return_value=self.x.clone())
        fake_rope.rotary_position_embedding = mock_rope

        orig = {k: sys.modules.pop(k, None) for k in
                ["mindiesd", "mindiesd.layers", "mindiesd.layers.rope"]}
        sys.modules["mindiesd"] = fake_mindiesd
        sys.modules["mindiesd.layers"] = fake_layers
        sys.modules["mindiesd.layers.rope"] = fake_rope

        try:
            freqs_cis = self._make_freqs_cis_real()
            with patch(
                "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
                return_value=True,
            ):
                apply_rotary_emb_qwen(
                    self.x, freqs_cis, use_real=True, use_real_unbind_dim=-1
                )
                mock_rope.assert_called_once()
                _, kwargs = mock_rope.call_args
                self.assertFalse(kwargs["head_first"])
                self.assertTrue(kwargs["fused"])
        finally:
            for key in ["mindiesd", "mindiesd.layers", "mindiesd.layers.rope"]:
                sys.modules.pop(key, None)
            for k, v in orig.items():
                if v is not None:
                    sys.modules[k] = v

    def test_npu_real_path_rotated_mode_mapping(self):
        """use_real_unbind_dim maps to correct rotated_mode."""
        import types

        fake_mindiesd = types.ModuleType("mindiesd")
        fake_layers = types.ModuleType("mindiesd.layers")
        fake_rope = types.ModuleType("mindiesd.layers.rope")
        fake_mindiesd.layers = fake_layers
        fake_layers.rope = fake_rope
        for mod in [fake_mindiesd, fake_layers, fake_rope]:
            mod.__path__ = []
            mod.__file__ = f"<fake:{mod.__name__}>"

        orig = {k: sys.modules.pop(k, None) for k in
                ["mindiesd", "mindiesd.layers", "mindiesd.layers.rope"]}
        sys.modules["mindiesd"] = fake_mindiesd
        sys.modules["mindiesd.layers"] = fake_layers
        sys.modules["mindiesd.layers.rope"] = fake_rope

        try:
            freqs_cis = self._make_freqs_cis_real()
            test_cases = [(-1, "rotated_half"), (-2, "rotated_interleaved")]

            for unbind_dim, expected_mode in test_cases:
                mock_rope = MagicMock(return_value=self.x.clone())
                fake_rope.rotary_position_embedding = mock_rope

                with patch(
                    "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
                    return_value=True,
                ):
                    apply_rotary_emb_qwen(
                        self.x, freqs_cis, use_real=True, use_real_unbind_dim=unbind_dim
                    )
                _, kwargs = mock_rope.call_args
                self.assertEqual(kwargs["rotated_mode"], expected_mode)
        finally:
            for key in ["mindiesd", "mindiesd.layers", "mindiesd.layers.rope"]:
                sys.modules.pop(key, None)
            for k, v in orig.items():
                if v is not None:
                    sys.modules[k] = v

    def test_dtype_preserved(self):
        """Output dtype matches input."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = self.x.to(dtype)
            freqs_cis = self._make_freqs_cis_real()
            with patch(
                "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
                return_value=False,
            ):
                out = apply_rotary_emb_qwen(
                    x, freqs_cis, use_real=True, use_real_unbind_dim=-1
                )
            self.assertEqual(out.dtype, dtype)

    def test_different_head_dims(self):
        """Works with different head dimensions."""
        for D in [32, 64, 128, 256]:
            x = torch.randn(2, 16, 4, D)
            cos = torch.randn(16, D)
            sin = torch.randn(16, D)
            with patch(
                "diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available",
                return_value=False,
            ):
                out = apply_rotary_emb_qwen(
                    x, (cos, sin), use_real=True, use_real_unbind_dim=-1
                )
            self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()