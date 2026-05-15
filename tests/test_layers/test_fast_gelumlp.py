import unittest

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward

from diffsynth_engine.layers.mlp import FastGELUMLP, _GELUProj


class TestFastGELUMLP(unittest.TestCase):
    """Test FastGELUMLP NPU wrapper."""

    def setUp(self):
        self.dim = 64
        self.batch_size = 2
        self.seq_len = 16

    def _make_input(self, batch=None, seq_len=None):
        b = batch or self.batch_size
        s = seq_len or self.seq_len
        return torch.randn(b, s, self.dim)

    def test_forward_output_shape(self):
        """Output shape matches input shape."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input()
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_no_nan(self):
        """Output contains no NaN."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input()
        out = mlp(x)
        self.assertFalse(torch.isnan(out).any())

    def test_forward_no_inf(self):
        """Output contains no Inf."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input()
        out = mlp(x)
        self.assertFalse(torch.isinf(out).any())

    def test_equivalence_to_feedforward(self):
        """FastGELUMLP output matches diffusers FeedForward with same weights."""
        x = self._make_input()

        diffusers_ff = FeedForward(
            dim=self.dim, dim_out=self.dim, activation_fn="gelu-approximate"
        )
        our_mlp = FastGELUMLP(self.dim, dim_out=self.dim)

        # Copy weights: diffusers net[0].proj → our net[0].proj
        with torch.no_grad():
            our_mlp.net[0].proj.weight.copy_(diffusers_ff.net[0].proj.weight)
            our_mlp.net[0].proj.bias.copy_(diffusers_ff.net[0].proj.bias)
            our_mlp.net[2].weight.copy_(diffusers_ff.net[2].weight)
            our_mlp.net[2].bias.copy_(diffusers_ff.net[2].bias)

        diffusers_out = diffusers_ff(x)
        our_out = our_mlp(x)

        self.assertTrue(torch.allclose(our_out, diffusers_out, atol=1e-5))

    def test_checkpoint_key_compatibility(self):
        """state_dict keys match diffusers FeedForward for checkpoint loading."""
        diffusers_ff = FeedForward(
            dim=self.dim, dim_out=self.dim, activation_fn="gelu-approximate"
        )
        our_mlp = FastGELUMLP(self.dim, dim_out=self.dim)

        diffusers_keys = set(diffusers_ff.state_dict().keys())
        our_keys = set(our_mlp.state_dict().keys())

        self.assertEqual(diffusers_keys, our_keys)

    def test_fallback_gelu_matches_manual(self):
        """Fallback path uses F.gelu(approximate='tanh')."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input()

        # Manually compute what fallback does
        projected = mlp.net[0].proj(x)
        manual_gelu = F.gelu(projected, approximate="tanh")
        manual_out = mlp.net[2](manual_gelu)

        our_out = mlp(x)
        self.assertTrue(torch.allclose(our_out, manual_out, atol=1e-5))

    def test_mult_parameter(self):
        """mult controls inner_dim = dim * mult."""
        for mult in [2, 4, 8]:
            mlp = FastGELUMLP(self.dim, mult=mult)
            self.assertEqual(mlp.net[0].proj.out_features, self.dim * mult)
            self.assertEqual(mlp.net[2].in_features, self.dim * mult)

    def test_dim_out_custom(self):
        """dim_out != dim produces correct output shape."""
        dim_out = 128
        mlp = FastGELUMLP(self.dim, dim_out=dim_out, mult=4)
        x = self._make_input()
        out = mlp(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, dim_out))

    def test_dim_out_defaults_to_dim(self):
        """dim_out defaults to dim when not specified."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input()
        out = mlp(x)
        self.assertEqual(out.shape[-1], self.dim)

    def test_dropout_is_zero(self):
        """Dropout probability is 0.0 (inactive)."""
        mlp = FastGELUMLP(self.dim)
        self.assertEqual(mlp.net[1].p, 0.0)

    def test_dropout_inactive_in_train_mode(self):
        """Even in train mode, Dropout(0.0) doesn't change output."""
        mlp = FastGELUMLP(self.dim)
        mlp.train()
        x = self._make_input()

        out1 = mlp(x)
        out2 = mlp(x)

        self.assertTrue(torch.equal(out1, out2))

    def test_batch_size_1(self):
        """Works with batch size 1."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input(batch=1)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_large_batch(self):
        """Works with batch size 8."""
        mlp = FastGELUMLP(self.dim)
        x = self._make_input(batch=8)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)


class TestGELUProj(unittest.TestCase):
    """Test _GELUProj wrapper class."""

    def test_proj_linear_registered(self):
        """_GELUProj has .proj attribute matching diffusers structure."""
        dim, inner_dim = 64, 256
        mod = _GELUProj(dim, inner_dim)
        self.assertIsInstance(mod.proj, torch.nn.Linear)
        self.assertEqual(mod.proj.in_features, dim)
        self.assertEqual(mod.proj.out_features, inner_dim)

    def test_forward_applies_gelu_approximate(self):
        """_GELUProj.forward applies F.gelu(approximate='tanh')."""
        dim, inner_dim = 64, 256
        mod = _GELUProj(dim, inner_dim)
        x = torch.randn(2, 16, dim)

        out = mod(x)
        expected = F.gelu(x, approximate="tanh")

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_proj_not_called_in_forward(self):
        """_GELUProj.forward does NOT call proj—only applies GELU."""
        dim, inner_dim = 64, 256
        mod = _GELUProj(dim, inner_dim)
        x = torch.randn(2, 16, dim)  # dim-sized, NOT inner_dim

        # If proj were called, this would fail on shape mismatch
        out = mod(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
