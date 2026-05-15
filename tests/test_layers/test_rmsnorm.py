import unittest

import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

from diffsynth_engine.layers.norm import RMSNorm


class TestRMSNorm(unittest.TestCase):
    """Test RMSNorm NPU wrapper."""

    def setUp(self):
        self.hidden_size = 64
        self.eps = 1e-6
        self.batch_size = 2
        self.seq_len = 16

    def _make_input(self, batch=None):
        b = batch or self.batch_size
        return torch.randn(b, self.seq_len, self.hidden_size)

    def test_forward_output_shape(self):
        norm = RMSNorm(self.hidden_size, self.eps)
        x = self._make_input()
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_no_nan(self):
        norm = RMSNorm(self.hidden_size, self.eps)
        x = self._make_input()
        out = norm(x)
        self.assertFalse(torch.isnan(out).any())

    def test_equivalence_to_diffusers_rmsnorm(self):
        """Same weight → same output as DiffusersRMSNorm."""
        x = self._make_input()
        diffusers_norm = DiffusersRMSNorm(self.hidden_size, self.eps)
        our_norm = RMSNorm(self.hidden_size, self.eps)

        with torch.no_grad():
            our_norm.weight.copy_(diffusers_norm.weight)

        diffusers_out = diffusers_norm(x)
        our_out = our_norm(x)

        self.assertTrue(torch.allclose(our_out, diffusers_out, atol=1e-5))

    def test_weight_sharing_with_fallback(self):
        """self.weight and self._fallback.weight share the same storage."""
        norm = RMSNorm(self.hidden_size, self.eps)
        self.assertIs(norm.weight, norm._fallback.weight)

        with torch.no_grad():
            new_weight = torch.randn_like(norm.weight)
            norm.weight.copy_(new_weight)

        self.assertTrue(torch.equal(norm.weight, norm._fallback.weight))

    def test_checkpoint_load_restore(self):
        """load_state_dict applies weights to both paths via strict=False."""
        norm = RMSNorm(self.hidden_size, self.eps)
        ref = DiffusersRMSNorm(self.hidden_size, self.eps)
        x = self._make_input()

        # Load diffusers state into our norm (strict=False because
        # _fallback.weight is not a direct parameter - it's aliased)
        norm.load_state_dict(ref.state_dict(), strict=False)

        out_our = norm(x)
        out_ref = ref(x)
        self.assertTrue(torch.allclose(out_our, out_ref, atol=1e-5))

    def test_eps_propagation(self):
        """Different eps values produce different outputs."""
        x = self._make_input()

        norm_small_eps = RMSNorm(self.hidden_size, eps=1e-8)
        norm_large_eps = RMSNorm(self.hidden_size, eps=1e-3)

        with torch.no_grad():
            norm_large_eps.weight.copy_(norm_small_eps.weight)

        out1 = norm_small_eps(x)
        out2 = norm_large_eps(x)

        self.assertFalse(torch.allclose(out1, out2, atol=1e-5))

    def test_batch_size_1(self):
        norm = RMSNorm(self.hidden_size, self.eps)
        x = self._make_input(batch=1)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_state_dict_no_fallback_keys(self):
        """state_dict must NOT contain _fallback.* keys for strict checkpoint loading."""
        norm = RMSNorm(self.hidden_size, self.eps)
        sd = norm.state_dict()
        self.assertIn("weight", sd)
        fallback_keys = [k for k in sd if "_fallback" in k]
        self.assertEqual(fallback_keys, [], f"unexpected keys: {fallback_keys}")

    def test_strict_load_state_dict(self):
        """strict=True loading from DiffusersRMSNorm state_dict must succeed."""
        ref = DiffusersRMSNorm(self.hidden_size, self.eps)
        norm = RMSNorm(self.hidden_size, self.eps)
        norm.load_state_dict(ref.state_dict(), strict=True)
        self.assertTrue(torch.equal(norm.weight, ref.weight))

    def test_different_hidden_sizes(self):
        for hidden_size in [32, 128, 256]:
            norm = RMSNorm(hidden_size, self.eps)
            x = torch.randn(2, 8, hidden_size)
            out = norm(x)
            self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()