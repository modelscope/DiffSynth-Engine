import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.layers.attention.backends.mindie_attn import (
    MindieAttentionBackend,
    MindieAttentionImpl,
)


def _make_mock_module(name):
    return types.ModuleType(name)


def _install_fake_mindiesd():
    """Install fake mindiesd package hierarchy into sys.modules.

    Structure: mindiesd.layers.flash_attn.attention_forward (module)
    The attention_forward function lives as an attribute ON the module,
    so `from X import attention_forward` resolves correctly.
    """
    mindiesd = _make_mock_module("mindiesd")
    layers = _make_mock_module("mindiesd.layers")
    flash_attn = _make_mock_module("mindiesd.layers.flash_attn")
    attn_fwd = _make_mock_module("mindiesd.layers.flash_attn.attention_forward")

    mindiesd.layers = layers
    layers.flash_attn = flash_attn
    flash_attn.attention_forward = attn_fwd

    for mod in [mindiesd, layers, flash_attn, attn_fwd]:
        mod.__path__ = []
        mod.__file__ = f"<fake:{mod.__name__}>"
        sys.modules[mod.__name__] = mod


def _remove_fake_mindiesd():
    for key in ["mindiesd", "mindiesd.layers", "mindiesd.layers.flash_attn",
                "mindiesd.layers.flash_attn.attention_forward"]:
        sys.modules.pop(key, None)


class TestMindieAttentionBackend(unittest.TestCase):
    """Test MindieAttentionBackend static interface."""

    def test_get_type_returns_mindie(self):
        self.assertEqual(MindieAttentionBackend.get_type(), AttentionType.MINDIE)

    def test_get_impl_cls(self):
        self.assertIs(MindieAttentionBackend.get_impl_cls(), MindieAttentionImpl)

    def test_get_metadata_cls(self):
        self.assertIs(MindieAttentionBackend.get_metadata_cls(), AttentionMetadata)

    def test_get_builder_cls_none(self):
        self.assertIsNone(MindieAttentionBackend.get_builder_cls())

    def test_get_supported_head_sizes_empty(self):
        self.assertEqual(MindieAttentionBackend.get_supported_head_sizes(), [])

    def test_supports_head_size_any(self):
        self.assertTrue(MindieAttentionBackend.supports_head_size(64))
        self.assertTrue(MindieAttentionBackend.supports_head_size(128))
        self.assertTrue(MindieAttentionBackend.supports_head_size(256))

    def test_mindie_in_attention_type_enum(self):
        self.assertEqual(AttentionType.MINDIE.name, "MINDIE")


class TestMindieAttentionImpl(unittest.TestCase):
    """Test MindieAttentionImpl initialization and forward."""

    def _make_qkv(self, B=2, H=8, S=32, D=64):
        return (
            torch.randn(B, H, S, D),
            torch.randn(B, H, S, D),
            torch.randn(B, H, S, D),
        )

    def setUp(self):
        _install_fake_mindiesd()

    def tearDown(self):
        _remove_fake_mindiesd()

    def _install_mock_attn_forward(self):
        """Install mock attention_forward as attribute on the module.

        The SUT does `from mindiesd.layers.flash_attn.attention_forward import attention_forward`.
        By putting the mock on attn_fwd.attention_forward, the import resolves to the mock.
        """
        mock = MagicMock()
        sys.modules["mindiesd.layers.flash_attn.attention_forward"].attention_forward = mock
        return mock

    def test_init_default_kv_heads(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64)
        self.assertEqual(impl.num_kv_groups, 1)

    def test_init_gqa(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64, num_kv_heads=4)
        self.assertEqual(impl.num_kv_groups, 2)

    def test_init_stores_params(self):
        impl = MindieAttentionImpl(
            num_heads=8, head_size=64, softmax_scale=0.5,
            causal=True, num_kv_heads=2,
        )
        self.assertEqual(impl.num_heads, 8)
        self.assertEqual(impl.head_size, 64)
        self.assertEqual(impl.softmax_scale, 0.5)
        self.assertEqual(impl.causal, True)
        self.assertEqual(impl.num_kv_groups, 4)

    def test_init_scale_default_none(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64)
        self.assertIsNone(impl.softmax_scale)

    def test_init_extra_args_ignored(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64, some_extra="value")
        self.assertEqual(impl.num_heads, 8)

    def test_forward_shape(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64)
        q, k, v = self._make_qkv()
        mock = self._install_mock_attn_forward()
        mock.return_value = q.clone()

        out = impl.forward(q, k, v)
        self.assertEqual(out.shape, q.shape)

    def test_forward_scale_default(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64, softmax_scale=None)
        q, k, v = self._make_qkv()
        mock = self._install_mock_attn_forward()
        mock.return_value = q.clone()

        impl.forward(q, k, v)
        _, kwargs = mock.call_args
        self.assertAlmostEqual(kwargs["scale"], 64 ** -0.5, places=6)

    def test_forward_scale_explicit(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64, softmax_scale=0.25)
        q, k, v = self._make_qkv()
        mock = self._install_mock_attn_forward()
        mock.return_value = q.clone()

        impl.forward(q, k, v)
        _, kwargs = mock.call_args
        self.assertEqual(kwargs["scale"], 0.25)

    def test_forward_passes_attn_mask(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64)
        q, k, v = self._make_qkv()
        mask = torch.ones(2, 32, 32)
        mock = self._install_mock_attn_forward()
        mock.return_value = q.clone()

        impl.forward(q, k, v, attn_mask=mask)
        _, kwargs = mock.call_args
        self.assertIs(kwargs["attn_mask"], mask)

    def test_forward_fused_and_head_first_flags(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64)
        q, k, v = self._make_qkv()
        mock = self._install_mock_attn_forward()
        mock.return_value = q.clone()

        impl.forward(q, k, v)
        _, kwargs = mock.call_args
        self.assertTrue(kwargs["fused"])
        self.assertFalse(kwargs["head_first"])

    def test_forward_with_attn_metadata_none(self):
        impl = MindieAttentionImpl(num_heads=8, head_size=64)
        q, k, v = self._make_qkv()
        mock = self._install_mock_attn_forward()
        mock.return_value = q.clone()

        out = impl.forward(q, k, v, attn_metadata=None)
        self.assertEqual(out.shape, q.shape)


if __name__ == "__main__":
    unittest.main()