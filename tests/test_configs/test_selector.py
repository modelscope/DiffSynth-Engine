import unittest
from unittest.mock import patch, MagicMock

import torch

from diffsynth_engine.layers.attention import AttentionType
from diffsynth_engine.layers.attention.selector import get_attn_backend


class TestGetAttnBackendSelector(unittest.TestCase):
    """Test get_attn_backend selector auto-detect logic."""

    def setUp(self):
        get_attn_backend.cache_clear()

    def tearDown(self):
        get_attn_backend.cache_clear()

    @patch(
        "diffsynth_engine.layers.attention.selector.is_npu_available",
        return_value=True,
    )
    def test_auto_detect_npu_selects_mindie(self, _mock):
        """attn_type=None on NPU → MINDIE (requires NPU available check patched)."""
        with patch(
            "diffsynth_engine.layers.attention.backends.mindie_attn.is_npu_available",
            return_value=True,
        ):
            backend = get_attn_backend(head_size=64, attn_type=None)
            self.assertEqual(backend.get_type(), AttentionType.MINDIE)

    @patch(
        "diffsynth_engine.layers.attention.selector.is_npu_available",
        return_value=False,
    )
    def test_auto_detect_non_npu_selects_sdpa(self, _mock):
        """attn_type=None on non-NPU → SDPA."""
        backend = get_attn_backend(head_size=64, attn_type=None)
        self.assertEqual(backend.get_type(), AttentionType.SDPA)

    @patch(
        "diffsynth_engine.layers.attention.selector.is_npu_available",
        return_value=True,
    )
    def test_explicit_sdpa_on_npu_not_overridden(self, _mock):
        """Explicit attn_type=SDPA on NPU → SDPA (not overridden)."""
        backend = get_attn_backend(head_size=64, attn_type=AttentionType.SDPA)
        self.assertEqual(backend.get_type(), AttentionType.SDPA)

    @patch(
        "diffsynth_engine.layers.attention.selector.is_npu_available",
        return_value=True,
    )
    def test_explicit_mindie_on_npu(self, _mock):
        """Explicit attn_type=MINDIE on NPU → MINDIE."""
        with patch(
            "diffsynth_engine.layers.attention.backends.mindie_attn.is_npu_available",
            return_value=True,
        ):
            backend = get_attn_backend(head_size=64, attn_type=AttentionType.MINDIE)
            self.assertEqual(backend.get_type(), AttentionType.MINDIE)

    @patch(
        "diffsynth_engine.layers.attention.selector.is_npu_available",
        return_value=False,
    )
    def test_explicit_mindie_on_non_npu_raises(self, _mock):
        """Explicit MINDIE on non-NPU → RuntimeError."""
        with self.assertRaises(RuntimeError):
            get_attn_backend(head_size=64, attn_type=AttentionType.MINDIE)

    def test_mindie_in_registry(self):
        """MINDIE backend is registered in the backends dict."""
        from diffsynth_engine.layers.attention.selector import _attention_backends
        self.assertIn(AttentionType.MINDIE, _attention_backends)

    @patch(
        "diffsynth_engine.layers.attention.selector.is_npu_available",
        return_value=False,
    )
    def test_auto_detect_non_npu_with_fa2(self, _mock):
        """Explicit FA2 on non-NPU works (backends loaded on demand)."""
        with self.assertRaises(RuntimeError):
            # FA2 check_availability will fail without flash_attn installed
            get_attn_backend(head_size=64, attn_type=AttentionType.FA2)


if __name__ == "__main__":
    unittest.main()