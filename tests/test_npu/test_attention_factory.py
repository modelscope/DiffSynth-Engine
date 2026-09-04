"""Unit tests for the attention factory function."""

from unittest.mock import patch

import pytest

from diffsynth_engine.layers.attention.factory import create_parallel_attention
from diffsynth_engine.layers.attention.layer import USPAttention


class TestCreateParallelAttention:
    """Tests for create_parallel_attention factory function."""

    def test_import_from_factory_module(self):
        """Verify direct import from factory module works."""
        from diffsynth_engine.layers.attention.factory import create_parallel_attention as fn

        assert callable(fn)

    def test_import_from_package(self):
        """Verify import from package __init__ works."""
        from diffsynth_engine.layers.attention import create_parallel_attention as fn

        assert callable(fn)

    @patch("diffsynth_engine.utils.platform.is_mindie_sd_available", return_value=False)
    @patch("diffsynth_engine.distributed.parallel_state.is_sp_group_initialized", return_value=False)
    def test_returns_usp_attention_when_no_mindie(self, mock_sp, mock_mindie):
        """On GPU/CPU (no MindIE), factory should return USPAttention."""
        attn = create_parallel_attention(
            num_heads=24,
            head_size=128,
            attn_type=None,
        )
        assert isinstance(attn, USPAttention)

    @patch("diffsynth_engine.utils.platform.is_mindie_sd_available", return_value=True)
    @patch("diffsynth_engine.distributed.parallel_state.is_sp_group_initialized", return_value=False)
    def test_returns_usp_attention_when_sp_not_initialized(self, mock_sp, mock_mindie):
        """Even if MindIE is available, without SP group we fall back to USPAttention."""
        attn = create_parallel_attention(
            num_heads=24,
            head_size=128,
            attn_type=None,
        )
        assert isinstance(attn, USPAttention)

    @patch("diffsynth_engine.utils.platform.is_mindie_sd_available", return_value=False)
    @patch("diffsynth_engine.distributed.parallel_state.is_sp_group_initialized", return_value=True)
    def test_returns_usp_attention_when_mindie_unavailable(self, mock_sp, mock_mindie):
        """If SP is initialized but MindIE is not available, return USPAttention."""
        attn = create_parallel_attention(
            num_heads=24,
            head_size=128,
            attn_type=None,
        )
        assert isinstance(attn, USPAttention)

    @patch("diffsynth_engine.utils.platform.is_mindie_sd_available", return_value=False)
    @patch("diffsynth_engine.distributed.parallel_state.is_sp_group_initialized", return_value=False)
    def test_parameters_passed_correctly(self, mock_sp, mock_mindie):
        """Verify that parameters are correctly forwarded to USPAttention."""
        attn = create_parallel_attention(
            num_heads=32,
            head_size=64,
            attn_type=None,
            num_kv_heads=8,
            scatter_idx=2,
            gather_idx=1,
        )
        assert isinstance(attn, USPAttention)
        assert attn.num_heads == 32
        assert attn.head_size == 64
        assert attn.num_kv_heads == 8
        assert attn.scatter_idx == 2
        assert attn.gather_idx == 1

    @patch("diffsynth_engine.utils.platform.is_mindie_sd_available", return_value=False)
    @patch("diffsynth_engine.distributed.parallel_state.is_sp_group_initialized", return_value=False)
    def test_default_parameters(self, mock_sp, mock_mindie):
        """Test factory with minimal required parameters."""
        attn = create_parallel_attention(
            num_heads=16,
            head_size=128,
        )
        assert isinstance(attn, USPAttention)
        assert attn.num_heads == 16
        assert attn.head_size == 128
        # num_kv_heads defaults to num_heads when None
        assert attn.num_kv_heads == 16
