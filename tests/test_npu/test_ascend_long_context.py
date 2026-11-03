"""Tests for use_ascend_long_ctx_attn without NPU hardware."""

from unittest.mock import patch

import pytest

from diffsynth_engine.layers.attention import AttentionType, ascend_long_context, use_ascend_long_ctx_attn


@pytest.mark.parametrize(
    "attn_type,mindie_available,sp_initialized,ring_degree,expected",
    [
        ("mindie", True, True, 1, True),
        (AttentionType.MINDIE, True, True, 1, True),
        ("sdpa", True, True, 1, False),
        (None, True, True, 1, False),
        ("mindie", False, True, 1, False),
        ("mindie", True, False, 1, False),
        ("mindie", True, True, 2, False),
    ],
)
def test_use_ascend_long_ctx_attn(attn_type, mindie_available, sp_initialized, ring_degree, expected):
    with (
        patch.object(ascend_long_context, "is_mindie_sd_available", return_value=mindie_available),
        patch.object(ascend_long_context, "is_sp_group_initialized", return_value=sp_initialized),
        patch.object(ascend_long_context, "get_ring_parallel_world_size", return_value=ring_degree) as get_ring,
    ):
        assert use_ascend_long_ctx_attn(attn_type) is expected
        if not sp_initialized:
            get_ring.assert_not_called()
