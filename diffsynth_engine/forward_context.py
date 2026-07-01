# Adapted from https://github.com/vllm-project/vllm

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diffsynth_engine.layers.attention import AttentionMetadata


@dataclass
class ForwardContext:
    attn_metadata: Optional[AttentionMetadata] = None
    attn_type: Optional[str] = None


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


@contextmanager
def override_forward_context(forward_context: Optional[ForwardContext] = None):
    """A context manager that overrides the current forward context."""
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    attn_metadata: Optional[AttentionMetadata] = None,
    attn_type: Optional[str] = None,
):
    """A context manager to that stores the current forward context."""
    forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        attn_type=attn_type,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        # TODO: perf
        pass
