from typing import Tuple

import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

try:
    from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

    FLASH_ATTN_4_AVAILABLE = True
except ImportError:
    FLASH_ATTN_4_AVAILABLE = False


class FlashAttention4Backend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not FLASH_ATTN_4_AVAILABLE:
            error_msg = "FlashAttention4 backend is not available. Please visit https://github.com/Dao-AILab/flash-attention and follow the installation instructions."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.FA4

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return FlashAttention4Impl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]


class FlashAttention4Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        window_size: Tuple[int, int] | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if cu_seqlens_q is not None:
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
                window_size=window_size if window_size is not None else (-1, -1),
            )
        else:
            output = flash_attn_func(
                query,
                key,
                value,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
                window_size=window_size if window_size is not None else (-1, -1),
            )
        return output
