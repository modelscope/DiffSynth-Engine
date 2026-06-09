import torch
from einops import rearrange

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

try:
    from sageattn3 import sageattn3_blackwell

    SAGE_ATTN_3_AVAILABLE = True
except ImportError:
    SAGE_ATTN_3_AVAILABLE = False


class SageAttention3Backend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not SAGE_ATTN_3_AVAILABLE:
            error_msg = "SageAttention3 backend is not available. Please visit https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell and follow the installation instructions."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.SAGE3)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return SageAttention3Impl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]


class SageAttention3Impl(AttentionImpl):
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
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        query = rearrange(query, "b s n d -> b n s d")
        key = rearrange(key, "b s n d -> b n s d")
        value = rearrange(value, "b s n d -> b n s d")

        output = sageattn3_blackwell(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=self.causal,
        )
        output = rearrange(output, "b n s d -> b s n d")
        return output
