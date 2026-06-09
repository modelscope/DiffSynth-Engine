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
    from sageattention import sageattn, sageattn_varlen

    SAGE_ATTN_2_AVAILABLE = True
except ImportError:
    SAGE_ATTN_2_AVAILABLE = False


class SageAttention2Backend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not SAGE_ATTN_2_AVAILABLE:
            error_msg = "SageAttention2 backend is not available. Please visit https://github.com/thu-ml/SageAttention and follow the installation instructions."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.SAGE2

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return SageAttention2Impl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]


class SageAttention2Impl(AttentionImpl):
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
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        query = rearrange(query, "b s n d -> b n s d")
        key = rearrange(key, "b s n d -> b n s d")
        value = rearrange(value, "b s n d -> b n s d")

        if cu_seqlens_q is not None:
            output = sageattn_varlen(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )
        else:
            output = sageattn(
                query,
                key,
                value,
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )
        output = rearrange(output, "b n s d -> b s n d")
        return output

    def forward_with_lse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = rearrange(query, "b s n d -> b n s d")
        key = rearrange(key, "b s n d -> b n s d")
        value = rearrange(value, "b s n d -> b n s d")

        if cu_seqlens_q is not None:
            raise NotImplementedError("sageattn_varlen can not return lse.")
        else:
            output, lse = sageattn(
                query,
                key,
                value,
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
                return_lse=True,
            )
        output = rearrange(output, "b n s d -> b s n d")
        return output, lse
