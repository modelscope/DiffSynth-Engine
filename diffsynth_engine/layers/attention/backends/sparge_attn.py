from dataclasses import dataclass

import torch
from einops import rearrange

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

try:
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

    SPARGE_ATTN_AVAILABLE = True
except ImportError:
    SPARGE_ATTN_AVAILABLE = False


@dataclass
class SpargeAttentionMetadata(AttentionMetadata):
    topk: float = 0.5


class SpargeAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def build(self, topk: float = 0.5, **kwargs) -> SpargeAttentionMetadata:
        return SpargeAttentionMetadata(topk=topk)


class SpargeAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not SPARGE_ATTN_AVAILABLE:
            error_msg = "SpargeAttention backend is not available. Please visit https://github.com/thu-ml/SpargeAttn and follow the installation instructions."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.SPARGE

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return SpargeAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return SpargeAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return SpargeAttentionMetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]


class SpargeAttentionImpl(AttentionImpl):
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
        attn_metadata: SpargeAttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        query = rearrange(query, "b s n d -> b n s d")
        key = rearrange(key, "b s n d -> b n s d")
        value = rearrange(value, "b s n d -> b n s d")

        topk = attn_metadata.topk if attn_metadata is not None else 0.5

        output = spas_sage2_attn_meansim_topk_cuda(
            query,
            key,
            value,
            attn_mask=attn_mask,
            topk=topk,
            is_causal=self.causal,
        )
        output = rearrange(output, "b n s d -> b s n d")
        return output
