import torch
import torch.nn.functional as F
from einops import rearrange

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)

_scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention


class SDPABackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        pass

    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.SDPA

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return SDPAImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


class SDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_kv_groups = num_heads // num_kv_heads
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

        enable_gqa = self.num_kv_groups > 1 and attn_mask is None
        if self.num_kv_groups > 1 and not enable_gqa:
            key = torch.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = torch.repeat_interleave(value, self.num_kv_groups, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=self.causal,
            scale=self.softmax_scale,
            enable_gqa=enable_gqa,
        )
        output = rearrange(output, "b n s d -> b s n d")
        return output

    def forward_with_lse(
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

        if self.num_kv_groups > 1:
            key = torch.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = torch.repeat_interleave(value, self.num_kv_groups, dim=1)

        seq_len = query.shape[2]
        output, lse = _scaled_dot_product_efficient_attention(
            query,
            key,
            value,
            attn_bias=attn_mask,
            compute_log_sumexp=True,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )[:2]

        output = rearrange(output, "b n s d -> b s n d")
        # the returned lse is padded but not restored, so we need to slice it
        lse = lse[:, :, :seq_len]
        return output, lse
