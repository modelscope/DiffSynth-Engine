import torch
from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.utils.import_utils import is_npu_available


class MindieAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not is_npu_available():
            raise RuntimeError("NPU is not available, cannot use MINDIE attention backend")

    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.MINDIE

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MindieAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type:
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


class MindieAttentionImpl(AttentionImpl):
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
        self.num_heads = num_heads
        self.head_size = head_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata=None,
    ) -> torch.Tensor:
        from mindiesd.layers.flash_attn.attention_forward import attention_forward

        scale = self.softmax_scale
        if scale is None:
            scale = self.head_size ** -0.5

        out = attention_forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            scale=scale,
            fused=True,
            head_first=False,
        )
        return out