import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.utils.platform import is_npu_available


class MindieAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not is_npu_available():
            raise RuntimeError("NPU is not available, cannot use MINDIE attention backend")

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.MINDIE)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MindieAttentionImpl

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
        self.scale = softmax_scale or (head_size**-0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from mindiesd.layers.flash_attn.attention_forward import attention_forward

        return attention_forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            scale=self.scale,
            fused=True,
            head_first=False,
        )
