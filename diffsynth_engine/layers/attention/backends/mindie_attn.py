import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.platform import is_npu_available

logger = logging.get_logger(__name__)

try:
    from mindiesd.layers.flash_attn.attention_forward import attention_forward

    MINDIESD_ATTN_AVAILABLE = is_npu_available() 
except ImportError:
    MINDIESD_ATTN_AVAILABLE = False

class MindieAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not MINDIESD_ATTN_AVAILABLE:
            error_msg = "MindiesdAttention backend is not available. Please visit https://gitcode.com/Ascend/MindIE-SD/blob/dev/docs/zh/features/core_layers.md and follow the installation instructions."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
        return attention_forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            scale=self.scale,
            fused=True,
            head_first=False,
        )
