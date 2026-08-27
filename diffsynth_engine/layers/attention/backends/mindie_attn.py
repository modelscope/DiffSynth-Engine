import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class MindieAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def build(self, **kwargs) -> AttentionMetadata:
        return AttentionMetadata()


class MindieAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        from diffsynth_engine.platforms import AscendPlatform

        if not AscendPlatform.supports("device"):
            error_msg = "MindIE attention requires an available Ascend NPU device."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        if not AscendPlatform.supports("mindie_attention"):
            error_msg = (
                "MindIE attention backend is not available. "
                "Install MindIE-SD 3.x matching the current torch_npu and CANN versions, "
                "and ensure mindiesd.layers.flash_attn.attention_forward works on NPU."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.MINDIE)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MindieAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return MindieAttentionMetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []

    @classmethod
    def supports_ring_attention(cls) -> bool:
        return False


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
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from mindiesd.layers.flash_attn.attention_forward import attention_forward

        # MindIE attention_forward 没有 causal 参数，只接受 attn_mask（布尔张量，
        # True 表示保留/参与计算，False 表示被屏蔽）。因此当请求 causal 且调用方未显式
        # 传入 attn_mask 时，手动构造下三角 causal mask 并通过 attn_mask 传入。
        # layout 为 "BSND"，故 q_seqlen = query.shape[1]、kv_seqlen = key.shape[1]。
        if self.causal and attn_mask is None:
            q_seqlen = query.shape[1]
            kv_seqlen = key.shape[1]
            attn_mask = torch.tril(
                torch.ones(q_seqlen, kv_seqlen, dtype=torch.bool, device=query.device)
            )

        return attention_forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            scale=self.softmax_scale,
            fused=True,
            head_first=False,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BSND",
        )
