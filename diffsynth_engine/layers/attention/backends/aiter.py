import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.platform import DTYPE_FP8

logger = logging.get_logger(__name__)

try:
    from aiter import flash_attn_fp8_pertensor_func as aiter_flash_attn_fp8
    from aiter import flash_attn_func as aiter_flash_attn

    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False


class AiterBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        if not AITER_AVAILABLE:
            error_msg = "Aiter backend is not available. Please visit https://github.com/ROCm/aiter and follow the installation instructions."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.AITER

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return AiterImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]


class AiterImpl(AttentionImpl):
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
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        output = aiter_flash_attn(
            query,
            key,
            value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return output


class AiterFP8Backend(AiterBackend):
    @staticmethod
    def get_type() -> AttentionType:
        return AttentionType.AITER_FP8

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return AiterFP8Impl


class AiterFP8Impl(AiterImpl):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # TODO: scaling
        original_dtype = query.dtype
        query = query.to(DTYPE_FP8)
        key = key.to(DTYPE_FP8)
        value = value.to(DTYPE_FP8)
        output = aiter_flash_attn_fp8(
            query,
            key,
            value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        output = output.to(original_dtype)
        return output
