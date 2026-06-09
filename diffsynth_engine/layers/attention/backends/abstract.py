# Adapted from https://github.com/vllm-project/vllm

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class AttentionType(str, enum.Enum):
    SDPA = "sdpa"
    FA2 = "fa2"
    FA3 = "fa3"
    FA3_FP8 = "fa3_fp8"
    FA4 = "fa4"
    AITER = "aiter"
    AITER_FP8 = "aiter_fp8"
    SAGE2 = "sage2"
    SAGE3 = "sage3"
    SPARGE = "sparge"

    def __str__(self) -> str:
        return self.value


class AttentionBackend(ABC):
    """Abstract class for diffusion attention backends."""

    @staticmethod
    @abstractmethod
    def check_availability() -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_type() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_supported_head_sizes() -> list[int]:
        """Get the list of supported head sizes for this backend."""
        raise NotImplementedError

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        if (not supported_head_sizes) or head_size in supported_head_sizes:
            return True

        logger.error(
            f"Attention backend {cls.get_type()!r} does not support head size {head_size}. "
            f"Supported head sizes: {supported_head_sizes}"
        )
        return False

    @classmethod
    def supports_ring_attention(cls) -> bool:
        return False


@dataclass
class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build(self, **kwargs) -> AttentionMetadata:
        raise NotImplementedError


class AttentionImpl(ABC, Generic[T]):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
