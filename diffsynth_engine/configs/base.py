from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch

from diffsynth_engine.layers.attention import AttentionType
from diffsynth_engine.registry import get_attn_backend
from diffsynth_engine.utils import logging
from diffsynth_engine.platforms import get_device_type, resolve_platform

logger = logging.get_logger(__name__)


@dataclass
class AttentionParams:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpargeAttentionParams(AttentionParams):
    topk: float = 0.5


@dataclass
class PipelineConfig:
    model_path: str
    model_dtype: torch.dtype = torch.bfloat16
    text_encoder_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.float32
    device: str | torch.device = "auto"

    pipeline_class_name: str | None = None

    # vae
    vae_tiled: bool = False
    vae_tile_size: int | Tuple[int, int] = (256, 256)
    vae_tile_stride: int | Tuple[int, int] = (192, 192)

    # attention
    attn_type: AttentionType | str = AttentionType.SDPA
    attn_params: Optional[AttentionParams] = None

    # optimization
    use_torch_compile: bool = False

    # parallelism
    parallelism: int = 1
    use_cfg_parallel: bool = False
    sp_ulysses_degree: Optional[int] = None
    sp_ring_degree: Optional[int] = None
    tp_degree: Optional[int] = None
    use_vae_parallel: bool = False
    use_fsdp: bool = False

    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any]) -> "PipelineConfig":
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in args_dict.items() if k in field_names}
        return cls(**filtered_dict)

    def __post_init__(self):
        self.attn_type = str(self.attn_type)
        init_parallel_config(self)
        validate_attn_config(self)
        init_device_config(self)


def init_parallel_config(config: PipelineConfig):
    if config.parallelism <= 0:
        raise ValueError(f"parallelism must be a positive integer, got {config.parallelism}")

    cfg_degree = 2 if config.use_cfg_parallel else 1  # TODO: support cfg_degree > 2

    if config.tp_degree is not None and config.tp_degree <= 0:
        raise ValueError(f"tp_degree must be None or a positive integer, got {config.tp_degree}")
    if config.sp_ulysses_degree is not None and config.sp_ulysses_degree <= 0:
        raise ValueError(f"sp_ulysses_degree must be None or a positive integer, got {config.sp_ulysses_degree}")
    if config.sp_ring_degree is not None and config.sp_ring_degree <= 0:
        raise ValueError(f"sp_ring_degree must be None or a positive integer, got {config.sp_ring_degree}")

    config.tp_degree = config.tp_degree or 1
    config.sp_ring_degree = config.sp_ring_degree or 1
    config.sp_ulysses_degree = config.sp_ulysses_degree or (
        config.parallelism // (cfg_degree * config.tp_degree * config.sp_ring_degree)
    )

    parallel_degree = cfg_degree * config.tp_degree * config.sp_ulysses_degree * config.sp_ring_degree
    if parallel_degree != config.parallelism:
        raise ValueError(
            f"parallelism ({config.parallelism}) must equal cfg_degree({cfg_degree}) * "
            f"tp_degree({config.tp_degree}) * sp_ulysses_degree({config.sp_ulysses_degree}) * "
            f"sp_ring_degree({config.sp_ring_degree}) = {parallel_degree}"
        )

    if config.tp_degree > 1 and config.use_fsdp:
        raise ValueError("TP and FSDP cannot be enabled together; set tp_degree=None or use_fsdp=False .")

    if config.use_torch_compile and config.use_fsdp:
        logger.warning("torch.compile + FSDP may produce graph breaks")

    if config.use_vae_parallel:
        assert config.parallelism > 1, "use_vae_parallel requires parallelism > 1"
        if not config.vae_tiled:
            config.vae_tiled = True
            logger.warning("setting vae_tiled to True since use_vae_parallel is enabled")


def validate_attn_config(config: PipelineConfig):
    attn_backend = get_attn_backend(config.attn_type)
    if config.sp_ring_degree is not None and config.sp_ring_degree > 1:
        if not attn_backend.supports_ring_attention():
            raise ValueError(f"Attention backend {config.attn_type!r} does not support ring attention.")


def init_device_config(config: PipelineConfig):
    if config.device is None or (isinstance(config.device, str) and config.device.lower() in ("auto", "")):
        config.device = get_device_type()
        return
    # Validate that the explicit device type is registered (fail fast at construction).
    resolve_platform(config.device)
