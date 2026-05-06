from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch

from diffsynth_engine.layers.attention import AttentionType
from diffsynth_engine.utils import logging

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
    device: str | torch.device = "cuda"

    pipeline_class_name: str | None = None

    # vae
    vae_tiled: bool = False
    vae_tile_size: int | Tuple[int, int] = (256, 256)
    vae_tile_stride: int | Tuple[int, int] = (192, 192)

    # attention
    attn_type: AttentionType | None = None  # None = auto-detect
    attn_params: Optional[AttentionParams] = None

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
        init_parallel_config(self)


def init_parallel_config(config: PipelineConfig):
    assert config.parallelism in (1, 2, 4, 8), "parallelism must be 1, 2, 4 or 8"

    cfg_degree = 2 if config.use_cfg_parallel else 1

    if config.tp_degree is not None:
        assert config.sp_ulysses_degree is None and config.sp_ring_degree is None, (
            "not allowed to enable sequence parallel and tensor parallel together; "
            "either set sp_ulysses_degree=None, sp_ring_degree=None or set tp_degree=None during pipeline initialization"
        )
        assert config.use_fsdp is False, (
            "not allowed to enable fully sharded data parallel and tensor parallel together; "
            "either set use_fsdp=False or set tp_degree=None during pipeline initialization"
        )
        config.sp_ulysses_degree = 1
        config.sp_ring_degree = 1
    elif config.sp_ulysses_degree is None and config.sp_ring_degree is None:
        # use ulysses if not specified
        config.sp_ulysses_degree = config.parallelism // cfg_degree
        config.sp_ring_degree = 1
        config.tp_degree = 1
    elif config.sp_ulysses_degree is not None and config.sp_ring_degree is not None:
        config.tp_degree = 1
    else:
        raise ValueError("sp_ulysses_degree and sp_ring_degree must be specified together")

    assert config.parallelism == cfg_degree * config.tp_degree * config.sp_ulysses_degree * config.sp_ring_degree, (
        f"parallelism ({config.parallelism}) must be equal to cfg_degree ({cfg_degree}) * "
        f"tp_degree ({config.tp_degree}) * "
        f"sp_ulysses_degree ({config.sp_ulysses_degree}) * "
        f"sp_ring_degree ({config.sp_ring_degree})"
    )

    if config.use_vae_parallel:
        assert config.parallelism > 1, "use_vae_parallel requires parallelism > 1"
        if not config.vae_tiled:
            config.vae_tiled = True
            logger.warning("setting vae_tiled to True since use_vae_parallel is enabled")
