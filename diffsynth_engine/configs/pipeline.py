import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from diffsynth_engine.configs.controlnet import ControlType


@dataclass
class BaseConfig:
    model_path: str | os.PathLike | List[str | os.PathLike]
    model_dtype: torch.dtype
    batch_cfg: bool = False
    vae_tiled: bool = False
    vae_tile_size: int | Tuple[int, int] = 256
    vae_tile_stride: int | Tuple[int, int] = 256
    device: str = "cuda"
    offload_mode: Optional[str] = None


@dataclass
class AttentionConfig:
    dit_attn_impl: str = "auto"
    # Sparge Attention
    sparge_smooth_k: bool = True
    sparge_cdfthreshd: float = 0.6
    sparge_simthreshd1: float = 0.98
    sparge_pvthreshd: float = 50.0


@dataclass
class OptimizationConfig:
    use_fp8_linear: bool = False
    use_fbcache: bool = False
    fbcache_relative_l1_threshold: float = 0.05
    use_torch_compile: bool = False


@dataclass
class ParallelConfig:
    parallelism: int = 1
    use_cfg_parallel: bool = False
    cfg_degree: Optional[int] = None
    sp_ulysses_degree: Optional[int] = None
    sp_ring_degree: Optional[int] = None
    tp_degree: Optional[int] = None
    use_fsdp: bool = False


@dataclass
class SDPipelineConfig(BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    clip_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    model_dtype: torch.dtype = torch.float16
    clip_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        device: str = "cuda",
        offload_mode: Optional[str] = None,
    ) -> "SDPipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            offload_mode=offload_mode,
        )


@dataclass
class SDXLPipelineConfig(BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    clip_l_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    clip_g_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    model_dtype: torch.dtype = torch.float16
    clip_l_dtype: torch.dtype = torch.float16
    clip_g_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        device: str = "cuda",
        offload_mode: Optional[str] = None,
    ) -> "SDXLPipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            offload_mode=offload_mode,
        )


@dataclass
class FluxPipelineConfig(AttentionConfig, OptimizationConfig, ParallelConfig, BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    clip_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    t5_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    model_dtype: torch.dtype = torch.bfloat16
    clip_dtype: torch.dtype = torch.bfloat16
    t5_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16

    load_text_encoder: bool = True
    control_type: ControlType = ControlType.normal

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        device: str = "cuda",
        parallelism: int = 1,
        offload_mode: Optional[str] = None,
    ) -> "FluxPipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            parallelism=parallelism,
            use_fsdp=True,
            offload_mode=offload_mode,
        )

    def __post_init__(self):
        init_parallel_config(self)


@dataclass
class WanPipelineConfig(AttentionConfig, OptimizationConfig, ParallelConfig, BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    t5_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    image_encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    model_dtype: torch.dtype = torch.bfloat16
    t5_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16
    image_encoder_dtype: torch.dtype = torch.bfloat16

    # default params set by model type
    boundary: Optional[float] = field(default=None, init=False)  # boundary
    shift: Optional[float] = field(default=None, init=False)  # RecifitedFlowScheduler shift factor
    cfg_scale: Optional[float | Tuple[float, float]] = field(default=None, init=False)  # default CFG scale
    num_inference_steps: Optional[int] = field(default=None, init=False)  # default inference steps
    fps: Optional[int] = field(default=None, init=False)  # default FPS

    # override BaseConfig
    vae_tiled: bool = True
    vae_tile_size: Tuple[int, int] = (34, 34)
    vae_tile_stride: Tuple[int, int] = (18, 16)

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        image_encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        device: str = "cuda",
        parallelism: int = 1,
        offload_mode: Optional[str] = None,
    ) -> "WanPipelineConfig":
        return cls(
            model_path=model_path,
            image_encoder_path=image_encoder_path,
            device=device,
            parallelism=parallelism,
            use_cfg_parallel=True,
            use_fsdp=True,
            offload_mode=offload_mode,
        )

    def __post_init__(self):
        init_parallel_config(self)


@dataclass
class QwenImagePipelineConfig(AttentionConfig, OptimizationConfig, ParallelConfig, BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    model_dtype: torch.dtype = torch.bfloat16
    encoder_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.float32

    # override OptimizationConfig
    fbcache_relative_l1_threshold = 0.009

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        device: str = "cuda",
        parallelism: int = 1,
        offload_mode: Optional[str] = None,
    ) -> "QwenImagePipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            encoder_path=encoder_path,
            vae_path=vae_path,
            parallelism=parallelism,
            use_cfg_parallel=True,
            use_fsdp=True,
            offload_mode=offload_mode,
        )

    def __post_init__(self):
        init_parallel_config(self)


@dataclass
class BaseStateDicts:
    model: Optional[Dict[str, torch.Tensor]] = None
    vae: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class SDStateDicts(BaseStateDicts):
    clip: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class SDXLStateDicts(BaseStateDicts):
    clip_l: Optional[Dict[str, torch.Tensor]] = None
    clip_g: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class FluxStateDicts(BaseStateDicts):
    t5: Optional[Dict[str, torch.Tensor]] = None
    clip: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class QwenImageStateDicts(BaseStateDicts):
    encoder: Optional[Dict[str, torch.Tensor]] = None


def init_parallel_config(config: FluxPipelineConfig | QwenImagePipelineConfig | WanPipelineConfig):
    assert config.parallelism in (1, 2, 4, 8), "parallelism must be 1, 2, 4 or 8"
    config.batch_cfg = True if config.parallelism > 1 and config.use_cfg_parallel else config.batch_cfg

    if config.use_cfg_parallel is True and config.cfg_degree is not None:
        raise ValueError("use_cfg_parallel and cfg_degree should not be specified together")
    config.cfg_degree = (2 if config.use_cfg_parallel else 1) if config.cfg_degree is None else config.cfg_degree

    if config.tp_degree is not None:
        assert config.sp_ulysses_degree is None and config.sp_ring_degree is None, (
            "not allowed to enable sequence parallel and tensor parallel together; "
            "either set sp_ulysses_degree=None, sp_ring_degree=None or set tp_degree=None during pipeline initialization"
        )
        assert config.use_fsdp is False, (
            "not allowed to enable fully sharded data parallel and tensor parallel together; "
            "either set use_fsdp=False or set tp_degree=None during pipeline initialization"
        )
        assert config.parallelism == config.cfg_degree * config.tp_degree, (
            f"parallelism ({config.parallelism}) must be equal to cfg_degree ({config.cfg_degree}) * tp_degree ({config.tp_degree})"
        )
        config.sp_ulysses_degree = 1
        config.sp_ring_degree = 1
    elif config.sp_ulysses_degree is None and config.sp_ring_degree is None:
        # use ulysses if not specified
        config.sp_ulysses_degree = config.parallelism // config.cfg_degree
        config.sp_ring_degree = 1
        config.tp_degree = 1
    elif config.sp_ulysses_degree is not None and config.sp_ring_degree is not None:
        assert config.parallelism == config.cfg_degree * config.sp_ulysses_degree * config.sp_ring_degree, (
            f"parallelism ({config.parallelism}) must be equal to cfg_degree ({config.cfg_degree}) * "
            f"sp_ulysses_degree ({config.sp_ulysses_degree}) * sp_ring_degree ({config.sp_ring_degree})"
        )
        config.tp_degree = 1
    else:
        raise ValueError("sp_ulysses_degree and sp_ring_degree must be specified together")
