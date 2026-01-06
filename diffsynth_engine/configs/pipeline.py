import os
import torch
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

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
    offload_to_disk: bool = False


class AttnImpl(Enum):
    AUTO = "auto"
    EAGER = "eager"  # Native Attention
    FA2 = "fa2"  # Flash Attention 2
    FA3 = "fa3"  # Flash Attention 3
    FA3_FP8 = "fa3_fp8"  # Flash Attention 3 with FP8
    FA4 = "fa4"  # Flash Attention 4
    AITER = "aiter"  # Aiter Flash Attention
    AITER_FP8 = "aiter_fp8"  # Aiter Flash Attention with FP8
    XFORMERS = "xformers"  # XFormers
    SDPA = "sdpa"  # Scaled Dot Product Attention
    SAGE = "sage"  # Sage Attention
    SPARGE = "sparge"  # Sparge Attention
    VSA = "vsa"  # Video Sparse Attention


@dataclass
class SpargeAttentionParams:
    smooth_k: bool = True
    cdfthreshd: float = 0.6
    simthreshd1: float = 0.98
    pvthreshd: float = 50.0


@dataclass
class VideoSparseAttentionParams:
    sparsity: float = 0.9


@dataclass
class AttentionConfig:
    dit_attn_impl: AttnImpl = AttnImpl.AUTO
    attn_params: Optional[SpargeAttentionParams | VideoSparseAttentionParams] = None


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
        offload_to_disk: bool = False,
    ) -> "SDPipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            offload_mode=offload_mode,
            offload_to_disk=offload_to_disk,
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
        offload_to_disk: bool = False,
    ) -> "SDXLPipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            offload_mode=offload_mode,
            offload_to_disk=offload_to_disk,
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
        offload_to_disk: bool = False,
    ) -> "FluxPipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            parallelism=parallelism,
            use_fsdp=True if parallelism > 1 else False,
            offload_mode=offload_mode,
            offload_to_disk=offload_to_disk,
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
        offload_to_disk: bool = False,
    ) -> "WanPipelineConfig":
        return cls(
            model_path=model_path,
            image_encoder_path=image_encoder_path,
            device=device,
            parallelism=parallelism,
            use_cfg_parallel=True if parallelism > 1 else False,
            use_fsdp=True if parallelism > 1 else False,
            offload_mode=offload_mode,
            offload_to_disk=offload_to_disk,
        )

    def __post_init__(self):
        init_parallel_config(self)


@dataclass
class WanSpeech2VideoPipelineConfig(WanPipelineConfig):
    audio_encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    audio_encoder_dtype: torch.dtype = torch.float32

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        audio_encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        device: str = "cuda",
        parallelism: int = 1,
        offload_mode: Optional[str] = None,
    ) -> "WanSpeech2VideoPipelineConfig":
        return cls(
            model_path=model_path,
            audio_encoder_path=audio_encoder_path,
            device=device,
            parallelism=parallelism,
            use_cfg_parallel=True if parallelism > 1 else False,
            use_fsdp=True if parallelism > 1 else False,
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

    load_encoder: bool = True

    # override OptimizationConfig
    fbcache_relative_l1_threshold = 0.009

    # svd
    use_nunchaku: Optional[bool] = field(default=None, init=False)
    use_nunchaku_awq: Optional[bool] = field(default=None, init=False)
    use_nunchaku_attn: Optional[bool] = field(default=None, init=False)

    # for 2511
    use_zero_cond_t: bool = False

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        device: str = "cuda",
        parallelism: int = 1,
        offload_mode: Optional[str] = None,
        offload_to_disk: bool = False,
        use_zero_cond_t: bool = False,
    ) -> "QwenImagePipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            encoder_path=encoder_path,
            vae_path=vae_path,
            parallelism=parallelism,
            use_cfg_parallel=True if parallelism > 1 else False,
            use_fsdp=True if parallelism > 1 else False,
            offload_mode=offload_mode,
            offload_to_disk=offload_to_disk,
            use_zero_cond_t=use_zero_cond_t,
        )

    def __post_init__(self):
        init_parallel_config(self)


@dataclass
class HunyuanPipelineConfig(BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    model_dtype: torch.dtype = torch.float16
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_dtype: torch.dtype = torch.float16
    image_encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    image_encoder_dtype: torch.dtype = torch.float16


@dataclass
class ZImagePipelineConfig(AttentionConfig, OptimizationConfig, ParallelConfig, BaseConfig):
    model_path: str | os.PathLike | List[str | os.PathLike]
    model_dtype: torch.dtype = torch.bfloat16
    vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    vae_dtype: torch.dtype = torch.bfloat16
    encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None
    encoder_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def basic_config(
        cls,
        model_path: str | os.PathLike | List[str | os.PathLike],
        encoder_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        vae_path: Optional[str | os.PathLike | List[str | os.PathLike]] = None,
        device: str = "cuda",
        parallelism: int = 1,
        offload_mode: Optional[str] = None,
        offload_to_disk: bool = False,
    ) -> "ZImagePipelineConfig":
        return cls(
            model_path=model_path,
            device=device,
            encoder_path=encoder_path,
            vae_path=vae_path,
            parallelism=parallelism,
            use_cfg_parallel=True if parallelism > 1 else False,
            use_fsdp=True if parallelism > 1 else False,
            offload_mode=offload_mode,
            offload_to_disk=offload_to_disk,
        )

    def __post_init__(self):
        init_parallel_config(self)


@dataclass
class BaseStateDicts:
    pass


@dataclass
class SDStateDicts:
    model: Dict[str, torch.Tensor]
    clip: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]


@dataclass
class SDXLStateDicts:
    model: Dict[str, torch.Tensor]
    clip_l: Dict[str, torch.Tensor]
    clip_g: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]


@dataclass
class FluxStateDicts:
    model: Dict[str, torch.Tensor]
    t5: Dict[str, torch.Tensor]
    clip: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]


@dataclass
class WanStateDicts:
    model: Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]
    t5: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]
    image_encoder: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class WanS2VStateDicts:
    model: Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]
    t5: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]
    audio_encoder: Dict[str, torch.Tensor]


@dataclass
class QwenImageStateDicts:
    model: Dict[str, torch.Tensor]
    encoder: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]


@dataclass
class ZImageStateDicts:
    model: Dict[str, torch.Tensor]
    encoder: Dict[str, torch.Tensor]
    vae: Dict[str, torch.Tensor]


def init_parallel_config(config: FluxPipelineConfig | QwenImagePipelineConfig | WanPipelineConfig | ZImagePipelineConfig):
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


@dataclass
class LoraConfig:
    scale: float
    scheduler_config: Optional[Dict] = None
