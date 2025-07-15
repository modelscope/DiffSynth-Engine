import os
import torch
from dataclasses import dataclass
from typing import Tuple, Optional

from diffsynth_engine.config.controlnet import ControlType


@dataclass
class DiffsynthConfig:
    # Model configs
    model_path: str | os.PathLike  # UNet or DiT model path
    clip_l_path: Optional[str | os.PathLike] = None  # CLIP-L model path for SD1.5, SDXL or FLUX
    clip_g_path: Optional[str | os.PathLike] = None  # CLIP-G model path for SDXL
    t5_path: Optional[str | os.PathLike] = None  # T5 model path for FLUX or Wan
    vae_path: Optional[str | os.PathLike] = None  # VAE model path for SD1.5, SDXL, FLUX or Wan
    image_encoder_path: Optional[str | os.PathLike] = None  # Image encoder model path for Wan
    model_dtype: torch.dtype = torch.bfloat16  # UNet or DiT model dtype
    clip_l_dtype: torch.dtype = torch.bfloat16  # CLIP-L model dtype
    clip_g_dtype: torch.dtype = torch.bfloat16  # CLIP-G model dtype
    t5_dtype: torch.dtype = torch.bfloat16  # T5 model dtype
    vae_dtype: torch.dtype = torch.bfloat16  # VAE model dtype
    image_encoder_dtype: torch.dtype = torch.bfloat16  # Image encoder model dtype
    device: str = "cuda"
    # FLUX configs
    load_text_encoder: bool = True  # Whether to load CLIP/T5 text encoder for FLUX
    control_type: ControlType = ControlType.normal  # Control type for FLUX

    # Sampling configs
    shift: Optional[float] = None  # Scheduler shift factor

    # Runtime configs
    batch_cfg: bool = False
    vae_tiled: Optional[bool] = None
    vae_tile_size: Optional[int | Tuple[int, int]] = None
    vae_tile_stride: Optional[int | Tuple[int, int]] = None

    # Attention configs
    dit_attn_impl: str = "auto"  # Attention implementation for FLUX or Wan DiT
    # Sparge Attention configs
    sparge_smooth_k: bool = True
    sparge_cdfthreshd: float = 0.6
    sparge_simthreshd1: float = 0.98
    sparge_pvthreshd: float = 50.0

    # Optimazation configs
    offload_mode: Optional[str] = None  # Offload model params to CPU
    use_fp8_linear: bool = False  # Use FP8 inference in linear layers, available for FLUX
    # FBCache configs
    use_fbcache: bool = False  # Use FBCache accleration, available for FLUX and Wan
    fbcache_relative_l1_threshold: float = 0.05

    # Parallel configs
    parallelism: int = 1  # Number of parallel devices, available for FLUX and Wan
    use_cfg_parallel: bool = False
    cfg_degree: Optional[int] = None
    sp_ulysses_degree: Optional[int] = None
    sp_ring_degree: Optional[int] = None
    tp_degree: Optional[int] = None
    use_fsdp: bool = False
