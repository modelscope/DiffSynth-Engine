import torch
from typing import Dict

from diffsynth_engine.models.vae import VAEDecoder, VAEEncoder


class SD3VAEEncoder(VAEEncoder):
    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.float32):
        super().__init__(
            latent_channels=16,
            scaling_factor=1.5305,
            shift_factor=0.0609,
            use_quant_conv=False,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        model = cls(device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model


class SD3VAEDecoder(VAEDecoder):
    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.float32):
        super().__init__(
            latent_channels=16,
            scaling_factor=1.5305,
            shift_factor=0.0609,
            use_post_quant_conv=False,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        model = cls(device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
