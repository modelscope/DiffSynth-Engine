import torch
from typing import Dict

from diffsynth_engine.models.vae import VAEDecoder, VAEEncoder
from diffsynth_engine.models.utils import no_init_weights


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
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype)
        model.load_state_dict(state_dict)
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
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype)
        model.load_state_dict(state_dict)
        return model
