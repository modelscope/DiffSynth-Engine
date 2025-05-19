import torch
from typing import Dict

from diffsynth_engine.models.vae import VAEDecoder, VAEEncoder
from diffsynth_engine.models.utils import no_init_weights


class SDXLVAEEncoder(VAEEncoder):
    def __init__(self, attn_impl: str = "auto", device: str = "cuda:0", dtype: torch.dtype = torch.float32):
        super().__init__(
            latent_channels=4,
            scaling_factor=0.13025,
            shift_factor=0,
            use_quant_conv=True,
            attn_impl=attn_impl,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, attn_impl: str = "auto"
    ):
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype, attn_impl=attn_impl)
        model.load_state_dict(state_dict)
        return model


class SDXLVAEDecoder(VAEDecoder):
    def __init__(self, attn_impl: str = "auto", device: str = "cuda:0", dtype: torch.dtype = torch.float32):
        super().__init__(
            latent_channels=4,
            scaling_factor=0.13025,
            shift_factor=0,
            use_post_quant_conv=True,
            attn_impl=attn_impl,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, attn_impl: str = "auto"
    ):
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype, attn_impl=attn_impl)
        model.load_state_dict(state_dict)
        return model
