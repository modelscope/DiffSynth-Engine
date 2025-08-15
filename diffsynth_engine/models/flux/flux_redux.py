from torch.nn.modules.module import Module
import torch
import torch.nn as nn
from typing import Dict
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.models.base import PreTrainedModel
from diffsynth_engine.models.text_encoder.siglip import SiglipImageEncoder


class FluxReduxImageEmbedder(nn.Module):
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3, device=device, dtype=dtype)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        return self.redux_down(nn.functional.silu(self.redux_up(x)))

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ):
        model = cls(device="meta", dtype=dtype)
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model


class FluxRedux(PreTrainedModel):
    def __init__(
        self, image_encoder: SiglipImageEncoder, image_embedder: FluxReduxImageEmbedder, scale: float = 1.0
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.image_embedder = image_embedder
        self.scale = scale

    def set_scale(self, scale):
        self.scale = scale

    def forward(self, images):
        image_embeds = self.image_encoder(images)
        image_embeds = self.image_embedder(image_embeds)
        return self.scale * image_embeds

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16, **kwargs
    ):
        model_path = fetch_model("muse/google-siglip-so400m-patch14-384", path="model.safetensors")
        image_encoder = SiglipImageEncoder.from_pretrained(model_path, use_head=False, device=device, dtype=dtype)
        image_embedder: Module = FluxReduxImageEmbedder.from_state_dict(state_dict, device=device, dtype=dtype)
        return cls(image_encoder, image_embedder)
