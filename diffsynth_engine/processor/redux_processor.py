from torch import nn
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel
from utils.download import fetch_model
from utils.loader import load_safetensors


class ReduxImageEncoder(nn.Module):
    siglip_model_name = "google/siglip-so400m-patch14-384"

    def __init__(
        self,
        device,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        flux_redux_path: str = None,
        google_siglip_path: str = None,
        dtype=torch.bfloat16,
    ) -> None:
        super().__init__()

        self.redux_dim = redux_dim
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype

        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

        if flux_redux_path is None:
            flux_redux_path = fetch_model("muse/flux1-redux-dev", revision="v1", path="flux1-redux-dev.safetensors")
        state_dict = load_safetensors(flux_redux_path)
        self.load_state_dict(state_dict, strict=False, assign=True)
        self.redux_up.to(device=device)
        self.redux_down.to(device=device)

        if google_siglip_path is None:
            google_siglip_path = fetch_model("muse/siglip-so400m-patch14-384", revision="v1", path="model.safetensors")
        self.siglip = SiglipVisionModel.from_pretrained(google_siglip_path).to(dtype=dtype, device=device)
        self.normalize = SiglipImageProcessor.from_pretrained(google_siglip_path)

    def __call__(self, x: Image.Image) -> torch.Tensor:
        imgs = self.normalize.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)
        _encoded_x = self.siglip(**imgs.to(device=self.device, dtype=self.dtype)).last_hidden_state
        projected_x = self.redux_down(nn.functional.silu(self.redux_up(_encoded_x)))

        return projected_x
