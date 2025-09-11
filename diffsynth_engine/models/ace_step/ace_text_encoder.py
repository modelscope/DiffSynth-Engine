from typing import Dict, Any
import json

import torch

from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.utils.constants import ACE_TEXT_ENCODER_CONFIG_FILE


class ACETextEncoder(WanTextEncoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ) -> "ACETextEncoder":
        model = cls(**config, device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    @staticmethod
    def get_model_config() -> dict:
        config_file = ACE_TEXT_ENCODER_CONFIG_FILE
        with open(config_file, "r") as f:
            config = json.load(f)
        return config