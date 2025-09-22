import json
import torch
from typing import Dict

from diffsynth_engine.models.sd import SDTextEncoder
from diffsynth_engine.models.text_encoder.t5 import T5EncoderModel
from diffsynth_engine.models.base import StateDictConverter
from diffsynth_engine.utils.constants import FLUX_TEXT_ENCODER_CONFIG_FILE
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

with open(FLUX_TEXT_ENCODER_CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


class FluxTextEncoder1StateDictConverter(StateDictConverter):
    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["diffusers"]["rename_dict"]
        attn_rename_dict = config["diffusers"]["attn_rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "text_model.final_layer_norm.weight" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return state_dict


class FluxTextEncoder1(SDTextEncoder):
    converter = FluxTextEncoder1StateDictConverter()

    def __init__(self, vocab_size=49408, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__(vocab_size=vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids, clip_skip=2):
        embeds = self.token_embedding(input_ids)
        embeds += self.position_embeds.to(device=embeds.device, dtype=embeds.dtype)
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder in self.encoders:
            embeds = encoder(embeds, attn_mask=attn_mask)
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        return embeds, pooled_embeds

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, vocab_size: int = 49408
    ):
        model = cls(device="meta", dtype=dtype, vocab_size=vocab_size)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model


class FluxTextEncoder2(T5EncoderModel):
    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__(
            embed_dim=4096,
            vocab_size=32128,
            num_encoder_layers=24,
            d_ff=10240,
            num_heads=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.0,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids):
        return super().forward(input_ids=input_ids)
