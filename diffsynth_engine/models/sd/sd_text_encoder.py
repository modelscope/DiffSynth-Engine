import json
import torch
import torch.nn as nn
from typing import Dict

from diffsynth_engine.models.text_encoder.clip import CLIPEncoderLayer
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.utils.constants import SD_TEXT_ENCODER_CONFIG_FILE
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

with open(SD_TEXT_ENCODER_CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


class SDTextEncoderStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

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

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["civitai"]["rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            if name not in rename_dict:
                continue
            name_ = rename_dict[name]
            if name == "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight":
                param = param.reshape((1, param.shape[0], param.shape[1]))
            state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        elif "text_model.encoder.layers.0.layer_norm1.weight" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return state_dict


class SDTextEncoder(PreTrainedModel):
    converter = SDTextEncoderStateDictConverter()

    def __init__(
        self,
        embed_dim=768,
        vocab_size=49408,
        max_position_embeddings=77,
        num_encoder_layers=12,
        encoder_intermediate_size=3072,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # token_embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = nn.Parameter(
            torch.zeros(1, max_position_embeddings, embed_dim, device=device, dtype=dtype)
        )

        # encoders
        self.encoders = nn.ModuleList(
            [
                CLIPEncoderLayer(embed_dim, encoder_intermediate_size, device=device, dtype=dtype)
                for _ in range(num_encoder_layers)
            ]
        )

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = nn.LayerNorm(embed_dim, device=device, dtype=dtype)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=1):
        clip_skip = max(clip_skip, 1)
        embeds = self.token_embedding(input_ids)
        embeds += self.position_embeds.to(device=embeds.device, dtype=embeds.dtype)
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)

        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                break
        embeds = self.final_layer_norm(embeds)
        return embeds

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        embed_dim: int = 768,
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        num_encoder_layers: int = 12,
        encoder_intermediate_size: int = 3072,
    ):
        model = cls(
            device="meta",
            dtype=dtype,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_encoder_layers=num_encoder_layers,
            encoder_intermediate_size=encoder_intermediate_size,
        )
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
