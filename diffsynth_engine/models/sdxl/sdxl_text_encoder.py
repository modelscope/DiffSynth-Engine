import json
import torch
import torch.nn as nn
from typing import Dict

from diffsynth_engine.models.text_encoder.clip import CLIPEncoderLayer
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter, split_suffix
from diffsynth_engine.utils.constants import SDXL_TEXT_ENCODER_CONFIG_FILE
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

with open(SDXL_TEXT_ENCODER_CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


class SDXLTextEncoderStateDictConverter(StateDictConverter):
    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["diffusers"]["te1_rename_dict"]
        attn_rename_dict = config["diffusers"]["te1_attn_rename_dict"]
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                if layer_id == "11":
                    # we don't need the last layer
                    continue
                state_dict_[name_] = param
        return state_dict_

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["civitai"]["te1_rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            if not name.startswith("conditioner.embedders.0"):
                continue
            if name == "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight":
                param = param.reshape((1, param.shape[0], param.shape[1]))
                name = "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding"
                suffix = ""
            else:
                name, suffix = split_suffix(name)
            if name in rename_dict:
                name_ = rename_dict[name] + suffix
                state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        elif "text_model.final_layer_norm.weight" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return state_dict


class SDXLTextEncoder2StateDictConverter(StateDictConverter):
    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["diffusers"]["te2_rename_dict"]
        attn_rename_dict = config["diffusers"]["te2_attn_rename_dict"]
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        return state_dict_

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["civitai"]["te2_rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            if not name.startswith("conditioner.embedders.1"):
                continue
            if name.endswith(".in_proj_weight"):
                name = name.replace(".in_proj_weight", ".in_proj")
                suffix = ".weight"
            elif name.endswith(".in_proj_bias"):
                name = name.replace(".in_proj_bias", ".in_proj")
                suffix = ".bias"
            elif name == "conditioner.embedders.1.model.text_projection":
                name = "conditioner.embedders.1.model.text_projection"
                suffix = ".weight"
                param = param.T
            elif name == "conditioner.embedders.1.model.positional_embedding":
                name = "conditioner.embedders.1.model.positional_embedding"
                suffix = ""
                param = param.reshape((1, param.shape[0], param.shape[1]))
            else:
                name, suffix = split_suffix(name)

            if name in rename_dict:
                if isinstance(rename_dict[name], str):
                    name_ = rename_dict[name] + suffix
                    state_dict_[name_] = param
                else:
                    length = param.shape[0] // 3
                    for i, rename in enumerate(rename_dict[name]):
                        name_ = rename + suffix
                        state_dict_[name_] = param[i * length : i * length + length]
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        elif "text_model.final_layer_norm.weight" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return state_dict


class SDXLTextEncoder(PreTrainedModel):
    converter = SDXLTextEncoderStateDictConverter()

    def __init__(
        self,
        embed_dim=768,
        vocab_size=49408,
        max_position_embeddings=77,
        num_encoder_layers=11,
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

        # The text encoder is different to that in Stable Diffusion 1.x.
        # It does not include final_layer_norm.

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=2):
        clip_skip = max(
            clip_skip - 1, 1
        )  # Because we did not load the last layer of the encoder, the clip_skip needs to be decreased by 1.
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                break
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
        num_encoder_layers: int = 11,
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


class SDXLTextEncoder2(PreTrainedModel):
    converter = SDXLTextEncoder2StateDictConverter()

    def __init__(
        self,
        embed_dim=1280,
        vocab_size=49408,
        max_position_embeddings=77,
        num_encoder_layers=32,
        encoder_intermediate_size=5120,
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
                CLIPEncoderLayer(
                    embed_dim,
                    encoder_intermediate_size,
                    num_heads=20,
                    head_dim=64,
                    use_quick_gelu=False,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = nn.LayerNorm(embed_dim, device=device, dtype=dtype)

        # text_projection
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=2):
        clip_skip = max(clip_skip, 1)
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                hidden_states = embeds
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        pooled_embeds = self.text_projection(pooled_embeds)
        return hidden_states, pooled_embeds

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        embed_dim: int = 1280,
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        num_encoder_layers: int = 32,
        encoder_intermediate_size: int = 5120,
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
