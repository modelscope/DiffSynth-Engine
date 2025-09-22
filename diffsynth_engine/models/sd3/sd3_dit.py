import json
import torch
import torch.nn as nn
from typing import Dict, Optional
from einops import rearrange

from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm, AdaLayerNormZero
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.utils.constants import SD3_DIT_CONFIG_FILE
from diffsynth_engine.models.basic.attention import attention
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

with open(SD3_DIT_CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


class SD3DiTStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dim = 1536
        rename_dict = config["diffusers"]["rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                if name == "pos_embed.pos_embed":
                    param = param.reshape((1, 192, 192, 1536))
                state_dict_[rename_dict[name]] = param
            elif name.endswith(".weight") or name.endswith(".bias"):
                suffix = ".weight" if name.endswith(".weight") else ".bias"
                prefix = name[: -len(suffix)]
                if prefix in rename_dict:
                    state_dict_[rename_dict[prefix] + suffix] = param
                elif prefix.startswith("transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict:
                        name_: str = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                        if "linear_a" in name_:
                            attn_param, mlp_param = param[: 3 * dim], param[3 * dim :]
                            state_dict_[name_.replace("linear_a", "norm_msa_a.linear")] = attn_param
                            state_dict_[name_.replace("linear_a", "norm_mlp_a.linear")] = mlp_param
                        elif "linear_b" in name_:
                            attn_param, mlp_param = param[: 3 * dim], param[3 * dim :]
                            state_dict_[name_.replace("linear_b", "norm_msa_b.linear")] = attn_param
                            state_dict_[name_.replace("linear_b", "norm_mlp_b.linear")] = mlp_param
                        else:
                            state_dict_[name_] = param
        return state_dict_

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dim = 1536
        rename_dict = config["civitai"]["rename_dict"]
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name.startswith("model.diffusion_model.joint_blocks.23.context_block.adaLN_modulation.1."):
                    param = torch.concat([param[1536:], param[:1536]], axis=0)
                elif name.startswith("model.diffusion_model.final_layer.adaLN_modulation.1."):
                    param = torch.concat([param[1536:], param[:1536]], axis=0)
                elif name == "model.diffusion_model.pos_embed":
                    param = param.reshape((1, 192, 192, 1536))
                if isinstance(rename_dict[name], str):
                    state_dict_[rename_dict[name]] = param
                else:
                    name_ = rename_dict[name][0].replace(".a_to_q.", ".a_to_qkv.").replace(".b_to_q.", ".b_to_qkv.")
                    if "linear_a" in name_:
                        attn_param, mlp_param = param[: 3 * dim], param[3 * dim :]
                        state_dict_[name_.replace("linear_a", "norm_msa_a.linear")] = attn_param
                        state_dict_[name_.replace("linear_a", "norm_mlp_a.linear")] = mlp_param
                    elif "linear_b" in name_:
                        attn_param, mlp_param = param[: 3 * dim], param[3 * dim :]
                        state_dict_[name_.replace("linear_b", "norm_msa_b.linear")] = attn_param
                        state_dict_[name_.replace("linear_b", "norm_mlp_b.linear")] = mlp_param
                    else:
                        state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "model.diffusion_model.context_embedder.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        elif "time_text_embed.timestep_embedder.linear_1" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return state_dict


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=16,
        embed_dim=1536,
        pos_embed_max_size=192,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, device=device, dtype=dtype
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.pos_embed_max_size, self.pos_embed_max_size, 1536, device=device, dtype=dtype)
        )

    def cropped_pos_embed(self, height, width):
        height = height // self.patch_size
        width = width // self.patch_size
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed[:, top : top + height, left : left + width, :].flatten(1, 2)
        return spatial_pos_embed

    def forward(self, latent):
        height, width = latent.shape[-2:]
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)
        pos_embed = self.cropped_pos_embed(height, width)
        return latent + pos_embed


class JointAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.a_to_qkv = nn.Linear(dim_a, dim_a * 3, device=device, dtype=dtype)
        self.b_to_qkv = nn.Linear(dim_b, dim_b * 3, device=device, dtype=dtype)

        self.a_to_out = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.b_to_out = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)

        self.attn_impl = attn_impl

    def forward(self, image, text):
        qkv = torch.concat([self.a_to_qkv(image), self.b_to_qkv(text)], dim=1)
        q, k, v = rearrange(qkv, "b s (h d) -> b s h d").chunk(3, dim=2)
        attn_out = attention(q, k, v, self.attn_impl)
        attn_out = rearrange(attn_out, "b s h d -> b s (h d)")
        image, text = attn_out[:, : image.shape[1]], attn_out[:, image.shape[1] :]
        image = self.a_to_out(image)
        text = self.b_to_out(text)
        return image, text


class JointTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.attn = JointAttention(
            dim, dim, num_attention_heads, dim // num_attention_heads, device=device, dtype=dtype
        )
        # Image
        self.norm_msa_a = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.norm_mlp_a = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.ff_a = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )
        # Text
        self.norm_msa_b = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.norm_mlp_b = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.ff_b = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )

    def forward(self, image, text, t_emb):
        # AdaLayerNorm-Zero for Image and Text MSA
        image_in, gate_a = self.norm_msa_a(image, t_emb)
        text_in, gate_b = self.norm_msa_b(text, t_emb)
        image_out, text_out = self.attn(image_in, text_in)
        image = image + gate_a * image_out
        text = text + gate_b * text_out

        # AdaLayerNorm-Zero for Image MLP
        image_in, gate_a = self.norm_mlp_a(image, t_emb)
        image = image + gate_a * self.ff_a(image_in)

        # AdaLayerNorm-Zero for Text MLP
        text_in, gate_b = self.norm_mlp_b(text, t_emb)
        text = text + gate_b * self.ff_b(text_in)
        return image, text


class SD3DiT(PreTrainedModel):
    converter = SD3DiTStateDictConverter()

    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.pos_embedder = PatchEmbed(
            patch_size=2, in_channels=16, embed_dim=1536, pos_embed_max_size=192, device=device, dtype=dtype
        )
        self.time_embedder = TimestepEmbeddings(256, 1536, device=device, dtype=dtype)
        self.pooled_text_embedder = nn.Sequential(
            nn.Linear(2048, 1536, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(1536, 1536, device=device, dtype=dtype),
        )
        self.context_embedder = nn.Linear(4096, 1536, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([JointTransformerBlock(1536, 24, device=device, dtype=dtype) for _ in range(24)])
        self.norm_out = AdaLayerNorm(1536, device=device, dtype=dtype)
        self.proj_out = nn.Linear(1536, 64, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states,
        timestep,
        prompt_emb,
        pooled_prompt_emb,
    ):
        t_emb = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
        prompt_emb = self.context_embedder(prompt_emb)

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embedder(hidden_states)

        for block in self.blocks:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, t_emb)

        hidden_states = self.norm_out(hidden_states, t_emb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(
            hidden_states, "B (H W) (P Q C) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        model = cls(device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
