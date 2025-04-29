import json
import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange

from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils.constants import SD3_DIT_CONFIG_FILE
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

with open(SD3_DIT_CONFIG_FILE, "r") as f:
    config = json.load(f)


class SD3DiTStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
                        name_ = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
        return state_dict_

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        only_out_a=False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = nn.Linear(dim_a, dim_a * 3, device=device, dtype=dtype)
        self.b_to_qkv = nn.Linear(dim_b, dim_b * 3, device=device, dtype=dtype)

        self.a_to_out = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        if not only_out_a:
            self.b_to_out = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)

    def forward(self, hidden_states_a, hidden_states_b):
        batch_size = hidden_states_a.shape[0]

        qkv = torch.concat([self.a_to_qkv(hidden_states_a), self.b_to_qkv(hidden_states_b)], dim=1)
        qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)

        hidden_states = nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_a, hidden_states_b = (
            hidden_states[:, : hidden_states_a.shape[1]],
            hidden_states[:, hidden_states_a.shape[1] :],
        )
        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        else:
            hidden_states_b = self.b_to_out(hidden_states_b)
            return hidden_states_a, hidden_states_b


class JointTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim, device=device, dtype=dtype)
        self.norm1_b = AdaLayerNorm(dim, device=device, dtype=dtype)

        self.attn = JointAttention(
            dim, dim, num_attention_heads, dim // num_attention_heads, device=device, dtype=dtype
        )

        self.norm2_a = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.ff_a = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )

        self.norm2_b = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.ff_b = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )

    def forward(self, hidden_states_a, hidden_states_b, temb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b


class JointTransformerFinalBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim, device=device, dtype=dtype)
        self.norm1_b = AdaLayerNorm(dim, single=True, device=device, dtype=dtype)

        self.attn = JointAttention(
            dim, dim, num_attention_heads, dim // num_attention_heads, only_out_a=True, device=device, dtype=dtype
        )

        self.norm2_a = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.ff_a = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )

    def forward(self, hidden_states_a, hidden_states_b, temb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a = self.attn(norm_hidden_states_a, norm_hidden_states_b)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        return hidden_states_a, hidden_states_b


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
        self.blocks = nn.ModuleList(
            [JointTransformerBlock(1536, 24, device=device, dtype=dtype) for _ in range(23)]
            + [JointTransformerFinalBlock(1536, 24, device=device, dtype=dtype)]
        )
        self.norm_out = AdaLayerNorm(1536, single=True, device=device, dtype=dtype)
        self.proj_out = nn.Linear(1536, 64, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states,
        timestep,
        prompt_emb,
        pooled_prompt_emb,
        use_gradient_checkpointing=False,
    ):
        conditioning = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
        prompt_emb = self.context_embedder(prompt_emb)

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embedder(hidden_states)

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    prompt_emb,
                    conditioning,
                    use_reentrant=False,
                )
            else:
                hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning)

        hidden_states = self.norm_out(hidden_states, conditioning)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(
            hidden_states, "B (H W) (P Q C) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
