import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from einops import rearrange

from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm, AdaLayerNormSingle, RoPEEmbedding, RMSNorm
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.constants import FLUX_DIT_CONFIG_FILE
from diffsynth_engine.utils import logging
from diffsynth_engine.models.basic.attention import attention

logger = logging.get_logger(__name__)

with open(FLUX_DIT_CONFIG_FILE, "r") as f:
    config = json.load(f)


class FluxDiTStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_rename_dict = config["diffusers"]["global_rename_dict"]
        rename_dict = config["diffusers"]["rename_dict"]
        rename_dict_single = config["diffusers"]["rename_dict_single"]
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.endswith(".weight") or name.endswith(".bias"):
                suffix = ".weight" if name.endswith(".weight") else ".bias"
                prefix = name[: -len(suffix)]
                if prefix in global_rename_dict:
                    state_dict_[global_rename_dict[prefix] + suffix] = param
                elif prefix.startswith("transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict:
                        name_ = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                elif prefix.startswith("single_transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "single_blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict_single:
                        name_ = ".".join(names[:2] + [rename_dict_single[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                    else:
                        pass
                else:
                    pass
        for name in list(state_dict_.keys()):
            if ".proj_in_besides_attn." in name:
                name_ = name.replace(".proj_in_besides_attn.", ".to_qkv_mlp.")
                param = torch.concat(
                    [
                        state_dict_[name.replace(".proj_in_besides_attn.", ".a_to_q.")],
                        state_dict_[name.replace(".proj_in_besides_attn.", ".a_to_k.")],
                        state_dict_[name.replace(".proj_in_besides_attn.", ".a_to_v.")],
                        state_dict_[name],
                    ],
                    dim=0,
                )
                state_dict_[name_] = param
                state_dict_.pop(name.replace(".proj_in_besides_attn.", ".a_to_q."))
                state_dict_.pop(name.replace(".proj_in_besides_attn.", ".a_to_k."))
                state_dict_.pop(name.replace(".proj_in_besides_attn.", ".a_to_v."))
                state_dict_.pop(name)
        for name in list(state_dict_.keys()):
            for component in ["a", "b"]:
                if f".{component}_to_q." in name:
                    name_ = name.replace(f".{component}_to_q.", f".{component}_to_qkv.")
                    param = torch.concat(
                        [
                            state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                            state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                            state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                        ],
                        dim=0,
                    )
                    state_dict_[name_] = param
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_q."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_k."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_v."))
        return state_dict_

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["civitai"]["rename_dict"]
        suffix_rename_dict = config["civitai"]["suffix_rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            name = name.replace("model.diffusion_model.", "")
            names = name.split(".")
            if name in rename_dict:
                if name.startswith("final_layer.adaLN_modulation.1."):
                    param = torch.concat([param[3072:], param[:3072]], dim=0)
                state_dict_[rename_dict[name]] = param
            elif names[0] == "double_blocks":
                name_ = f"blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                state_dict_[name_] = param
            elif names[0] == "single_blocks":
                if ".".join(names[2:]) in suffix_rename_dict:
                    name_ = f"single_blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                    state_dict_[name_] = param
            else:
                pass
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "txt_in.weight" in state_dict or "model.diffusion_model.txt_in.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        elif "time_text_embed.timestep_embedder.linear_1.weight" in state_dict:
            state_dict = self._from_diffusers(state_dict)
            logger.info("use diffusers format state dict")
        else:
            logger.info("user diffsynth format state dict")
        return state_dict


class FluxJointAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
        only_out_a=False,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = nn.Linear(dim_a, dim_a * 3, device=device, dtype=dtype)
        self.b_to_qkv = nn.Linear(dim_b, dim_b * 3, device=device, dtype=dtype)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_q_b = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k_b = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.a_to_out = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        if not only_out_a:
            self.b_to_out = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.attn_impl = attn_impl

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def forward(self, hidden_states_a, hidden_states_b, image_rotary_emb):
        batch_size = hidden_states_a.shape[0]

        # Part A
        qkv_a = self.a_to_qkv(hidden_states_a)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        # Part B
        qkv_b = self.b_to_qkv(hidden_states_b)
        qkv_b = qkv_b.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=1)
        q_b, k_b = self.norm_q_b(q_b), self.norm_k_b(k_b)

        q = torch.concat([q_b, q_a], dim=2)
        k = torch.concat([k_b, k_a], dim=2)
        v = torch.concat([v_b, v_a], dim=2)

        q, k = self.apply_rope(q, k, image_rotary_emb)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        hidden_states = attention(q, k, v, attn_impl=self.attn_impl)
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_b, hidden_states_a = (
            hidden_states[:, : hidden_states_b.shape[1]],
            hidden_states[:, hidden_states_b.shape[1] :],
        )
        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        else:
            hidden_states_b = self.b_to_out(hidden_states_b)
            return hidden_states_a, hidden_states_b


class FluxJointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim, device=device, dtype=dtype)
        self.norm1_b = AdaLayerNorm(dim, device=device, dtype=dtype)

        self.attn = FluxJointAttention(
            dim, dim, num_attention_heads, dim // num_attention_heads, attn_impl=attn_impl, device=device, dtype=dtype
        )

        self.norm2_a = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.ff_a = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim, device=device, dtype=dtype)
        )

        self.norm2_b = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.ff_b = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )

    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b, image_rotary_emb)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.dim = dim

        self.norm = AdaLayerNormSingle(dim, device=device, dtype=dtype)
        self.to_qkv_mlp = nn.Linear(dim, dim * (3 + 4), device=device, dtype=dtype)
        self.norm_q_a = RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k_a = RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.proj_out = nn.Linear(dim * 5, dim)
        self.attn_impl = attn_impl

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def process_attention(self, hidden_states, image_rotary_emb):
        batch_size = hidden_states.shape[0]

        qkv = hidden_states.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        q, k = self.norm_q_a(q), self.norm_k_a(k)

        q, k = self.apply_rope(q, k, image_rotary_emb)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        hidden_states = attention(q, k, v, attn_impl=self.attn_impl)
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states

    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb):
        residual = hidden_states_a
        norm_hidden_states, gate = self.norm(hidden_states_a, emb=temb)
        hidden_states_a = self.to_qkv_mlp(norm_hidden_states)
        attn_output, mlp_hidden_states = hidden_states_a[:, :, : self.dim * 3], hidden_states_a[:, :, self.dim * 3 :]

        attn_output = self.process_attention(attn_output, image_rotary_emb)
        mlp_hidden_states = nn.functional.gelu(mlp_hidden_states, approximate="tanh")

        hidden_states_a = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states_a = gate.unsqueeze(1) * self.proj_out(hidden_states_a)
        hidden_states_a = residual + hidden_states_a

        return hidden_states_a, hidden_states_b


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim, device: str, dtype: torch.dtype):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=True, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False, device=device, dtype=dtype)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None] + shift[:, None]
        return x


class FluxDiT(PreTrainedModel):
    converter = FluxDiTStateDictConverter()

    def __init__(
        self,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.time_embedder = TimestepEmbeddings(256, 3072, device=device, dtype=dtype)
        self.guidance_embedder = TimestepEmbeddings(256, 3072, device=device, dtype=dtype)
        self.pooled_text_embedder = nn.Sequential(
            nn.Linear(768, 3072, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(3072, 3072, device=device, dtype=dtype),
        )
        self.context_embedder = nn.Linear(4096, 3072, device=device, dtype=dtype)
        self.x_embedder = nn.Linear(64, 3072, device=device, dtype=dtype)

        self.blocks = nn.ModuleList(
            [FluxJointTransformerBlock(3072, 24, attn_impl=attn_impl, device=device, dtype=dtype) for _ in range(19)]
        )
        self.single_blocks = nn.ModuleList(
            [FluxSingleTransformerBlock(3072, 24, attn_impl=attn_impl, device=device, dtype=dtype) for _ in range(38)]
        )

        self.final_norm_out = AdaLayerNormContinuous(3072, device=device, dtype=dtype)
        self.final_proj_out = nn.Linear(3072, 64, device=device, dtype=dtype)

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(
            hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    def prepare_image_ids(self, latents):
        batch_size, _, height, width = latents.shape
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(device=latents.device, dtype=latents.dtype)

        return latent_image_ids

    def forward(
        self,
        hidden_states,
        timestep,
        prompt_emb,
        pooled_prompt_emb,
        guidance,
        text_ids,
        image_ids=None,
        use_gradient_checkpointing=False,
        controlnet_double_block_output=None,
        controlnet_single_block_output=None,
        **kwargs,
    ):
        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        with fp8_inference(fp8_linear_enabled), gguf_inference():
            if image_ids is None:
                image_ids = self.prepare_image_ids(hidden_states)

            # warning: keep the order of time_embedding + guidance_embedding + pooled_text_embedding
            # addition of floating point numbers does not meet commutative law
            conditioning = self.time_embedder(timestep, hidden_states.dtype)
            if self.guidance_embedder is not None:
                guidance = guidance * 1000
                conditioning += self.guidance_embedder(guidance, hidden_states.dtype)
            conditioning += self.pooled_text_embedder(pooled_prompt_emb)
            prompt_emb = self.context_embedder(prompt_emb)
            image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

            height, width = hidden_states.shape[-2:]
            hidden_states = self.patchify(hidden_states)
            hidden_states = self.x_embedder(hidden_states)

            for i, block in enumerate(self.blocks):
                if self.training and use_gradient_checkpointing:
                    hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden_states,
                        prompt_emb,
                        conditioning,
                        image_rotary_emb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
                if controlnet_double_block_output is not None:
                    interval_control = len(self.blocks) / len(controlnet_double_block_output)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_double_block_output[i // interval_control]

            hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
            for block in self.single_blocks:
                if self.training and use_gradient_checkpointing:
                    hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden_states,
                        prompt_emb,
                        conditioning,
                        image_rotary_emb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
                if controlnet_single_block_output is not None:
                    interval_control = len(self.single_blocks) / len(controlnet_double_block_output)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_single_block_output[i // interval_control]

            hidden_states = hidden_states[:, prompt_emb.shape[1] :]
            hidden_states = self.final_norm_out(hidden_states, conditioning)
            hidden_states = self.final_proj_out(hidden_states)
            hidden_states = self.unpatchify(hidden_states, height, width)
            return hidden_states

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        attn_impl: Optional[str] = None,
    ):
        with no_init_weights():
            model = torch.nn.utils.skip_init(
                cls,
                device=device,
                dtype=dtype,
                attn_impl=attn_impl,
            )
            model = model.requires_grad_(False)  # for loading gguf
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
