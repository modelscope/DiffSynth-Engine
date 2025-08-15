import json
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional
from einops import rearrange

from diffsynth_engine.models.basic.transformer_helper import (
    AdaLayerNormZero,
    AdaLayerNorm,
    RoPEEmbedding,
    RMSNorm,
)
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.constants import FLUX_DIT_CONFIG_FILE
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)
from diffsynth_engine.utils import logging


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
        dim = 3072
        for name, param in state_dict.items():
            if name.endswith(".weight") or name.endswith(".bias"):
                suffix = ".weight" if name.endswith(".weight") else ".bias"
                prefix = name[: -len(suffix)]
                if prefix in global_rename_dict:
                    # Fix load diffusers format weights [issue](https://github.com/modelscope/DiffSynth-Engine/issues/90).
                    if prefix.startswith("norm_out.linear"):
                        param = torch.concat([param[dim:], param[:dim]], dim=0)
                    state_dict_[global_rename_dict[prefix] + suffix] = param
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
                param = torch.concat(
                    [
                        state_dict_[name.replace(".proj_in_besides_attn.", ".a_to_q.")],
                        state_dict_[name.replace(".proj_in_besides_attn.", ".a_to_k.")],
                        state_dict_[name.replace(".proj_in_besides_attn.", ".a_to_v.")],
                    ],
                    dim=0,
                )
                state_dict_[name.replace(".proj_in_besides_attn.", ".attn.to_qkv.")] = param
                state_dict_[name.replace(".proj_in_besides_attn.", ".mlp.0.")] = state_dict_[name]
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
        dim = 3072
        rename_dict = config["civitai"]["rename_dict"]
        suffix_rename_dict = config["civitai"]["suffix_rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            name = name.replace("model.diffusion_model.", "")
            names = name.split(".")
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            elif names[0] == "double_blocks":
                name_ = f"blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
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
            elif names[0] == "single_blocks":
                if ".".join(names[2:]) in suffix_rename_dict:
                    name_ = f"single_blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                    if "to_qkv_mlp" in name_:
                        attn_param, mlp_param = param[: 3 * dim], param[3 * dim :]
                        state_dict_[name_.replace("to_qkv_mlp", "attn.to_qkv")] = attn_param
                        state_dict_[name_.replace("to_qkv_mlp", "mlp.0")] = mlp_param
                    else:
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
            logger.info("use diffsynth format state dict")
        return state_dict


def apply_rope(xq, xk, freqs_cis):
    xq_ = rearrange(xq, "b s h (d p q) -> b h s d p q", p=1, q=2)
    xk_ = rearrange(xk, "b s h (d p q) -> b h s d p q", p=1, q=2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    xq_out = rearrange(xq_out, "b h s d q -> b s h (d q)", q=2)
    xk_out = rearrange(xk_out, "b h s d q -> b s h (d q)", q=2)
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


class FluxDoubleAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.a_to_qkv = nn.Linear(dim_a, dim_a * 3, device=device, dtype=dtype)
        self.b_to_qkv = nn.Linear(dim_b, dim_b * 3, device=device, dtype=dtype)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_q_b = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k_b = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.a_to_out = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.b_to_out = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.attn_kwargs = attn_kwargs if attn_kwargs is not None else {}

    def attention_callback(self, attn_out_a, attn_out_b, x_a, x_b, q_a, q_b, k_a, k_b, v_a, v_b, rope_emb, image_emb):
        return attn_out_a, attn_out_b

    def forward(self, image, text, rope_emb, image_emb):
        q_a, k_a, v_a = rearrange(self.a_to_qkv(image), "b s (h d) -> b s h d", h=(3 * self.num_heads)).chunk(3, dim=2)
        q_b, k_b, v_b = rearrange(self.b_to_qkv(text), "b s (h d) -> b s h d", h=(3 * self.num_heads)).chunk(3, dim=2)
        q = torch.cat([self.norm_q_b(q_b), self.norm_q_a(q_a)], dim=1)
        k = torch.cat([self.norm_k_b(k_b), self.norm_k_a(k_a)], dim=1)
        v = torch.cat([v_b, v_a], dim=1)
        q, k = apply_rope(q, k, rope_emb)
        attn_out = attention_ops.attention(q, k, v, **self.attn_kwargs)
        attn_out = rearrange(attn_out, "b s h d -> b s (h d)").to(q.dtype)
        text_out, image_out = attn_out[:, : text.shape[1]], attn_out[:, text.shape[1] :]
        image_out, text_out = self.attention_callback(
            attn_out_a=image_out,
            attn_out_b=text_out,
            x_a=image,
            x_b=text,
            q_a=q_a,
            q_b=q_b,
            k_a=k_a,
            k_b=k_b,
            v_a=v_a,
            v_b=v_b,
            rope_emb=rope_emb,
            image_emb=image_emb,
        )
        return self.a_to_out(image_out), self.b_to_out(text_out)


class FluxDoubleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.attn = FluxDoubleAttention(
            dim, dim, num_heads, dim // num_heads, attn_kwargs=attn_kwargs, device=device, dtype=dtype
        )
        # Image
        self.norm_msa_a = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.norm_mlp_a = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.ff_a = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim, device=device, dtype=dtype)
        )
        # Text
        self.norm_msa_b = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.norm_mlp_b = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.ff_b = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )

    def forward(self, image, text, t_emb, rope_emb, image_emb=None):
        # AdaLayerNorm-Zero for Image and Text MSA
        image_in, gate_a = self.norm_msa_a(image, t_emb)
        text_in, gate_b = self.norm_msa_b(text, t_emb)
        image_out, text_out = self.attn(image_in, text_in, rope_emb, image_emb)
        image = image + gate_a * image_out
        text = text + gate_b * text_out

        # AdaLayerNorm-Zero for Image MLP
        image_in, gate_a = self.norm_mlp_a(image, t_emb)
        image = image + gate_a * self.ff_a(image_in)

        # AdaLayerNorm-Zero for Text MLP
        text_in, gate_b = self.norm_mlp_b(text, t_emb)
        text = text + gate_b * self.ff_b(text_in)
        return image, text


class FluxSingleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, device=device, dtype=dtype)
        self.norm_q_a = RMSNorm(dim // num_heads, eps=1e-6, device=device, dtype=dtype)
        self.norm_k_a = RMSNorm(dim // num_heads, eps=1e-6, device=device, dtype=dtype)
        self.attn_kwargs = attn_kwargs if attn_kwargs is not None else {}

    def attention_callback(self, attn_out, x, q, k, v, rope_emb, image_emb):
        return attn_out

    def forward(self, x, rope_emb, image_emb):
        q, k, v = rearrange(self.to_qkv(x), "b s (h d) -> b s h d", h=(3 * self.num_heads)).chunk(3, dim=2)
        q, k = apply_rope(self.norm_q_a(q), self.norm_k_a(k), rope_emb)
        attn_out = attention_ops.attention(q, k, v, **self.attn_kwargs)
        attn_out = rearrange(attn_out, "b s h d -> b s (h d)").to(q.dtype)
        return self.attention_callback(attn_out=attn_out, x=x, q=q, k=k, v=v, rope_emb=rope_emb, image_emb=image_emb)


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.norm = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.attn = FluxSingleAttention(dim, num_heads, attn_kwargs=attn_kwargs, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
        )
        self.proj_out = nn.Linear(dim * 5, dim)

    def forward(self, x, t_emb, rope_emb, image_emb=None):
        h, gate = self.norm(x, emb=t_emb)
        attn_output = self.attn(h, rope_emb, image_emb)
        mlp_output = self.mlp(h)
        return x + gate * self.proj_out(torch.cat([attn_output, mlp_output], dim=2))


class FluxDiT(PreTrainedModel):
    converter = FluxDiTStateDictConverter()
    _supports_parallelization = True

    def __init__(
        self,
        in_channel: int = 64,
        attn_kwargs: Optional[Dict[str, Any]] = None,
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
        # normal flux has 64 channels, bfl canny and depth has 128 channels, bfl fill has 384 channels, bfl redux has 64 channels
        self.x_embedder = nn.Linear(in_channel, 3072, device=device, dtype=dtype)

        self.blocks = nn.ModuleList(
            [
                FluxDoubleTransformerBlock(3072, 24, attn_kwargs=attn_kwargs, device=device, dtype=dtype)
                for _ in range(19)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(3072, 24, attn_kwargs=attn_kwargs, device=device, dtype=dtype)
                for _ in range(38)
            ]
        )
        self.final_norm_out = AdaLayerNorm(3072, device=device, dtype=dtype)
        self.final_proj_out = nn.Linear(3072, 64, device=device, dtype=dtype)

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(
            hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    @staticmethod
    def prepare_image_ids(latents: torch.Tensor):
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
        image_emb,
        guidance,
        text_ids,
        image_ids=None,
        controlnet_double_block_output=None,
        controlnet_single_block_output=None,
        **kwargs,
    ):
        h, w = hidden_states.shape[-2:]
        if image_ids is None:
            image_ids = self.prepare_image_ids(hidden_states)
        controlnet_double_block_output = (
            controlnet_double_block_output if controlnet_double_block_output is not None else ()
        )
        controlnet_single_block_output = (
            controlnet_single_block_output if controlnet_single_block_output is not None else ()
        )

        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        use_cfg = hidden_states.shape[0] > 1
        with (
            fp8_inference(fp8_linear_enabled),
            gguf_inference(),
            cfg_parallel(
                (
                    hidden_states,
                    timestep,
                    prompt_emb,
                    pooled_prompt_emb,
                    image_emb,
                    guidance,
                    text_ids,
                    image_ids,
                    *controlnet_double_block_output,
                    *controlnet_single_block_output,
                ),
                use_cfg=use_cfg,
            ),
        ):
            # warning: keep the order of time_embedding + guidance_embedding + pooled_text_embedding
            # addition of floating point numbers does not meet commutative law
            conditioning = self.time_embedder(timestep, hidden_states.dtype)
            if self.guidance_embedder is not None:
                guidance = (guidance.to(torch.float32) * 1000).to(hidden_states.dtype)
                conditioning += self.guidance_embedder(guidance, hidden_states.dtype)
            conditioning += self.pooled_text_embedder(pooled_prompt_emb)
            rope_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
            text_rope_emb = rope_emb[:, :, : text_ids.size(1)]
            image_rope_emb = rope_emb[:, :, text_ids.size(1) :]
            hidden_states = self.patchify(hidden_states)

            with sequence_parallel(
                (
                    hidden_states,
                    prompt_emb,
                    text_rope_emb,
                    image_rope_emb,
                    *controlnet_double_block_output,
                    *controlnet_single_block_output,
                ),
                seq_dims=(
                    1,
                    1,
                    2,
                    2,
                    *(1 for _ in controlnet_double_block_output),
                    *(1 for _ in controlnet_single_block_output),
                ),
            ):
                hidden_states = self.x_embedder(hidden_states)
                prompt_emb = self.context_embedder(prompt_emb)
                rope_emb = torch.cat((text_rope_emb, image_rope_emb), dim=2)

                for i, block in enumerate(self.blocks):
                    hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, rope_emb, image_emb)
                    if len(controlnet_double_block_output) > 0:
                        interval_control = len(self.blocks) / len(controlnet_double_block_output)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states = hidden_states + controlnet_double_block_output[i // interval_control]
                hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
                for i, block in enumerate(self.single_blocks):
                    hidden_states = block(hidden_states, conditioning, rope_emb, image_emb)
                    if len(controlnet_single_block_output) > 0:
                        interval_control = len(self.single_blocks) / len(controlnet_double_block_output)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states = hidden_states + controlnet_single_block_output[i // interval_control]

                hidden_states = hidden_states[:, prompt_emb.shape[1] :]
                hidden_states = self.final_norm_out(hidden_states, conditioning)
                hidden_states = self.final_proj_out(hidden_states)
                (hidden_states,) = sequence_parallel_unshard((hidden_states,), seq_dims=(1,), seq_lens=(h * w // 4,))

            hidden_states = self.unpatchify(hidden_states, h, w)
            (hidden_states,) = cfg_parallel_unshard((hidden_states,), use_cfg=use_cfg)
            return hidden_states

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        in_channel: int = 64,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        model = cls(
            device="meta",
            dtype=dtype,
            in_channel=in_channel,
            attn_kwargs=attn_kwargs,
        )
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def compile_repeated_blocks(self, *args, **kwargs):
        for block in self.blocks:
            block.compile(*args, **kwargs)

    def get_fsdp_modules(self):
        return ["blocks", "single_blocks"]
