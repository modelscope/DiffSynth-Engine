import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from einops import rearrange
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.flux.flux_dit import (
    FluxDoubleTransformerBlock,
    RoPEEmbedding,
    TimestepEmbeddings,
)


class FluxControlNetStateDictConverter(StateDictConverter):
    def __init__(self):
        super().__init__()

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dim = 3072
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if "attn.to_q" in new_key:
                q = state_dict[new_key]
                k = state_dict[new_key.replace("attn.to_q", "attn.to_k")]
                v = state_dict[new_key.replace("attn.to_q", "attn.to_v")]
                new_key = new_key.replace("transformer_blocks", "blocks")
                new_key = new_key.replace("attn.to_q", "attn.a_to_qkv")
                new_state_dict[new_key] = torch.cat((q, k, v), dim=0)
            elif "attn.add_q_proj" in new_key:
                q = state_dict[new_key]
                k = state_dict[new_key.replace("attn.add_q_proj", "attn.add_k_proj")]
                v = state_dict[new_key.replace("attn.add_q_proj", "attn.add_v_proj")]
                new_key = new_key.replace("transformer_blocks", "blocks")
                new_key = new_key.replace("attn.add_q_proj", "attn.b_to_qkv")
                new_state_dict[new_key.replace("attn.add_q_proj", "attn.b_to_qkv")] = torch.cat((q, k, v), dim=0)
            elif (
                "attn.to_k" in new_key
                or "attn.to_v" in new_key
                or "attn.add_k_proj" in new_key
                or "attn.add_v_proj" in new_key
            ):
                continue
            else:
                new_key = new_key.replace("transformer_blocks", "blocks")
                new_key = new_key.replace("controlnet_blocks", "blocks_proj")
                new_key = new_key.replace("time_text_embed.guidance_embedder", "guidance_embedder")
                new_key = new_key.replace("time_text_embed.timestep_embedder", "time_embedder")
                new_key = new_key.replace("time_text_embed.text_embedder.linear_1", "pooled_text_embedder.0")
                new_key = new_key.replace("time_text_embed.text_embedder.linear_2", "pooled_text_embedder.2")
                new_key = new_key.replace("transformer_blocks", "blocks")
                new_key = new_key.replace("time_embedder.linear_1", "time_embedder.timestep_embedder.0")
                new_key = new_key.replace("time_embedder.linear_2", "time_embedder.timestep_embedder.2")
                new_key = new_key.replace("guidance_embedder.linear_1", "guidance_embedder.timestep_embedder.0")
                new_key = new_key.replace("guidance_embedder.linear_2", "guidance_embedder.timestep_embedder.2")
                # joint block
                new_key = new_key.replace("norm1.linear", "norm1_a.linear")
                new_key = new_key.replace("norm1_context.linear", "norm1_b.linear")
                new_key = new_key.replace("attn.to_out.0", "attn.a_to_out")
                new_key = new_key.replace("attn.to_add_out", "attn.b_to_out")
                new_key = new_key.replace("attn.norm_q", "attn.norm_q_a")
                new_key = new_key.replace("attn.norm_k", "attn.norm_k_a")
                new_key = new_key.replace("attn.norm_added_q", "attn.norm_q_b")
                new_key = new_key.replace("attn.norm_added_k", "attn.norm_k_b")
                new_key = new_key.replace("ff.net", "ff_a")
                new_key = new_key.replace("ff_context.net", "ff_b")
                new_key = new_key.replace("0.proj", "0")
                if "norm1_a" in new_key:
                    attn_param, mlp_param = value[: 3 * dim], value[3 * dim :]
                    new_state_dict[new_key.replace("norm1_a", "norm_msa_a")] = attn_param
                    new_state_dict[new_key.replace("norm1_a", "norm_mlp_a")] = mlp_param
                elif "norm1_b" in new_key:
                    attn_param, mlp_param = value[: 3 * dim], value[3 * dim :]
                    new_state_dict[new_key.replace("norm1_b", "norm_msa_b")] = attn_param
                    new_state_dict[new_key.replace("norm1_b", "norm_mlp_b")] = mlp_param
                else:
                    new_state_dict[new_key] = value
        return new_state_dict

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._from_diffusers(state_dict)


class FluxControlNet(PreTrainedModel):
    converter = FluxControlNetStateDictConverter()

    def __init__(
        self,
        condition_channels: int = 64,
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
        self.controlnet_x_embedder = nn.Linear(condition_channels, 3072)
        self.blocks = nn.ModuleList(
            [FluxDoubleTransformerBlock(3072, 24, device=device, dtype=dtype) for _ in range(6)]
        )
        # controlnet projection
        self.blocks_proj = nn.ModuleList(
            [nn.Linear(3072, 3072, device=device, dtype=dtype) for _ in range(len(self.blocks))]
        )

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        control_condition: torch.Tensor,
        control_scale: float,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        pooled_prompt_emb: torch.Tensor,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: torch.Tensor,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        hidden_states = self.x_embedder(hidden_states) + self.controlnet_x_embedder(control_condition)
        condition = (
            self.time_embedder(timestep, hidden_states.dtype)
            + self.guidance_embedder(guidance * 1000, hidden_states.dtype)
            + self.pooled_text_embedder(pooled_prompt_emb)
        )
        prompt_emb = self.context_embedder(prompt_emb)
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

        # double block
        double_block_outputs = []
        for i, block in enumerate(self.blocks):
            hidden_states, prompt_emb = block(
                hidden_states, prompt_emb, condition, image_rotary_emb, attn_kwargs=attn_kwargs
            )
            double_block_outputs.append(self.blocks_proj[i](hidden_states))

        # apply control scale
        double_block_outputs = [control_scale * output for output in double_block_outputs]
        return double_block_outputs, None

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        if "controlnet_x_embedder.weight" in state_dict:
            condition_channels = state_dict["controlnet_x_embedder.weight"].shape[1]
        else:
            condition_channels = 64

        model = cls(condition_channels=condition_channels, device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
