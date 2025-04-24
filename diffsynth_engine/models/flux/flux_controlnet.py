import torch
import torch.nn as nn
from typing import Optional, Dict

from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.flux.flux_dit import (
    FluxJointTransformerBlock,
    FluxSingleTransformerBlock,
    RoPEEmbedding,
    TimestepEmbeddings,
)


class FluxControlNetStateDictConverter(StateDictConverter):
    def __init__(self):
        super().__init__()

    def _from_alimama_flux_inpainting(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 阿里妈妈

        return state_dict

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._from_alimama_flux_inpainting(state_dict)


class FluxControlNet(PreTrainedModel):
    def __init__(self, attn_impl: Optional[str] = None, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
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
        self.controlnet_x_embedder = nn.Linear(64 + 4, 3072)
        self.blocks = nn.ModuleList(
            [FluxJointTransformerBlock(3072, 24, attn_impl=attn_impl, device=device, dtype=dtype) for _ in range(19)]
        )
        self.single_blocks = nn.ModuleList(
            [FluxSingleTransformerBlock(3072, 24, attn_impl=attn_impl, device=device, dtype=dtype) for _ in range(38)]
        )
        # controlnet projection
        self.blocks_proj = nn.ModuleList(
            [nn.Linear(3072, 3072, device=device, dtype=dtype) for _ in range(len(self.blocks))]
        )
        self.single_blocks_proj = nn.ModuleList(
            [nn.Linear(3072, 3072, device=device, dtype=dtype) for _ in range(len(self.single_blocks))]
        )

    def get_patch_callback(self):
        def patch_callback(hidden_states, controlnet_outputs, index, patch_point:FluxPatchPoint):            
            
            pass

        return patch_callback
        

    def forward(
        self,
        hidden_states,
        control_condition,
        control_scale,
        timestep,
        prompt_emb,
        pooled_prompt_emb,
        guidance,
        image_ids,
        text_ids
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
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, condition, image_rotary_emb)
            double_block_outputs.append(self.blocks_proj[i](hidden_states))

        # single block
        single_block_outputs = []
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        for i, block in enumerate(self.single_blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, condition, image_rotary_emb)
            single_block_outputs.append(self.single_blocks_proj[i](hidden_states[:, prompt_emb.shape[1] :]))

        # apply control scale
        double_block_outputs = [control_scale * output for output in double_block_outputs]
        single_block_outputs = [control_scale * output for output in single_block_outputs]

        return double_block_outputs, single_block_outputs
