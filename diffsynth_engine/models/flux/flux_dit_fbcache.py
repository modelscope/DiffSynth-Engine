import torch
import numpy as np
from typing import Any, Dict, List, Optional

from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)
from diffsynth_engine.utils import logging
from diffsynth_engine.models.flux.flux_dit import FluxDiT

logger = logging.get_logger(__name__)


class FluxDiTFBCache(FluxDiT):
    def __init__(
        self,
        in_channel: int = 64,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        relative_l1_threshold: float = 0.05,
    ):
        super().__init__(in_channel=in_channel, device=device, dtype=dtype)
        self.relative_l1_threshold = relative_l1_threshold
        self.step_count = 0
        self.num_inference_steps = 0

    def is_relative_l1_below_threshold(self, prev_residual, residual, threshold):
        if threshold <= 0.0:
            return False

        if prev_residual.shape != residual.shape:
            return False

        mean_diff = (prev_residual - residual).abs().mean()
        mean_prev_residual = prev_residual.abs().mean()
        diff = mean_diff / mean_prev_residual
        return diff.item() < threshold

    def refresh_cache_status(self, num_inference_steps):
        self.step_count = 0
        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        pooled_prompt_emb: torch.Tensor,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: torch.Tensor,
        image_emb: torch.Tensor | None = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_double_block_output: List[torch.Tensor] | None = None,
        controlnet_single_block_output: List[torch.Tensor] | None = None,
        **kwargs,
    ):
        image_seq_len = hidden_states.shape[1]
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
                    image_ids,
                    text_ids,
                    guidance,
                    image_emb,
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
                guidance = guidance * 1000
                conditioning += self.guidance_embedder(guidance, hidden_states.dtype)
            conditioning += self.pooled_text_embedder(pooled_prompt_emb)
            rope_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
            text_rope_emb = rope_emb[:, :, : text_ids.size(1)]
            image_rope_emb = rope_emb[:, :, text_ids.size(1) :]

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

                # first block
                original_hidden_states = hidden_states
                hidden_states, prompt_emb = self.blocks[0](
                    hidden_states, prompt_emb, conditioning, rope_emb, image_emb, attn_kwargs
                )
                first_hidden_states_residual = hidden_states - original_hidden_states

                (first_hidden_states_residual,) = sequence_parallel_unshard(
                    (first_hidden_states_residual,), seq_dims=(1,), seq_lens=(image_seq_len,)
                )

                if self.step_count == 0 or self.step_count == (self.num_inference_steps - 1):
                    should_calc = True
                else:
                    skip = self.is_relative_l1_below_threshold(
                        first_hidden_states_residual,
                        self.prev_first_hidden_states_residual,
                        threshold=self.relative_l1_threshold,
                    )
                    should_calc = not skip
                self.step_count += 1

                if not should_calc:
                    hidden_states += self.previous_residual
                else:
                    self.prev_first_hidden_states_residual = first_hidden_states_residual

                    first_hidden_states = hidden_states.clone()
                    for i, block in enumerate(self.blocks[1:]):
                        hidden_states, prompt_emb = block(
                            hidden_states, prompt_emb, conditioning, rope_emb, image_emb, attn_kwargs
                        )
                        if len(controlnet_double_block_output) > 0:
                            interval_control = len(self.blocks) / len(controlnet_double_block_output)
                            interval_control = int(np.ceil(interval_control))
                            hidden_states = hidden_states + controlnet_double_block_output[i // interval_control]
                    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
                    for i, block in enumerate(self.single_blocks):
                        hidden_states = block(hidden_states, conditioning, rope_emb, image_emb, attn_kwargs)
                        if len(controlnet_single_block_output) > 0:
                            interval_control = len(self.single_blocks) / len(controlnet_double_block_output)
                            interval_control = int(np.ceil(interval_control))
                            hidden_states = hidden_states + controlnet_single_block_output[i // interval_control]

                    hidden_states = hidden_states[:, prompt_emb.shape[1] :]

                    previous_residual = hidden_states - first_hidden_states
                    self.previous_residual = previous_residual

                hidden_states = self.final_norm_out(hidden_states, conditioning)
                hidden_states = self.final_proj_out(hidden_states)
                (hidden_states,) = sequence_parallel_unshard((hidden_states,), seq_dims=(1,), seq_lens=(image_seq_len,))

            (hidden_states,) = cfg_parallel_unshard((hidden_states,), use_cfg=use_cfg)

            return hidden_states

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        in_channel: int = 64,
        relative_l1_threshold: float = 0.05,
    ):
        model = cls(
            device="meta",
            dtype=dtype,
            in_channel=in_channel,
            relative_l1_threshold=relative_l1_threshold,
        )
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
