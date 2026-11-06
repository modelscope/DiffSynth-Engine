"""First-Block Cache (FB-Cache) for the Qwen-Image DiT.

Idea
----
Adjacent denoising steps at high sigma differ very little. Instead of running
all 60 blocks every step, run block 0 only, compare its residual with the one
recorded on the previous step, and -- if the change is below a threshold --
reuse the previous step's *total* residual instead of running blocks 1..59.

Block 0 is the cheapest possible probe: one block out of 60, and its residual
already reflects the current step's conditioning.

Two guards keep the approximation from drifting:

* ``warmup_steps`` -- always compute the first N steps. The early steps decide
  the layout, and a wrong layout cannot be recovered later.
* ``max_skip_streak`` -- compute at least once every M steps. Repeated reuse
  compounds error, and this bounds it.

CFG has two independent branches
--------------------------------
With classifier-free guidance the pipeline calls the model twice per step: once
with the positive prompt embeddings and once with the negative ones, and those
have different sequence lengths (14 vs 12 tokens for the default test prompt).
The cached residuals therefore cannot be shared between them -- storing one
branch's residual and reusing it in the other compares tensors of different
shapes. Each branch gets its own state, keyed by ``fbcache_branch``.

Multi-rank correctness
----------------------
With sequence parallelism each rank holds a different shard, so a per-rank
threshold test can make ranks disagree: one takes the cached path (which skips
the collectives inside blocks 1..59) while another runs them, and the HCCL
group hangs. The decision value is therefore all-reduced with MAX before it is
compared against the threshold: if any rank wants to compute, everybody
computes.
"""

from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffsynth_engine.distributed.utils import sequence_parallel_shard, sequence_parallel_unshard
from diffsynth_engine.models.qwen_image.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    compute_text_seq_len_from_mask,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

__all__ = ["QwenImageTransformer2DModelFBCache"]


class QwenImageTransformer2DModelFBCache(QwenImageTransformer2DModel):
    """Qwen-Image DiT with first-block caching enabled.

    Extra ``__init__`` kwargs (not part of the diffusers config, so they keep
    their defaults when the model is built via ``from_config``):

    Args:
        fbcache_threshold: relative L1 change of block 0's residual below which
            the remaining blocks are skipped.
        fbcache_warmup_steps: number of leading steps that always compute.
        fbcache_max_skip_streak: force a full computation after this many
            consecutive skips (0 disables the guard).
    """

    def __init__(
        self,
        *args,
        fbcache_threshold: float = 0.05,
        fbcache_warmup_steps: int = 1,
        fbcache_max_skip_streak: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fbcache_threshold = fbcache_threshold
        self.fbcache_warmup_steps = fbcache_warmup_steps
        self.fbcache_max_skip_streak = fbcache_max_skip_streak
        self.reset_fbcache()

    # ------------------------------------------------------------------ state
    def reset_fbcache(self, total_steps: int | None = None) -> None:
        """Drop cached residuals. Call before each new sampling run.

        Args:
            total_steps: number of denoising steps about to run. Used to force a
                full computation on the last one; leave None if unknown.
        """
        self._fb_states: Dict[str, Dict[str, Any]] = {}
        self._fb_total_steps = total_steps

    def _fb_state(self, branch: str) -> Dict[str, Any]:
        state = self._fb_states.get(branch)
        if state is None:
            state = {
                "step": 0,
                "skip_streak": 0,
                "skip_total": 0,
                "prev_first": None,
                "prev_img": None,
                "prev_txt": None,
            }
            self._fb_states[branch] = state
        return state

    def _fb_sync_max(self, value: torch.Tensor) -> torch.Tensor:
        """MAX all-reduce inside the sequence-parallel group.

        It has to be the SP group, not the world: with CFG parallelism
        (``use_cfg_parallel=True``) different ranks run different branches, and
        a world-wide reduction would let the two branches' decisions cancel each
        other out. ``GroupCoordinator.all_reduce`` already shortcuts when the
        group holds a single rank.
        """
        from diffsynth_engine.distributed.parallel_state import get_sp_group, is_sp_group_initialized

        if not is_sp_group_initialized():
            return value.detach()
        buf = value.detach().reshape(1).float().clone()
        get_sp_group().all_reduce(buf, op=dist.ReduceOp.MAX)
        return buf.reshape(())

    def _fb_should_calc(self, state: Dict[str, Any], first_residual: torch.Tensor) -> bool:
        if state["step"] == 0 or state["prev_first"] is None:
            return True
        if state["step"] < self.fbcache_warmup_steps:
            return True
        # The last step settles the final detail, so approximating it is the most
        # visible error the cache could make.
        if self._fb_total_steps is not None and state["step"] >= self._fb_total_steps - 1:
            return True
        if self.fbcache_max_skip_streak > 0 and state["skip_streak"] >= self.fbcache_max_skip_streak:
            return True

        cur = first_residual.float()
        prev = state["prev_first"].float()
        # relative L1: normalise by the previous magnitude so the threshold is
        # scale-free (latent magnitudes shift a lot between steps)
        rel = (cur - prev).abs().mean() / prev.abs().mean().clamp(min=1e-8)
        rel = self._fb_sync_max(rel)
        return bool(rel.item() >= self.fbcache_threshold)

    # ------------------------------------------------------------------ model
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
        fbcache_branch: str = "default",
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        state = self._fb_state(fbcache_branch)

        image_seq_len = hidden_states.shape[1]
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        img_freqs, txt_freqs = image_rotary_emb
        hidden_states, encoder_hidden_states, img_freqs, txt_freqs, modulate_index = sequence_parallel_shard(
            (hidden_states, encoder_hidden_states, img_freqs, txt_freqs, modulate_index),
            seq_dims=(1, 1, 0, 0, 1),
        )
        image_rotary_emb = (img_freqs, txt_freqs)

        blocks = self.transformer_blocks
        img_stream_in = hidden_states
        txt_stream_in = encoder_hidden_states

        # Block 0 always runs: its residual is the probe for this step.
        encoder_hidden_states, hidden_states = blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=block_attention_kwargs,
            modulate_index=modulate_index,
        )
        first_residual = hidden_states - img_stream_in

        # ControlNet injects an external residual after every block. The cached
        # path skips the loop entirely and would drop it, so never cache then.
        use_cache = (
            controlnet_block_samples is None
            and not self._fb_should_calc(state, first_residual)
            and state["prev_img"] is not None
        )

        if use_cache:
            hidden_states = img_stream_in + state["prev_img"]
            encoder_hidden_states = txt_stream_in + state["prev_txt"]
            state["skip_streak"] += 1
            state["skip_total"] += 1
        else:
            for index_block, block in enumerate(blocks[1:], start=1):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )
                # same controlnet residual the base forward applies
                if controlnet_block_samples is not None:
                    interval_control = len(blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            # detach: these outlive the forward pass, and keeping the graph would
            # pin every intermediate tensor of the step
            state["prev_img"] = (hidden_states - img_stream_in).detach()
            state["prev_txt"] = (encoder_hidden_states - txt_stream_in).detach()
            state["prev_first"] = first_residual.detach()
            state["skip_streak"] = 0

        logger.debug(f"fbcache[{fbcache_branch}] step {state['step']}: {'skip' if use_cache else 'compute'}")
        state["step"] += 1

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        (output,) = sequence_parallel_unshard((output,), seq_dims=(1,), seq_lens=(image_seq_len,))

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
