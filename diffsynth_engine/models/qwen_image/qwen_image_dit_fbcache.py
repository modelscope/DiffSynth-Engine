import torch
from math import prod
from typing import Any, Dict, List, Optional, Tuple

from diffsynth_engine.models.qwen_image.qwen_image_dit import QwenImageDiT
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)


class QwenImageDiTFBCache(QwenImageDiT):
    """Qwen-Image DiT with First-Block Cache (FB-Cache).

    Differences from the previous upstream implementation:
    1. forward() signature aligned with the current QwenImageDiT: adds edit /
       context_latents / entity_text / entity_seq_lens / entity_masks, and
       txt_seq_lens is renamed to text_seq_lens
    2. zero_cond_t path restored (required by the 2511 model) and propagated
       from use_zero_cond_t in from_state_dict
    3. cache logic wrapped in the sequence_parallel context so it also holds
       when the sequence is sharded
    4. skip decision is all_reduce(MAX)'d across the SP group so that every rank
       takes the same branch (otherwise HCCL hangs)
    """

    def __init__(
        self,
        num_layers: int = 60,
        zero_cond_t: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        relative_l1_threshold: float = 0.05,
        warmup_steps: int = 0,
        max_skip_streak: int = 0,
    ):
        super().__init__(num_layers=num_layers, zero_cond_t=zero_cond_t, device=device, dtype=dtype)
        self.relative_l1_threshold = relative_l1_threshold
        # Force computation for the first N steps: early denoising changes the most
        self.warmup_steps = warmup_steps
        # Max consecutive skips (0 = unlimited); keeps the cached path from drifting
        self.max_skip_streak = max_skip_streak
        self.step_count = 0
        self.num_inference_steps = 0
        self.skip_streak = 0
        self.previous_residual = None
        self.prev_first_hidden_states_residual = None

    # ------------------------------------------------------------------ #
    # Cache decision
    # ------------------------------------------------------------------ #

    def is_relative_l1_below_threshold(self, prev_residual, residual, threshold) -> bool:
        if threshold <= 0.0:
            return False
        if prev_residual is None or residual is None:
            return False
        if prev_residual.shape != residual.shape:
            return False

        mean_diff = (prev_residual - residual).abs().mean()
        mean_prev_residual = prev_residual.abs().mean()
        if float(mean_prev_residual) == 0.0:
            return False
        diff = mean_diff / mean_prev_residual

        # Take a MAX across the SP group so every rank takes the same branch.
        # Otherwise some ranks skip blocks[1:] (which contain collective ops) and
        # HCCL either hangs or produces a corrupted image.
        import torch.distributed as dist

        from diffsynth_engine.utils.process_group import get_sp_group, get_sp_world_size

        if dist.is_initialized() and get_sp_world_size() > 1:
            t = diff.detach().to(torch.float32).reshape(1)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=get_sp_group())
            diff = t.reshape(())

        return diff.item() < threshold

    def refresh_cache_status(self, num_inference_steps: int):
        self.step_count = 0
        self.num_inference_steps = num_inference_steps
        self.skip_streak = 0
        self.previous_residual = None
        self.prev_first_hidden_states_residual = None

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        image: torch.Tensor,
        edit: Optional[List[torch.Tensor]] = None,
        timestep: torch.LongTensor = None,
        text: torch.Tensor = None,
        text_seq_lens: torch.LongTensor = None,
        context_latents: Optional[torch.Tensor] = None,
        entity_text: Optional[List[torch.Tensor]] = None,
        entity_seq_lens: Optional[List[torch.LongTensor]] = None,
        entity_masks: Optional[List[torch.Tensor]] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        h, w = image.shape[-2:]
        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        use_cfg = image.shape[0] > 1
        with (
            fp8_inference(fp8_linear_enabled),
            gguf_inference(),
            cfg_parallel(
                (
                    image,
                    *(edit if edit is not None else ()),
                    timestep,
                    text,
                    text_seq_lens,
                    *(entity_text if entity_text is not None else ()),
                    *(entity_seq_lens if entity_seq_lens is not None else ()),
                    *(entity_masks if entity_masks is not None else ()),
                    context_latents,
                ),
                use_cfg=use_cfg,
            ),
        ):
            if self.zero_cond_t:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = None
            conditioning = self.time_text_embed(timestep, image.dtype)
            video_fhw = [(1, h // 2, w // 2)]
            text_seq_len = text_seq_lens.max().item()
            image = self.patchify(image)
            image_seq_len = image.shape[1]

            if context_latents is not None:
                context_latents = context_latents.to(dtype=image.dtype)
                context_latents = self.patchify(context_latents)
                image = torch.cat([image, context_latents], dim=1)
                video_fhw += [(1, h // 2, w // 2)]
            if edit is not None:
                for img in edit:
                    img = img.to(dtype=image.dtype)
                    edit_h, edit_w = img.shape[-2:]
                    img = self.patchify(img)
                    image = torch.cat([image, img], dim=1)
                    video_fhw += [(1, edit_h // 2, edit_w // 2)]

            if self.zero_cond_t:
                modulate_index = torch.tensor(
                    [
                        [0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]])
                        for sample in [video_fhw]
                    ],
                    device=timestep.device,
                    dtype=torch.int,
                )
                modulate_index = modulate_index.unsqueeze(-1)

            rotary_emb = self.pos_embed(video_fhw, text_seq_len, image.device)

            image = self.img_in(image)
            text = self.txt_in(self.txt_norm(text[:, :text_seq_len]))

            attn_mask = None
            if entity_text is not None:
                text, rotary_emb, attn_mask = self.process_entity_masks(
                    text,
                    text_seq_lens,
                    rotary_emb,
                    video_fhw,
                    entity_text,
                    entity_seq_lens,
                    entity_masks,
                    image.device,
                    image.dtype,
                )

            img_freqs, txt_freqs = rotary_emb
            with sequence_parallel((image, text, img_freqs, txt_freqs, modulate_index), seq_dims=(1, 1, 0, 0, 1)):
                rotary_emb = (img_freqs, txt_freqs)

                # ---------------- FB-Cache core ----------------
                original_hidden_states = image
                text, image = self.transformer_blocks[0](
                    image=image,
                    text=text,
                    temb=conditioning,
                    rotary_emb=rotary_emb,
                    attn_mask=attn_mask,
                    attn_kwargs=attn_kwargs,
                    modulate_index=modulate_index,
                )
                first_hidden_states_residual = image - original_hidden_states

                is_last_step = self.num_inference_steps > 0 and self.step_count == (self.num_inference_steps - 1)
                in_warmup = self.step_count < self.warmup_steps
                streak_full = self.max_skip_streak > 0 and self.skip_streak >= self.max_skip_streak

                if self.step_count == 0 or is_last_step or in_warmup or streak_full:
                    should_calc = True
                else:
                    should_calc = not self.is_relative_l1_below_threshold(
                        self.prev_first_hidden_states_residual,
                        first_hidden_states_residual,
                        threshold=self.relative_l1_threshold,
                    )
                self.skip_streak = 0 if should_calc else (self.skip_streak + 1)
                self.step_count += 1

                if not should_calc and self.previous_residual is not None:
                    # Reuse the previous residual and skip blocks[1:]
                    image = image + self.previous_residual
                else:
                    self.prev_first_hidden_states_residual = first_hidden_states_residual
                    first_hidden_states = image.clone()
                    for block in self.transformer_blocks[1:]:
                        text, image = block(
                            image=image,
                            text=text,
                            temb=conditioning,
                            rotary_emb=rotary_emb,
                            attn_mask=attn_mask,
                            attn_kwargs=attn_kwargs,
                            modulate_index=modulate_index,
                        )
                    self.previous_residual = image - first_hidden_states
                # ------------------------------------------------

                if self.zero_cond_t:
                    conditioning = conditioning.chunk(2, dim=0)[0]
                image = self.norm_out(image, conditioning)
                image = self.proj_out(image)
                (image,) = sequence_parallel_unshard((image,), seq_dims=(1,), seq_lens=(image_seq_len,))

            image = image[:, :image_seq_len]
            image = self.unpatchify(image, h, w)

        (image,) = cfg_parallel_unshard((image,), use_cfg=use_cfg)
        return image

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        num_layers: int = 60,
        relative_l1_threshold: float = 0.05,
        use_zero_cond_t: bool = False,
    ):
        model = cls(
            device="meta",
            dtype=dtype,
            num_layers=num_layers,
            relative_l1_threshold=relative_l1_threshold,
            zero_cond_t=use_zero_cond_t,
        )
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
