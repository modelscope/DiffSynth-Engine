import torch
from math import prod
from typing import Any, Dict, List, Optional

from diffsynth_engine.models.qwen_image import QwenImageDiT
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)


class QwenImageDiTFBCache(QwenImageDiT):
    def __init__(
        self,
        num_layers: int = 60,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        relative_l1_threshold: float = 0.05,
        zero_cond_t: bool = False,
    ):
        super().__init__(num_layers=num_layers, device=device, dtype=dtype, zero_cond_t=zero_cond_t)
        self.relative_l1_threshold = relative_l1_threshold
        self.step_count = 0
        self.num_inference_steps = 0
        self._cache_stream = 0
        self._cache_states = [self._new_cache_state()]

    @staticmethod
    def _new_cache_state():
        return {
            "step_count": 0,
            "prev_first_hidden_states_residual": None,
            "previous_residual": None,
        }

    def is_relative_l1_below_threshold(self, prev_residual, residual, threshold):
        if threshold <= 0.0:
            return False

        if prev_residual.shape != residual.shape:
            return False

        mean_diff = (prev_residual - residual).abs().mean()
        mean_prev_residual = prev_residual.abs().mean()
        diff = mean_diff / mean_prev_residual
        return diff.item() < threshold

    def refresh_cache_status(self, num_inference_steps, num_cache_streams=1):
        self.step_count = 0
        self.num_inference_steps = num_inference_steps
        self._cache_stream = 0
        self._cache_states = [self._new_cache_state() for _ in range(num_cache_streams)]

    def set_cache_stream(self, stream_index):
        if stream_index < 0 or stream_index >= len(self._cache_states):
            raise ValueError(f"Invalid FB-cache stream index: {stream_index}")
        self._cache_stream = stream_index
        self.step_count = self._cache_states[stream_index]["step_count"]

    def forward(
        self,
        image: torch.Tensor,
        edit: torch.Tensor = None,
        text: torch.Tensor = None,
        timestep: torch.LongTensor = None,
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
                    [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in [video_fhw]],
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
                cache_state = self._cache_states[self._cache_stream]
                rotary_emb = (img_freqs, txt_freqs)
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

                if cache_state["step_count"] == 0 or cache_state["step_count"] == (self.num_inference_steps - 1):
                    should_calc = True
                else:
                    skip = self.is_relative_l1_below_threshold(
                        first_hidden_states_residual,
                        cache_state["prev_first_hidden_states_residual"],
                        threshold=self.relative_l1_threshold,
                    )
                    should_calc = not skip
                cache_state["step_count"] += 1
                self.step_count = cache_state["step_count"]

                if not should_calc:
                    image += cache_state["previous_residual"]
                else:
                    cache_state["prev_first_hidden_states_residual"] = first_hidden_states_residual
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
                    cache_state["previous_residual"] = image - first_hidden_states

                if self.zero_cond_t:
                    conditioning = conditioning.chunk(2, dim=0)[0]
                image = self.norm_out(image, conditioning)
                image = self.proj_out(image)
                (image,) = sequence_parallel_unshard((image,), seq_dims=(1,), seq_lens=(image_seq_len,))
            image = image[:, :image_seq_len]
            image = self.unpatchify(image, h, w)

        (image,) = cfg_parallel_unshard((image,), use_cfg=use_cfg)
        return image

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
