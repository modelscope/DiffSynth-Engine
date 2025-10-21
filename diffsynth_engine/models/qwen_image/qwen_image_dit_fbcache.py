import torch
from typing import Any, Dict, Optional

from diffsynth_engine.models.qwen_image import QwenImageDiT
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import cfg_parallel, cfg_parallel_unshard


class QwenImageDiTFBCache(QwenImageDiT):
    def __init__(
        self,
        num_layers: int = 60,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        relative_l1_threshold: float = 0.05,
    ):
        super().__init__(num_layers=num_layers, device=device, dtype=dtype)
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
        image: torch.Tensor,
        text: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        txt_seq_lens: torch.LongTensor = None,
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
                    text,
                    timestep,
                    txt_seq_lens,
                ),
                use_cfg=use_cfg,
            ),
        ):
            conditioning = self.time_text_embed(timestep, image.dtype)
            video_fhw = (1, h // 2, w // 2)  # frame, height, width
            max_length = txt_seq_lens.max().item()
            image_rotary_emb = self.pos_embed(video_fhw, max_length, image.device)

            image = self.patchify(image)
            image = self.img_in(image)
            text = self.txt_in(self.txt_norm(text[:, :max_length]))

            # first block
            original_hidden_states = image
            text, image = self.transformer_blocks[0](
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
                attn_kwargs=attn_kwargs,
            )
            first_hidden_states_residual = image - original_hidden_states

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
                image += self.previous_residual
            else:
                self.prev_first_hidden_states_residual = first_hidden_states_residual
                first_hidden_states = image.clone()

                for block in self.transformer_blocks[1:]:
                    text, image = block(
                        image=image,
                        text=text,
                        temb=conditioning,
                        image_rotary_emb=image_rotary_emb,
                        attn_kwargs=attn_kwargs,
                    )

                previous_residual = image - first_hidden_states
                self.previous_residual = previous_residual

            image = self.norm_out(image, conditioning)
            image = self.proj_out(image)

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
    ):
        model = cls(
            device="meta",
            dtype=dtype,
            num_layers=num_layers,
            relative_l1_threshold=relative_l1_threshold,
        )
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
