# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import os
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
from accelerate import init_empty_weights
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel

from diffsynth_engine.configs.wan import WanPipelineConfig
from diffsynth_engine.distributed.parallel_state import get_cfg_group, model_parallel_is_initialized
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.layers.attention import get_attn_backend
from diffsynth_engine.models.wan import AutoencoderKLWan, WanTransformer3DModel
from diffsynth_engine.pipelines.base import Pipeline
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.load_utils import load_model_weights

logger = logging.get_logger(__name__)


def basic_clean(text):
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(Pipeline):
    r"""
    Pipeline for text-to-video generation using Wan, adapted for DiffSynth-Engine.

    Changes from the original diffusers implementation:
    - Inherits from Pipeline (DiffSynth-Engine) instead of DiffusionPipeline
    - Removed WanLoraLoaderMixin (LoRA loading support)
    - Removed register_modules() — components are assigned directly
    - Removed model_cpu_offload_seq — CPU offload sequence declaration (DiffusionPipeline feature)
    - Removed _execution_device property — replaced with self.device
    - Removed maybe_free_model_hooks() — model offload hooks (DiffusionPipeline feature)
    - Removed replace_example_docstring decorator
    - Removed XLA-related logic (xm.mark_step, XLA_AVAILABLE)
    - Removed cache_context wrapping for transformer forward passes
    - Removed transformer_2 / boundary_ratio / expand_timesteps (Wan2.2 two-stage denoising)
    - Reimplemented from_pretrained as classmethod with model_path_or_config pattern
    - Added set_forward_context for transformer initialization and inference
    - Added _build_attn_metadata for attention metadata construction
    - Added _predict_noise_with_cfg for CFG-parallel denoising support
    - Added attn_backend initialization for DiffSynth-Engine attention system

    Args:
        pipeline_config (`WanPipelineConfig`):
            Configuration for the pipeline.
        tokenizer (`AutoTokenizer`):
            Tokenizer from T5, specifically the google/umt5-xxl variant.
        text_encoder (`UMT5EncoderModel`):
            T5 text encoder, specifically the google/umt5-xxl variant.
        vae (`AutoencoderKLWan`):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer (`WanTransformer3DModel`):
            Conditional Transformer to denoise the input latents.
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        pipeline_config: WanPipelineConfig,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(pipeline_config)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if self.vae is not None else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if self.vae is not None else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        head_dim = transformer.config.attention_head_dim
        self.attn_backend = get_attn_backend(
            head_size=head_dim,
            attn_type=pipeline_config.attn_type,
        )

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | WanPipelineConfig):
        """
        Load a WanPipeline from a pretrained model path or config.

        Args:
            model_path_or_config: Either a string path to the model directory or a WanPipelineConfig instance.

        Returns:
            WanPipeline: The loaded pipeline.
        """
        if isinstance(model_path_or_config, str):
            pipeline_config = WanPipelineConfig(model_path=model_path_or_config)
        else:
            pipeline_config = model_path_or_config

        if not os.path.exists(pipeline_config.model_path):
            raise FileNotFoundError(f"Model path not found: {pipeline_config.model_path}")

        # Load transformer
        transformer = cls.init_transformer(pipeline_config)

        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pipeline_config.model_path,
            subfolder="scheduler",
        )

        # Load VAE
        vae = cls.init_vae(pipeline_config)

        # Load text encoder
        text_encoder = cls.init_text_encoder(pipeline_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pipeline_config.model_path,
            subfolder="tokenizer",
        )

        return cls(
            pipeline_config=pipeline_config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    @staticmethod
    def init_transformer(pipeline_config: WanPipelineConfig, empty_weights: bool = False):
        logger.info("Initializing transformer...")
        with set_forward_context(attn_type=pipeline_config.attn_type):
            if empty_weights:
                with init_empty_weights():
                    config_dict = WanTransformer3DModel.load_config(
                        pipeline_config.model_path,
                        subfolder="transformer",
                        local_files_only=True,
                    )
                    model = WanTransformer3DModel.from_config(config_dict)
            else:
                model = WanTransformer3DModel.from_pretrained(
                    pipeline_config.model_path,
                    subfolder="transformer",
                    device=pipeline_config.device,
                    dtype=pipeline_config.model_dtype,
                )
        return model

    @staticmethod
    def init_text_encoder(pipeline_config: WanPipelineConfig, empty_weights: bool = False):
        logger.info("Initializing text encoder...")
        if empty_weights:
            with init_empty_weights():
                model = UMT5EncoderModel.from_pretrained(
                    pipeline_config.model_path,
                    subfolder="text_encoder",
                    local_files_only=True,
                )
            return model

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="text_encoder",
            device=pipeline_config.device,
            dtype=pipeline_config.text_encoder_dtype,
        )
        with init_empty_weights():
            model = UMT5EncoderModel.from_pretrained(
                pipeline_config.model_path,
                subfolder="text_encoder",
                local_files_only=True,
            )
        model.load_state_dict(state_dict, strict=False, assign=True)
        model.to(device=pipeline_config.device)
        return model

    @staticmethod
    def init_vae(pipeline_config: WanPipelineConfig, empty_weights: bool = False):
        logger.info("Initializing VAE...")
        if empty_weights:
            with init_empty_weights():
                config_dict = AutoencoderKLWan.load_config(
                    pipeline_config.model_path,
                    subfolder="vae",
                    local_files_only=True,
                )
                model = AutoencoderKLWan.from_config(config_dict)
            return model

        model = AutoencoderKLWan.from_pretrained(
            pipeline_config.model_path,
            subfolder="vae",
            device=pipeline_config.device,
            dtype=pipeline_config.vae_dtype,
        )
        return model

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            max_sequence_length (`int`, *optional*, defaults to 226):
                Maximum sequence length for the text encoder.
            device (`torch.device`, *optional*):
                torch device
            dtype (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def _build_attn_metadata(self, attn_params):
        if attn_params is None:
            return None

        builder_cls = self.attn_backend.get_builder_cls()
        builder = builder_cls()
        attn_params_dict = attn_params.to_dict()
        attn_metadata = builder.build(**attn_params_dict)
        return attn_metadata

    def _predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        attn_metadata,
        apply_cfg: bool,
        guidance_scale: float,
        use_cfg_parallel: bool,
        batch_size: int,
    ):
        """
        Predict noise with optional classifier-free guidance and CFG parallelism.

        Args:
            latents: Current noisy latents, shape (batch, channels, frames, height, width).
            timestep: Current timestep tensor, shape (batch,).
            prompt_embeds: Positive prompt embeddings tensor.
            negative_prompt_embeds: Negative prompt embeddings tensor.
            attn_metadata: Attention metadata for set_forward_context.
            apply_cfg: Whether to apply classifier-free guidance this step.
            guidance_scale: The CFG scale factor.
            use_cfg_parallel: Whether to use CFG parallelism across devices.
            batch_size: The actual batch size.

        Returns:
            noise_pred: The predicted noise tensor.
        """
        transformer_dtype = self.transformer.dtype

        if not apply_cfg:
            latent_model_input = latents.to(transformer_dtype)
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            return noise_pred.float()

        # CFG mode
        cfg_group, cfg_rank = None, None
        if use_cfg_parallel:
            if not model_parallel_is_initialized():
                raise RuntimeError("Model parallel groups must be initialized when use_cfg_parallel=True")
            cfg_group = get_cfg_group()
            cfg_rank = cfg_group.rank_in_group

        latent_model_input = latents.to(transformer_dtype)

        noise_pred_pos = torch.zeros_like(latents, dtype=torch.float32)
        noise_pred_neg = torch.zeros_like(latents, dtype=torch.float32)

        # Positive prompt forward pass
        if not (use_cfg_parallel and cfg_rank != 0):
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_pos = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0].float()

        # Negative prompt forward pass
        if not use_cfg_parallel or cfg_rank != 0:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_neg = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    return_dict=False,
                )[0].float()

        # All-reduce for CFG parallel
        if use_cfg_parallel:
            noise_pred_pos = cfg_group.all_reduce(noise_pred_pos)
            noise_pred_neg = cfg_group.all_reduce(noise_pred_neg)

        # Apply CFG
        noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation.
            height (`int`, defaults to `480`):
                The height in pixels of the generated video.
            width (`int`, defaults to `832`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale for classifier-free guidance.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random generator(s) for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `WanPipelineOutput` instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Kwargs passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                Tensor inputs for the callback function.
            max_sequence_length (`int`, defaults to `512`):
                Maximum sequence length for the text encoder.

        Returns:
            `WanPipelineOutput` or `tuple`: Generated video frames.
        """
        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        patch_size = self.transformer.config.patch_size
        h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) for proper "
                f"patchification. Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        actual_batch_size = batch_size * num_videos_per_prompt

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0])

                attn_metadata = self._build_attn_metadata(self.pipeline_config.attn_params)

                noise_pred = self._predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    attn_metadata=attn_metadata,
                    apply_cfg=self.do_classifier_free_guidance,
                    guidance_scale=guidance_scale,
                    use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                    batch_size=actual_batch_size,
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
