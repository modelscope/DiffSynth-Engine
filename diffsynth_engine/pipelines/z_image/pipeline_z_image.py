# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py

# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from accelerate import init_empty_weights
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from diffsynth_engine.configs.z_image import ZImagePipelineConfig
from diffsynth_engine.distributed.parallel_state import get_cfg_group, model_parallel_is_initialized
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.layers.attention import get_attn_backend
from diffsynth_engine.models.z_image import ZImageTransformer2DModel
from diffsynth_engine.pipelines.base import Pipeline
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.load_utils import fix_state_dict_key, load_model_weights

logger = logging.get_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ZImagePipeline(Pipeline):
    r"""
    The Z-Image pipeline for text-to-image generation, adapted for DiffSynth-Engine.

    Changes from the original diffusers implementation:
    - Inherits from Pipeline (DiffSynth-Engine) instead of DiffusionPipeline
    - Removed ZImageLoraLoaderMixin (LoRA loading support)
    - Removed FromSingleFileMixin (single-file model loading)
    - Removed register_modules() — components are assigned directly
    - Removed model_cpu_offload_seq — CPU offload sequence declaration (DiffusionPipeline feature)
    - Removed _execution_device property — replaced with self.device
    - Removed maybe_free_model_hooks() — model offload hooks (DiffusionPipeline feature)
    - Removed replace_example_docstring decorator
    - Reimplemented from_pretrained as classmethod with model_path_or_config pattern
    - Added set_forward_context for transformer initialization and inference
    - Added _build_attn_metadata for attention metadata construction
    - Added _predict_noise_with_cfg for CFG-parallel denoising support
    - Added attn_backend initialization for DiffSynth-Engine attention system

    Args:
        pipeline_config (`ZImagePipelineConfig`):
            Configuration for the pipeline.
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder (`PreTrainedModel`):
            Text encoder model for encoding prompts into embeddings.
        tokenizer (`AutoTokenizer`):
            Tokenizer for the text encoder.
        transformer (`ZImageTransformer2DModel`):
            Conditional Transformer architecture to denoise the encoded image latents.
    """

    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        pipeline_config: ZImagePipelineConfig,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ):
        super().__init__(pipeline_config)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if self.vae is not None else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        head_dim = transformer.config.dim // transformer.config.n_heads
        self.attn_backend = get_attn_backend(
            head_size=head_dim,
            attn_type=pipeline_config.attn_type,
        )

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | ZImagePipelineConfig):
        """
        Load a ZImagePipeline from a pretrained model path or config.

        Args:
            model_path_or_config: Either a string path to the model directory or a ZImagePipelineConfig instance.

        Returns:
            ZImagePipeline: The loaded pipeline.
        """
        if isinstance(model_path_or_config, str):
            pipeline_config = ZImagePipelineConfig(model_path=model_path_or_config)
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

        # Initialize pipeline
        return cls(
            pipeline_config=pipeline_config,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

    @staticmethod
    def init_transformer(pipeline_config: ZImagePipelineConfig, empty_weights: bool = False):
        logger.info("Initializing transformer...")
        with set_forward_context(attn_type=pipeline_config.attn_type):
            if empty_weights:
                with init_empty_weights():
                    config_dict = ZImageTransformer2DModel.load_config(
                        pipeline_config.model_path,
                        subfolder="transformer",
                        local_files_only=True,
                    )
                    model = ZImageTransformer2DModel.from_config(config_dict)
            else:
                model = ZImageTransformer2DModel.from_pretrained(
                    pipeline_config.model_path,
                    subfolder="transformer",
                    device=pipeline_config.device,
                    dtype=pipeline_config.model_dtype,
                )
        return model

    @staticmethod
    def init_text_encoder(pipeline_config: ZImagePipelineConfig, empty_weights: bool = False):
        logger.info("Initializing text encoder...")
        with init_empty_weights():
            config = AutoConfig.from_pretrained(
                pipeline_config.model_path,
                subfolder="text_encoder",
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_config(config)

        if empty_weights:
            return model

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="text_encoder",
            device=pipeline_config.device,
            dtype=pipeline_config.text_encoder_dtype,
        )
        if key_mapping := getattr(model, "_checkpoint_conversion_mapping", None):
            state_dict = fix_state_dict_key(state_dict, key_mapping)
        model.load_state_dict(state_dict, strict=False, assign=True)
        model.tie_weights()
        model.to(device=pipeline_config.device)
        return model

    @staticmethod
    def init_vae(pipeline_config: ZImagePipelineConfig, empty_weights: bool = False):
        logger.info("Initializing VAE...")
        with init_empty_weights():
            config_dict = AutoencoderKL.load_config(
                pipeline_config.model_path,
                subfolder="vae",
                local_files_only=True,
            )
            model = AutoencoderKL.from_config(config_dict)

        if empty_weights:
            return model

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="vae",
            device=pipeline_config.device,
            dtype=pipeline_config.vae_dtype,
        )
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device=pipeline_config.device)
        return model

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        device = device or self.device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

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
        prompt_embeds: List[torch.FloatTensor],
        negative_prompt_embeds: List[torch.FloatTensor],
        attn_metadata,
        apply_cfg: bool,
        guidance_scale: float,
        cfg_normalization: bool,
        use_cfg_parallel: bool,
        actual_batch_size: int,
    ):
        """
        Predict noise with optional classifier-free guidance and CFG parallelism.

        Args:
            latents: Current noisy latents, shape (batch, channels, height, width).
            timestep: Current timestep tensor, shape (batch,).
            prompt_embeds: List of positive prompt embeddings (variable length per item).
            negative_prompt_embeds: List of negative prompt embeddings (variable length per item).
            attn_metadata: Attention metadata for set_forward_context.
            apply_cfg: Whether to apply classifier-free guidance this step.
            guidance_scale: The CFG scale factor.
            cfg_normalization: Whether to apply CFG renormalization.
            use_cfg_parallel: Whether to use CFG parallelism across devices.
            actual_batch_size: The actual batch size (batch_size * num_images_per_prompt).

        Returns:
            noise_pred: The predicted noise tensor.
        """
        if not apply_cfg:
            # No CFG: single forward pass
            latent_model_input = latents.to(self)
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            with set_forward_context(attn_metadata=attn_metadata):
                model_out_list = self.transformer(latent_model_input_list, timestep, prompt_embeds, return_dict=False)[
                    0
                ]

            noise_pred = torch.stack([tensor.float() for tensor in model_out_list], dim=0)
            return noise_pred

        # CFG mode
        cfg_group, cfg_rank = None, None
        if use_cfg_parallel:
            if not model_parallel_is_initialized():
                raise RuntimeError("Model parallel groups must be initialized when use_cfg_parallel=True")
            cfg_group = get_cfg_group()
            cfg_rank = cfg_group.rank_in_group

        latents_typed = latents.to(self.pipeline_config.model_dtype)
        latents_typed = latents_typed.unsqueeze(2)

        pos_out_list = [torch.zeros_like(latents_typed[0])] * actual_batch_size
        neg_out_list = [torch.zeros_like(latents_typed[0])] * actual_batch_size

        # Positive prompt forward pass
        if not (use_cfg_parallel and cfg_rank != 0):
            latent_model_input_list = list(latents_typed.unbind(dim=0))
            with set_forward_context(attn_metadata=attn_metadata):
                pos_out_list = self.transformer(latent_model_input_list, timestep, prompt_embeds, return_dict=False)[0]

        # Negative prompt forward pass
        if not use_cfg_parallel or cfg_rank != 0:
            latent_model_input_list = list(latents_typed.unbind(dim=0))
            with set_forward_context(attn_metadata=attn_metadata):
                neg_out_list = self.transformer(
                    latent_model_input_list, timestep, negative_prompt_embeds, return_dict=False
                )[0]

        # All-reduce for CFG parallel
        pos_out = torch.stack([tensor.float() for tensor in pos_out_list], dim=0)
        neg_out = torch.stack([tensor.float() for tensor in neg_out_list], dim=0)

        if use_cfg_parallel:
            pos_out = cfg_group.all_reduce(pos_out)
            neg_out = cfg_group.all_reduce(neg_out)

        # Apply CFG
        noise_pred_list = []
        for j in range(actual_batch_size):
            pos = pos_out[j]
            neg = neg_out[j]
            pred = pos + guidance_scale * (pos - neg)

            # Renormalization
            if cfg_normalization:
                ori_pos_norm = torch.linalg.vector_norm(pos)
                new_pos_norm = torch.linalg.vector_norm(pred)
                if new_pos_norm > ori_pos_norm:
                    pred = pred * (ori_pos_norm / new_pos_norm)

            noise_pred_list.append(pred)

        noise_pred = torch.stack(noise_pred_list, dim=0)
        return noise_pred

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale for classifier-free guidance.
            cfg_normalization (`bool`, *optional*, defaults to False):
                Whether to apply CFG renormalization.
            cfg_truncation (`float`, *optional*, defaults to 1.0):
                Time-aware truncation for CFG. When normalized time exceeds this value, CFG is disabled.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt or prompts.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random generator(s) for deterministic generation.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `ZImagePipelineOutput` instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                Kwargs passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                Tensor inputs for the callback function.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for the prompt.

        Returns:
            `ZImagePipelineOutput` or `tuple`: Generated images.
        """
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self.device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # Encode prompts
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # Prepare timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware CFG truncation (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle CFG truncation
                current_guidance_scale = self.guidance_scale
                if self.do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                    if t_norm > cfg_truncation:
                        current_guidance_scale = 0.0

                # Determine whether to apply CFG this step
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                attn_metadata = self._build_attn_metadata(self.pipeline_config.attn_params)

                noise_pred = self._predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    attn_metadata=attn_metadata,
                    apply_cfg=apply_cfg,
                    guidance_scale=current_guidance_scale,
                    cfg_normalization=cfg_normalization,
                    use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                    actual_batch_size=actual_batch_size,
                )

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)
