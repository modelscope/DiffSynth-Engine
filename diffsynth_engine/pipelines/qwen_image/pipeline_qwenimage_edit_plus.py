# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py

# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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
import math
import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKLQwenImage
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffsynth_engine.configs.qwen_image import QwenImagePipelineConfig
from diffsynth_engine.distributed.parallel_state import get_cfg_group, model_parallel_is_initialized
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.layers.attention import get_attn_backend
from diffsynth_engine.models.qwen_image import QwenImageTransformer2DModel
from diffsynth_engine.pipelines.base import Pipeline
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.load_utils import fix_state_dict_key, load_model_weights

logger = logging.get_logger(__name__)


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


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


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    """Calculate dimensions based on target area and aspect ratio.

    Returns:
        Tuple[int, int]: (width, height)
    """
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class QwenImageEditPlusPipeline(Pipeline):
    r"""
    The Qwen-Image-Edit-Plus pipeline for image editing with simplified multi-image support.

    This is a simplified version that only supports Edit Plus style (dynamic image placeholders).

    Args:
        transformer ([`QwenImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`QwenTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        processor (`Qwen2VLProcessor`):
            Processor for handling multimodal inputs.
    """

    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        pipeline_config: QwenImagePipelineConfig,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__(pipeline_config)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024

        # Edit Plus style template (dynamic image placeholders)
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

        self.attn_backend = get_attn_backend(
            head_size=transformer.config.attention_head_dim,
            attn_type=pipeline_config.attn_type,
        )

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | QwenImagePipelineConfig):
        """
        Load a QwenImageEditPlusPipeline from a pretrained model path or config.

        Args:
            model_path_or_config: Either a string path to the model directory or a QwenImagePipelineConfig instance.

        Returns:
            QwenImageEditPlusPipeline: The loaded pipeline.
        """
        if isinstance(model_path_or_config, str):
            pipeline_config = QwenImagePipelineConfig(model_path=model_path_or_config)
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
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pipeline_config.model_path,
            subfolder="tokenizer",
        )

        # Load processor
        processor = Qwen2VLProcessor.from_pretrained(
            pipeline_config.model_path,
            subfolder="processor",
        )

        # Initialize pipeline
        return cls(
            pipeline_config=pipeline_config,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
        )

    @staticmethod
    def init_transformer(pipeline_config: QwenImagePipelineConfig, empty_weights: bool = False):
        with set_forward_context(attn_type=pipeline_config.attn_type):
            if empty_weights:
                with init_empty_weights():
                    config_dict = QwenImageTransformer2DModel.load_config(
                        pipeline_config.model_path,
                        subfolder="transformer",
                        local_files_only=True,
                    )
                    model = QwenImageTransformer2DModel.from_config(config_dict)
            else:
                model = QwenImageTransformer2DModel.from_pretrained(
                    pipeline_config.model_path,
                    subfolder="transformer",
                    device=pipeline_config.device,
                    dtype=pipeline_config.model_dtype,
                )
        return model

    @staticmethod
    def init_text_encoder(pipeline_config: QwenImagePipelineConfig, empty_weights: bool = False):
        logger.info("Initializing text encoder...")
        with init_empty_weights():
            config = Qwen2_5_VLConfig.from_pretrained(
                pipeline_config.model_path,
                subfolder="text_encoder",
                local_files_only=True,
            )
            model = Qwen2_5_VLForConditionalGeneration(config)

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
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device=pipeline_config.device)
        return model

    @staticmethod
    def init_vae(pipeline_config: QwenImagePipelineConfig, empty_weights: bool = False):
        logger.info("Initializing VAE...")
        with init_empty_weights():
            config_dict = AutoencoderKLQwenImage.load_config(
                pipeline_config.model_path,
                subfolder="vae",
                local_files_only=True,
            )
            model = AutoencoderKLQwenImage.from_config(config_dict)

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

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[torch.Tensor, List]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get prompt embeddings for the given prompts and images using Edit Plus style.

        Args:
            prompt: Single prompt string or list of prompts
            image: Can be:
                - Single PIL Image or tensor
                - List of PIL Images (multiple references)
            device: Target device
            dtype: Target dtype
        """
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Edit Plus style: dynamically build image placeholders
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        txt = [self.prompt_template_encode.format(base_img_prompt + e) for e in prompt]

        drop_idx = self.prompt_template_encode_start_idx

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values
            if hasattr(model_inputs, "pixel_values") and model_inputs.pixel_values is not None
            else None,
            image_grid_thw=model_inputs.image_grid_thw
            if hasattr(model_inputs, "image_grid_thw") and model_inputs.image_grid_thw is not None
            else None,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[Union[torch.Tensor, List]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""
        Encode prompt and images into embeddings.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image: Reference image(s) for encoding
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for prompt embeddings.
            max_sequence_length (`int`):
                Maximum sequence length for prompts.
        """
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        image = image.to(self.vae.dtype)
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents.to(self.pipeline_config.model_dtype)

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepare latents for denoising.

        Args:
            images: List of image tensors for single prompt (batch_size=1)
            batch_size: Number of samples to generate (must be 1 for multi-image)
            num_channels_latents: Number of latent channels
            height: Output height
            width: Output width
            dtype: Data type
            device: Target device
            generator: Random generator
            latents: Pre-generated latents (optional)
        """
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    img_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    img_latents = image
                if batch_size > img_latents.shape[0] and batch_size % img_latents.shape[0] == 0:
                    additional_image_per_prompt = batch_size // img_latents.shape[0]
                    img_latents = torch.cat([img_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > img_latents.shape[0] and batch_size % img_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {img_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    img_latents = torch.cat([img_latents], dim=0)

                image_latent_height, image_latent_width = img_latents.shape[3:]
                img_latents = self._pack_latents(
                    img_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(img_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

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
        image_latents: Optional[torch.Tensor],
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        img_shapes: List[List[tuple]],
        attn_metadata: Optional[Any],
        do_true_cfg: bool,
        true_cfg_scale: float,
        use_cfg_parallel: bool,
    ):
        """
        Predict noise with classifier-free guidance, supporting parallel CFG inference.

        Args:
            latents: Current latent tensor
            image_latents: Encoded reference image latents
            timestep: Current timestep
            prompt_embeds: Positive prompt embeddings
            prompt_embeds_mask: Attention mask for positive prompt
            negative_prompt_embeds: Negative prompt embeddings
            negative_prompt_embeds_mask: Attention mask for negative prompt
            img_shapes: Image shape information for transformer
            attn_metadata: Attention metadata
            do_true_cfg: Whether to apply classifier-free guidance
            true_cfg_scale: CFG scale factor
            use_cfg_parallel: Whether to use parallel CFG inference
        """
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        if not do_true_cfg:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]
            return noise_pred

        cfg_group, cfg_rank = None, None

        if use_cfg_parallel:
            if not model_parallel_is_initialized():
                raise RuntimeError("Model parallel groups must be initialized when use_cfg_parallel=True")

            cfg_group = get_cfg_group()
            cfg_rank = cfg_group.rank_in_group

        latents = latents.contiguous()
        noise_pred = torch.zeros_like(latents)
        neg_noise_pred = torch.zeros_like(latents)

        if not (use_cfg_parallel and cfg_rank != 0):
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

        if not use_cfg_parallel or cfg_rank != 0:
            with set_forward_context(attn_metadata=attn_metadata):
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

        if use_cfg_parallel:
            noise_pred = cfg_group.all_reduce(noise_pred)
            neg_noise_pred = cfg_group.all_reduce(neg_noise_pred)
        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
        original_dtype = comb_pred.dtype
        cond_norm = torch.norm(noise_pred.float(), dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred.float(), dim=-1, keepdim=True)
        noise_pred = (comb_pred.float() * (cond_norm / noise_norm)).to(original_dtype)
        return noise_pred

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale for classifier-free guidance.
            height (`int`, *optional*):
                The height in pixels of the generated image.
            width (`int`, *optional*):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            sigmas (`List[float]`, *optional*):
                Custom sigmas for the denoising process.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random generator(s) for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for negative prompt embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format ("pil", "np", or "latent").
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a QwenImagePipelineOutput.
            attention_kwargs (`dict`, *optional*):
                Additional attention kwargs.
            callback_on_step_end (`Callable`, *optional*):
                Callback function called at each step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                Tensor inputs for the callback.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length for prompts.

        Returns:
            [`QwenImagePipelineOutput`] or `tuple`:
            Generated images.
        """
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Check multi-image batch size constraint
        is_multi_image = isinstance(image, list)
        if is_multi_image and batch_size > 1:
            raise ValueError(
                f"QwenImageEditPlusPipeline currently only supports batch_size=1 for multi-image input, "
                f"but received batch_size={batch_size}. Please process prompts one at a time."
            )

        device = self.device

        # 3. Preprocess image
        condition_images = None
        vae_images = None
        vae_image_sizes = None

        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_images = []
            vae_images = []
            vae_image_sizes = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
                vae_image_sizes.append((vae_width, vae_height))

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                attn_metadata = self._build_attn_metadata(self.pipeline_config.attn_params)

                noise_pred = self._predict_noise_with_cfg(
                    latents=latents,
                    image_latents=image_latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask if do_true_cfg else None,
                    img_shapes=img_shapes,
                    attn_metadata=attn_metadata,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                )

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
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
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
