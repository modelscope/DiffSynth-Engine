# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_animate.py

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

from __future__ import annotations

import html
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import PIL
import regex as re
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers.pipelines.wan.image_processor import WanAnimateImageProcessor
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffsynth_engine.configs.wan import WanPipelineConfig
from diffsynth_engine.distributed.parallel_state import (
    get_cfg_group,
    is_cfg_group_initialized,
)
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.models.wan import AutoencoderKLWan, WanAnimateTransformer3DModel
from diffsynth_engine.pipelines.base import Pipeline
from diffsynth_engine.registry import get_attn_backend
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from diffusers.image_processor import PipelineImageInput


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


def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WanAnimatePipeline(Pipeline):
    r"""
    Pipeline for unified character animation and replacement using Wan-Animate.

    WanAnimatePipeline takes a character image, pose video, and face video as input, and generates a video in two
    modes:

    1. **Animation mode**: The model generates a video of the character image that mimics the human motion in the input
       pose and face videos. The character is animated based on the provided motion controls, creating a new animated
       video of the character.

    2. **Replacement mode**: The model replaces a character in a background video with the provided character image,
       using the pose and face videos for motion control. This mode requires additional `background_video` and
       `mask_video` inputs. The mask video should have black regions where the original content should be preserved and
       white regions where the new character should be generated.

    Args:
        pipeline_config (`WanPipelineConfig`):
            Configuration for the pipeline.
        tokenizer (`AutoTokenizer`):
            Tokenizer from T5, specifically the google/umt5-xxl variant.
        text_encoder (`UMT5EncoderModel`):
            T5 text encoder, specifically the google/umt5-xxl variant.
        image_encoder (`CLIPVisionModel`):
            CLIP vision model for encoding input images.
        image_processor (`CLIPImageProcessor`):
            CLIP image processor for preprocessing input images.
        vae (`AutoencoderKLWan`):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        scheduler (`UniPCMultistepScheduler`):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        transformer (`WanAnimateTransformer3DModel`):
            Conditional Transformer to denoise the input latents.
    """

    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        pipeline_config: WanPipelineConfig,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        image_processor: CLIPImageProcessor,
        image_encoder: CLIPVisionModel,
        transformer: WanAnimateTransformer3DModel,
    ):
        super().__init__(pipeline_config)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.image_encoder = image_encoder
        self.transformer = transformer
        self.scheduler = scheduler
        self.image_processor = image_processor

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if self.vae is not None else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if self.vae is not None else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.video_processor_for_mask = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, do_normalize=False, do_convert_grayscale=True
        )

        spatial_patch_size = self.transformer.config.patch_size[-2:] if self.transformer is not None else (2, 2)
        self.vae_image_processor = WanAnimateImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            spatial_patch_size=spatial_patch_size,
            resample="bilinear",
            fill_color=0,
        )

        head_dim = self.transformer.config.attention_head_dim
        self.attn_backend = get_attn_backend(pipeline_config.attn_type)
        if not self.attn_backend.supports_head_size(head_dim):
            raise ValueError(f"Attention backend {pipeline_config.attn_type!r} does not support head size {head_dim}.")

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | WanPipelineConfig):
        """
        Load a WanAnimatePipeline from a pretrained model path or config.

        Args:
            model_path_or_config: Either a string path to the model directory or a WanPipelineConfig instance.

        Returns:
            WanAnimatePipeline: The loaded pipeline.
        """
        if isinstance(model_path_or_config, str):
            pipeline_config = WanPipelineConfig(model_path=model_path_or_config)
        else:
            pipeline_config = model_path_or_config

        if not os.path.exists(pipeline_config.model_path):
            raise FileNotFoundError(f"Model path not found: {pipeline_config.model_path}")

        # Load transformer
        transformer = cls.init_transformer(WanAnimateTransformer3DModel, pipeline_config)
        transformer.eval()

        # Load scheduler
        scheduler_kwargs = {}
        if pipeline_config.flow_shift is not None:
            scheduler_kwargs["flow_shift"] = pipeline_config.flow_shift
        scheduler = UniPCMultistepScheduler.from_pretrained(
            pipeline_config.model_path,
            subfolder="scheduler",
            **scheduler_kwargs,
        )

        # Load VAE
        vae = cls.init_vae(AutoencoderKLWan, pipeline_config)
        vae.eval()

        # Load text encoder
        text_encoder = cls.init_text_encoder(UMT5EncoderModel, pipeline_config, strict=False)
        text_encoder.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pipeline_config.model_path,
            subfolder="tokenizer",
        )

        # Load image encoder
        image_encoder = cls.init_image_encoder(pipeline_config)

        # Load image processor
        image_processor = None
        image_processor_path = os.path.join(pipeline_config.model_path, "image_processor")
        if os.path.isdir(image_processor_path):
            image_processor = CLIPImageProcessor.from_pretrained(
                pipeline_config.model_path,
                subfolder="image_processor",
            )
            logger.info("Loaded image_processor from `image_processor` subfolder.")

        return cls(
            pipeline_config=pipeline_config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
            scheduler=scheduler,
        )

    @staticmethod
    def init_image_encoder(pipeline_config: WanPipelineConfig, empty_weights: bool = False):
        logger.info("Initializing image encoder...")
        image_encoder_path = os.path.join(pipeline_config.model_path, "image_encoder")
        if not os.path.isdir(image_encoder_path):
            logger.warning(f"image_encoder not found in {pipeline_config.model_path}.")
            return None

        if empty_weights:
            with init_empty_weights():
                model = CLIPVisionModel.from_pretrained(
                    pipeline_config.model_path,
                    subfolder="image_encoder",
                    local_files_only=True,
                )
            return model

        model = CLIPVisionModel.from_pretrained(
            pipeline_config.model_path,
            subfolder="image_encoder",
            dtype=torch.float32,
        )
        model.to(device=pipeline_config.device)
        return model

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.pipeline_config.text_encoder_dtype

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

        # Duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(self, image: PipelineImageInput, device: torch.device | None = None):
        device = device or self.device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
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
        image,
        pose_video,
        face_video,
        background_video,
        mask_video,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        mode=None,
        prev_segment_conditioning_frames=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if pose_video is None:
            raise ValueError("Provide `pose_video`. Cannot leave `pose_video` undefined.")
        if face_video is None:
            raise ValueError("Provide `face_video`. Cannot leave `face_video` undefined.")
        if not isinstance(pose_video, list) or not isinstance(face_video, list):
            raise ValueError("`pose_video` and `face_video` must be lists of PIL images.")
        if len(pose_video) == 0 or len(face_video) == 0:
            raise ValueError("`pose_video` and `face_video` must contain at least one frame.")
        if mode == "replace" and (background_video is None or mask_video is None):
            raise ValueError(
                "Provide `background_video` and `mask_video`. Cannot leave both `background_video` and `mask_video`"
                " undefined when mode is `replace`."
            )
        if mode == "replace" and (not isinstance(background_video, list) or not isinstance(mask_video, list)):
            raise ValueError("`background_video` and `mask_video` must be lists of PIL images when mode is `replace`.")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found"
                f" {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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

        if mode is not None and (not isinstance(mode, str) or mode not in ("animate", "replace")):
            raise ValueError(
                f"`mode` has to be of type `str` and in ('animate', 'replace') but its type is {type(mode)} and value is {mode}"
            )

        if prev_segment_conditioning_frames is not None and (
            not isinstance(prev_segment_conditioning_frames, int) or prev_segment_conditioning_frames not in (1, 5)
        ):
            raise ValueError(
                f"`prev_segment_conditioning_frames` has to be of type `int` and 1 or 5 but its type is"
                f" {type(prev_segment_conditioning_frames)} and value is {prev_segment_conditioning_frames}"
            )

    def get_i2v_mask(
        self,
        batch_size: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        mask_len: int = 1,
        mask_pixel_values: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        # mask_pixel_values shape (if supplied): [B, C = 1, T, latent_h, latent_w]
        if mask_pixel_values is None:
            mask_lat_size = torch.zeros(
                batch_size, 1, (latent_t - 1) * 4 + 1, latent_h, latent_w, dtype=dtype, device=device
            )
        else:
            mask_lat_size = mask_pixel_values.clone().to(device=device, dtype=dtype)
        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_h, latent_w
        ).transpose(1, 2)

        return mask_lat_size

    def prepare_reference_image_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        sample_mode: str = "argmax",
        generator: torch.Generator | list[torch.Generator] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        # image shape: (B, C, H, W) or (B, C, T, H, W)
        dtype = dtype or self.pipeline_config.vae_dtype
        if image.ndim == 4:
            image = image.unsqueeze(2)

        _, _, _, height, width = image.shape
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        image = image.to(device=device, dtype=dtype)
        if isinstance(generator, list):
            ref_image_latents = [
                retrieve_latents(self.vae.encode(image), generator=g, sample_mode=sample_mode) for g in generator
            ]
            ref_image_latents = torch.cat(ref_image_latents)
        else:
            ref_image_latents = retrieve_latents(self.vae.encode(image), generator, sample_mode)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(ref_image_latents.device, ref_image_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            ref_image_latents.device, ref_image_latents.dtype
        )
        ref_image_latents = (ref_image_latents - latents_mean) * latents_recip_std

        if ref_image_latents.shape[0] == 1 and batch_size > 1:
            ref_image_latents = ref_image_latents.expand(batch_size, -1, -1, -1, -1)

        reference_image_mask = self.get_i2v_mask(batch_size, 1, latent_height, latent_width, 1, None, dtype, device)
        reference_image_latents = torch.cat([reference_image_mask, ref_image_latents], dim=1)

        return reference_image_latents

    def prepare_prev_segment_cond_latents(
        self,
        prev_segment_cond_video: torch.Tensor | None = None,
        background_video: torch.Tensor | None = None,
        mask_video: torch.Tensor | None = None,
        batch_size: int = 1,
        segment_frame_length: int = 77,
        start_frame: int = 0,
        height: int = 720,
        width: int = 1280,
        prev_segment_cond_frames: int = 1,
        task: str = "animate",
        interpolation_mode: str = "bicubic",
        sample_mode: str = "argmax",
        generator: torch.Generator | list[torch.Generator] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        # prev_segment_cond_video shape: (B, C, T, H, W) in pixel space if supplied
        # background_video shape: (B, C, T, H, W) (same as prev_segment_cond_video shape)
        # mask_video shape: (B, 1, T, H, W) (same as prev_segment_cond_video, but with only 1 channel)
        dtype = dtype or self.pipeline_config.vae_dtype
        if prev_segment_cond_video is None:
            if task == "replace":
                prev_segment_cond_video = background_video[:, :, :prev_segment_cond_frames].to(dtype)
            else:
                cond_frames_shape = (batch_size, 3, prev_segment_cond_frames, height, width)
                prev_segment_cond_video = torch.zeros(cond_frames_shape, dtype=dtype, device=device)

        data_batch_size, channels, _, segment_height, segment_width = prev_segment_cond_video.shape
        num_latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        if segment_height != height or segment_width != width:
            logger.info(
                f"Interpolating prev segment cond video from ({segment_width}, {segment_height}) to ({width}, {height})"
            )
            prev_segment_cond_video = prev_segment_cond_video.transpose(1, 2).flatten(0, 1)
            prev_segment_cond_video = F.interpolate(
                prev_segment_cond_video, size=(height, width), mode=interpolation_mode
            )
            prev_segment_cond_video = prev_segment_cond_video.unflatten(0, (batch_size, -1)).transpose(1, 2)

        if task == "replace":
            remaining_segment = background_video[:, :, prev_segment_cond_frames:].to(dtype)
        else:
            remaining_segment_frames = segment_frame_length - prev_segment_cond_frames
            remaining_segment = torch.zeros(
                batch_size, channels, remaining_segment_frames, height, width, dtype=dtype, device=device
            )

        prev_segment_cond_video = prev_segment_cond_video.to(dtype=dtype)
        full_segment_cond_video = torch.cat([prev_segment_cond_video, remaining_segment], dim=2)

        if isinstance(generator, list):
            if data_batch_size == len(generator):
                prev_segment_cond_latents = [
                    retrieve_latents(self.vae.encode(full_segment_cond_video[i].unsqueeze(0)), g, sample_mode)
                    for i, g in enumerate(generator)
                ]
            elif data_batch_size == 1:
                prev_segment_cond_latents = [
                    retrieve_latents(self.vae.encode(full_segment_cond_video), g, sample_mode) for g in generator
                ]
            else:
                raise ValueError(
                    f"The batch size of the prev segment video should be either {len(generator)} or 1 but is"
                    f" {data_batch_size}"
                )
            prev_segment_cond_latents = torch.cat(prev_segment_cond_latents)
        else:
            prev_segment_cond_latents = retrieve_latents(
                self.vae.encode(full_segment_cond_video), generator, sample_mode
            )

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(prev_segment_cond_latents.device, prev_segment_cond_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            prev_segment_cond_latents.device, prev_segment_cond_latents.dtype
        )
        prev_segment_cond_latents = (prev_segment_cond_latents - latents_mean) * latents_recip_std

        if task == "replace":
            mask_video = 1 - mask_video
            mask_video = mask_video.permute(0, 2, 1, 3, 4)
            mask_video = mask_video.flatten(0, 1)
            mask_video = F.interpolate(mask_video, size=(latent_height, latent_width), mode="nearest")
            mask_pixel_values = mask_video.unflatten(0, (batch_size, -1))
            mask_pixel_values = mask_pixel_values.permute(0, 2, 1, 3, 4)  # output shape: [B, C = 1, T, H_lat, W_lat]
        else:
            mask_pixel_values = None
        prev_segment_cond_mask = self.get_i2v_mask(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            mask_len=prev_segment_cond_frames if start_frame > 0 else 0,
            mask_pixel_values=mask_pixel_values,
            dtype=dtype,
            device=device,
        )

        prev_segment_cond_latents = torch.cat([prev_segment_cond_mask, prev_segment_cond_latents], dim=1)
        return prev_segment_cond_latents

    def prepare_pose_latents(
        self,
        pose_video: torch.Tensor,
        batch_size: int = 1,
        sample_mode: str = "argmax",
        generator: torch.Generator | list[torch.Generator] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        # pose_video shape: (B, C, T, H, W)
        dtype = dtype if dtype is not None else self.pipeline_config.vae_dtype
        pose_video = pose_video.to(device=device, dtype=dtype)
        if isinstance(generator, list):
            pose_latents = [
                retrieve_latents(self.vae.encode(pose_video), generator=g, sample_mode=sample_mode) for g in generator
            ]
            pose_latents = torch.cat(pose_latents)
        else:
            pose_latents = retrieve_latents(self.vae.encode(pose_video), generator, sample_mode)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(pose_latents.device, pose_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            pose_latents.device, pose_latents.dtype
        )
        pose_latents = (pose_latents - latents_mean) * latents_recip_std
        if pose_latents.shape[0] == 1 and batch_size > 1:
            pose_latents = pose_latents.expand(batch_size, -1, -1, -1, -1)
        return pose_latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 77,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        # +1 for the conditioning frame
        shape = (batch_size, num_channels_latents, num_latent_frames + 1, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    def pad_video_frames(self, frames: list[Any], num_target_frames: int) -> list[Any]:
        """
        Pads an array-like video `frames` to `num_target_frames` using a "reflect"-like strategy. The frame dimension
        is assumed to be the first dimension. In the 1D case, we can visualize this strategy as follows:

        pad_video_frames([1, 2, 3, 4, 5], 10) -> [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
        """
        idx = 0
        flip = False
        target_frames = []
        while len(target_frames) < num_target_frames:
            target_frames.append(deepcopy(frames[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(frames) - 1:
                flip = not flip

        return target_frames

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

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
        reference_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        image_embeds: torch.Tensor | None,
        pose_latents: torch.Tensor,
        face_video_segment: torch.Tensor,
        motion_encode_batch_size: int | None,
        attn_metadata,
        apply_cfg: bool,
        guidance_scale: float,
        use_cfg_parallel: bool,
    ):
        """
        Predict noise with classifier-free guidance, supporting parallel CFG inference.

        For Wan Animate, the unconditional pass blanks out the face video (sets all pixels to -1)
        to remove face conditioning.

        Args:
            latents: Current noisy latents.
            reference_latents: Reference and previous-segment conditioning latents.
            timestep: Current timestep tensor.
            prompt_embeds: Positive prompt embeddings tensor.
            negative_prompt_embeds: Negative prompt embeddings tensor.
            image_embeds: Image embeddings tensor for cross-attention.
            pose_latents: Pose video latents.
            face_video_segment: Face video segment in pixel space.
            motion_encode_batch_size: Batch size for batched motion encoding.
            attn_metadata: Attention metadata for set_forward_context.
            apply_cfg: Whether to apply classifier-free guidance this step.
            guidance_scale: The CFG scale factor.
            use_cfg_parallel: Whether to use CFG parallelism across devices.

        Returns:
            noise_pred: The predicted noise tensor.
        """
        transformer_dtype = self.pipeline_config.model_dtype
        latent_model_input = torch.cat([latents, reference_latents], dim=1).to(transformer_dtype)

        if not apply_cfg:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    pose_hidden_states=pose_latents,
                    face_pixel_values=face_video_segment,
                    motion_encode_batch_size=motion_encode_batch_size,
                    return_dict=False,
                )[0]
            return noise_pred

        # CFG mode
        cfg_group, cfg_rank = None, None
        if use_cfg_parallel:
            if not is_cfg_group_initialized():
                raise RuntimeError("CFG group must be initialized when use_cfg_parallel=True")
            cfg_group = get_cfg_group()
            cfg_rank = cfg_group.rank_in_group

        noise_pred_pos = torch.zeros_like(latents, dtype=transformer_dtype)
        noise_pred_neg = torch.zeros_like(latents, dtype=transformer_dtype)

        # Positive prompt forward pass (conditional)
        if not (use_cfg_parallel and cfg_rank != 0):
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_pos = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    pose_hidden_states=pose_latents,
                    face_pixel_values=face_video_segment,
                    motion_encode_batch_size=motion_encode_batch_size,
                    return_dict=False,
                )[0]

        # Negative prompt forward pass (unconditional) - blank out face
        face_pixel_values_uncond = face_video_segment * 0 - 1
        if not use_cfg_parallel or cfg_rank != 0:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_neg = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    pose_hidden_states=pose_latents,
                    face_pixel_values=face_pixel_values_uncond,
                    motion_encode_batch_size=motion_encode_batch_size,
                    return_dict=False,
                )[0]

        # All-reduce for CFG parallel
        if use_cfg_parallel:
            noise_pred_pos = cfg_group.all_reduce(noise_pred_pos.float()).to(transformer_dtype)
            noise_pred_neg = cfg_group.all_reduce(noise_pred_neg.float()).to(transformer_dtype)

        # Apply CFG
        noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        pose_video: list[PIL.Image.Image],
        face_video: list[PIL.Image.Image],
        background_video: list[PIL.Image.Image] | None = None,
        mask_video: list[PIL.Image.Image] | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 720,
        width: int = 1280,
        segment_frame_length: int = 77,
        num_inference_steps: int = 20,
        mode: str = "animate",
        prev_segment_conditioning_frames: int = 1,
        motion_encode_batch_size: int | None = None,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input character image to condition the generation on. Must be an image, a list of images or a
                `torch.Tensor`.
            pose_video (`list[PIL.Image.Image]`):
                The input pose video to condition the generation on. Must be a list of PIL images.
            face_video (`list[PIL.Image.Image]`):
                The input face video to condition the generation on. Must be a list of PIL images.
            background_video (`list[PIL.Image.Image]`, *optional*):
                When mode is `"replace"`, the input background video to condition the generation on. Must be a list of
                PIL images.
            mask_video (`list[PIL.Image.Image]`, *optional*):
                When mode is `"replace"`, the input mask video to condition the generation on. Must be a list of PIL
                images.
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to avoid during video generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `720`):
                The height in pixels of the generated video.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated video.
            segment_frame_length (`int`, defaults to `77`):
                The number of frames in each generated video segment. The total frames of video generated will be equal
                to the number of frames in `pose_video`; we will generate the video in segments until we have hit this
                length. In general, should be 4N + 1, where N is a non-negative integer.
            num_inference_steps (`int`, defaults to `20`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            mode (`str`, defaults to `"animate"`):
                The mode of the generation. Choose between `"animate"` and `"replace"`.
            prev_segment_conditioning_frames (`int`, defaults to `1`):
                The number of frames from the previous video segment to be used for temporal guidance.
            motion_encode_batch_size (`int`, *optional*):
                The batch size for batched encoding of the face video via the motion encoder. This allows trading off
                inference speed for lower memory usage by setting a smaller batch size.
            guidance_scale (`float`, defaults to `1.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. By default, CFG is not used in Wan
                Animate inference.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `WanPipelineOutput` instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Attention kwargs dictionary.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during the inference with the following
                arguments: `callback_on_step_end(step: int, timestep: int, callback_kwargs: dict)`. `callback_kwargs`
                will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Returns:
            `WanPipelineOutput` or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            pose_video,
            face_video,
            background_video,
            mask_video,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            mode,
            prev_segment_conditioning_frames,
        )

        if segment_frame_length % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`segment_frame_length - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the"
                f" nearest number."
            )
            segment_frame_length = (
                segment_frame_length // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        segment_frame_length = max(segment_frame_length, 1)

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

        # Compute segment layout
        cond_video_frames = len(pose_video)
        effective_segment_length = segment_frame_length - prev_segment_conditioning_frames
        last_segment_frames = (cond_video_frames - prev_segment_conditioning_frames) % effective_segment_length
        if last_segment_frames == 0:
            num_padding_frames = 0
        else:
            num_padding_frames = effective_segment_length - last_segment_frames
        num_target_frames = cond_video_frames + num_padding_frames
        num_segments = num_target_frames // effective_segment_length

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

        transformer_dtype = self.pipeline_config.model_dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Preprocess and encode the reference (character) image
        image_height, image_width = self.vae_image_processor.get_default_height_width(image)
        if image_height != height or image_width != width:
            logger.warning(f"Reshaping reference image from ({image_width}, {image_height}) to ({width}, {height})")
        image_pixels = self.vae_image_processor.preprocess(image, height=height, width=width, resize_mode="fill").to(
            device, dtype=torch.float32
        )

        # Get CLIP features from the reference image
        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size * num_videos_per_prompt, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 5. Encode conditioning videos (pose, face)
        pose_video = self.pad_video_frames(pose_video, num_target_frames)
        face_video = self.pad_video_frames(face_video, num_target_frames)

        pose_video_width, pose_video_height = pose_video[0].size
        if pose_video_height != height or pose_video_width != width:
            logger.warning(
                f"Reshaping pose video from ({pose_video_width}, {pose_video_height}) to ({width}, {height})"
            )
        pose_video = self.video_processor.preprocess_video(pose_video, height=height, width=width).to(
            device, dtype=torch.float32
        )

        face_video_width, face_video_height = face_video[0].size
        expected_face_size = self.transformer.config.motion_encoder_size
        if face_video_width != expected_face_size or face_video_height != expected_face_size:
            logger.warning(
                f"Reshaping face video from ({face_video_width}, {face_video_height}) to ({expected_face_size},"
                f" {expected_face_size})"
            )
        face_video = self.video_processor.preprocess_video(
            face_video, height=expected_face_size, width=expected_face_size
        ).to(device, dtype=torch.float32)

        if mode == "replace":
            background_video = self.pad_video_frames(background_video, num_target_frames)
            mask_video = self.pad_video_frames(mask_video, num_target_frames)

            background_video = self.video_processor.preprocess_video(background_video, height=height, width=width).to(
                device, dtype=torch.float32
            )
            mask_video = self.video_processor_for_mask.preprocess_video(mask_video, height=height, width=width).to(
                device, dtype=torch.float32
            )

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables which stay constant for all inference segments
        num_channels_latents = self.vae.config.z_dim

        # Get VAE-encoded latents of the reference (character) image
        reference_image_latents = self.prepare_reference_image_latents(
            image_pixels, batch_size * num_videos_per_prompt, generator=generator, device=device
        )

        # 8. Loop over video inference segments
        start = 0
        end = segment_frame_length
        all_out_frames = []
        out_frames = None
        actual_batch_size = batch_size * num_videos_per_prompt

        for _ in range(num_segments):
            assert start + prev_segment_conditioning_frames < cond_video_frames

            # Sample noisy latents for the current inference segment
            latents = self.prepare_latents(
                actual_batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=segment_frame_length,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents if start == 0 else None,
            )

            pose_video_segment = pose_video[:, :, start:end]
            face_video_segment = face_video[:, :, start:end]

            face_video_segment = face_video_segment.expand(actual_batch_size, -1, -1, -1, -1)
            face_video_segment = face_video_segment.to(dtype=transformer_dtype)

            if start > 0:
                prev_segment_cond_video = out_frames[:, :, -prev_segment_conditioning_frames:].clone().detach()
            else:
                prev_segment_cond_video = None

            if mode == "replace":
                background_video_segment = background_video[:, :, start:end]
                mask_video_segment = mask_video[:, :, start:end]

                background_video_segment = background_video_segment.expand(actual_batch_size, -1, -1, -1, -1)
                mask_video_segment = mask_video_segment.expand(actual_batch_size, -1, -1, -1, -1)
            else:
                background_video_segment = None
                mask_video_segment = None

            pose_latents = self.prepare_pose_latents(
                pose_video_segment, actual_batch_size, generator=generator, device=device
            )
            pose_latents = pose_latents.to(dtype=transformer_dtype)

            prev_segment_cond_latents = self.prepare_prev_segment_cond_latents(
                prev_segment_cond_video,
                background_video=background_video_segment,
                mask_video=mask_video_segment,
                batch_size=actual_batch_size,
                segment_frame_length=segment_frame_length,
                start_frame=start,
                height=height,
                width=width,
                prev_segment_cond_frames=prev_segment_conditioning_frames,
                task=mode,
                generator=generator,
                device=device,
            )

            # Concatenate the reference latents in the frame dimension
            reference_latents = torch.cat([reference_image_latents, prev_segment_cond_latents], dim=2)

            # 8.1 Denoising loop
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
                        reference_latents=reference_latents,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image_embeds=image_embeds,
                        pose_latents=pose_latents,
                        face_video_segment=face_video_segment,
                        motion_encode_batch_size=motion_encode_batch_size,
                        attn_metadata=attn_metadata,
                        apply_cfg=self.do_classifier_free_guidance,
                        guidance_scale=guidance_scale,
                        use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                    )

                    # Compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            latents = latents.to(self.pipeline_config.vae_dtype)
            # Destandardize latents in preparation for Wan VAE decoding
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_recip_std + latents_mean
            # Skip the first latent frame (used for conditioning)
            out_frames = self.vae.decode(latents[:, :, 1:], return_dict=False)[0]

            if start > 0:
                out_frames = out_frames[:, :, prev_segment_conditioning_frames:]
            all_out_frames.append(out_frames)

            start += effective_segment_length
            end += effective_segment_length

            # Reset scheduler timesteps / state for next denoising loop
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        self._current_timestep = None
        assert start + prev_segment_conditioning_frames >= cond_video_frames

        if not output_type == "latent":
            video = torch.cat(all_out_frames, dim=2)[:, :, :cond_video_frames]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
