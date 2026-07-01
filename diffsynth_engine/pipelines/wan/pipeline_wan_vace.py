# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py

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
import json
import os
from typing import TYPE_CHECKING, Any, Callable

import PIL.Image
import regex as re
import torch
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel

from diffsynth_engine.configs.wan import WanPipelineConfig
from diffsynth_engine.distributed.parallel_state import (
    get_cfg_group,
    is_cfg_group_initialized,
)
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.models.wan import AutoencoderKLWan, WanVACETransformer3DModel
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


class WanVACEPipeline(Pipeline):
    r"""
    Pipeline for controllable generation using Wan.

    Args:
        pipeline_config (`WanPipelineConfig`):
            Configuration for the pipeline.
        tokenizer (`AutoTokenizer`):
            Tokenizer from T5, specifically the google/umt5-xxl variant.
        text_encoder (`UMT5EncoderModel`):
            T5 text encoder, specifically the google/umt5-xxl variant.
        vae (`AutoencoderKLWan`):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        scheduler (`UniPCMultistepScheduler`):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        transformer (`WanVACETransformer3DModel`, *optional*):
            Conditional Transformer to denoise the input latents.
        transformer_2 (`WanVACETransformer3DModel`, *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. If provided, enables
            two-stage denoising where `transformer` handles high-noise stages and `transformer_2` handles low-noise
            stages. If not provided, only `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
            The actual boundary timestep is calculated as `boundary_ratio * num_train_timesteps`. When provided,
            `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
    """

    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        pipeline_config: WanPipelineConfig,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        transformer: WanVACETransformer3DModel = None,
        transformer_2: WanVACETransformer3DModel = None,
        boundary_ratio: float | None = None,
    ):
        super().__init__(pipeline_config)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.scheduler = scheduler
        self.boundary_ratio = boundary_ratio

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if self.vae is not None else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if self.vae is not None else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        active_transformer = transformer if transformer is not None else transformer_2
        head_dim = active_transformer.config.attention_head_dim
        self.attn_backend = get_attn_backend(pipeline_config.attn_type)
        if not self.attn_backend.supports_head_size(head_dim):
            raise ValueError(f"Attention backend {pipeline_config.attn_type!r} does not support head size {head_dim}.")

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | WanPipelineConfig):
        """
        Load a WanVACEPipeline from a pretrained model path or config.

        Args:
            model_path_or_config: Either a string path to the model directory or a WanPipelineConfig instance.

        Returns:
            WanVACEPipeline: The loaded pipeline.
        """
        if isinstance(model_path_or_config, str):
            pipeline_config = WanPipelineConfig(model_path=model_path_or_config)
        else:
            pipeline_config = model_path_or_config

        if not os.path.exists(pipeline_config.model_path):
            raise FileNotFoundError(f"Model path not found: {pipeline_config.model_path}")

        model_index_path = os.path.join(pipeline_config.model_path, "model_index.json")
        model_index = {}
        boundary_ratio = None
        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
            boundary_ratio = model_index.get("boundary_ratio", None)
            if boundary_ratio is not None:
                logger.info(f"Loaded boundary_ratio={boundary_ratio} from model_index.json")

        # Load transformer
        transformer = cls.init_transformer(WanVACETransformer3DModel, pipeline_config)

        # Load transformer_2
        transformer_2 = None
        if "transformer_2" in model_index and model_index["transformer_2"] is not None:
            transformer_2_subfolder = "transformer_2"
            if os.path.isdir(os.path.join(pipeline_config.model_path, transformer_2_subfolder)):
                transformer_2 = cls.init_transformer(
                    WanVACETransformer3DModel, pipeline_config, subfolder=transformer_2_subfolder
                )
                logger.info(
                    f"Loaded transformer_2 from `{transformer_2_subfolder}` subfolder of {pipeline_config.model_path}."
                )
            else:
                logger.warning(
                    f"transformer_2 declared in model_index.json but subfolder "
                    f"'{transformer_2_subfolder}' not found in {pipeline_config.model_path}. Skipping."
                )

        # Load scheduler
        scheduler = UniPCMultistepScheduler.from_pretrained(pipeline_config.model_path, subfolder="scheduler")

        # Load VAE
        vae = cls.init_vae(AutoencoderKLWan, pipeline_config)

        # Load text encoder
        text_encoder = cls.init_text_encoder(UMT5EncoderModel, pipeline_config, strict=False)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pipeline_config.model_path, subfolder="tokenizer")

        return cls(
            pipeline_config=pipeline_config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
            boundary_ratio=boundary_ratio,
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
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

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

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
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        video=None,
        mask=None,
        reference_images=None,
        guidance_scale_2=None,
    ):
        if self.transformer is not None:
            base = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        elif self.transformer_2 is not None:
            base = self.vae_scale_factor_spatial * self.transformer_2.config.patch_size[1]
        else:
            raise ValueError(
                "`transformer` or `transformer_2` component must be set in order to run inference with this pipeline"
            )

        if height % base != 0 or width % base != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {base} but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if self.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: "
                f"{negative_prompt_embeds}. Please make sure to only forward one of the two."
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

        if video is not None:
            if mask is not None:
                if len(video) != len(mask):
                    raise ValueError(
                        f"Length of `video` {len(video)} and `mask` {len(mask)} do not match. Please make sure that"
                        " they have the same length."
                    )
            if reference_images is not None:
                is_pil_image = isinstance(reference_images, PIL.Image.Image)
                is_list_of_pil_images = isinstance(reference_images, list) and all(
                    isinstance(ref_img, PIL.Image.Image) for ref_img in reference_images
                )
                is_list_of_list_of_pil_images = isinstance(reference_images, list) and all(
                    isinstance(ref_img, list) and all(isinstance(r, PIL.Image.Image) for r in ref_img)
                    for ref_img in reference_images
                )
                if not (is_pil_image or is_list_of_pil_images or is_list_of_list_of_pil_images):
                    raise ValueError(
                        "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
                        f"`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
                    )
                if is_list_of_list_of_pil_images and len(reference_images) != 1:
                    raise ValueError(
                        "The pipeline only supports generating one video at a time at the moment. When passing a list "
                        "of list of reference images, where the outer list corresponds to the batch size and the inner "
                        "list corresponds to list of conditioning images per video, please make sure to only pass "
                        "one inner list of reference images (i.e., `[[<image1>, <image2>, ...]]`"
                    )
        elif mask is not None:
            raise ValueError("`mask` can only be passed if `video` is passed as well.")

    def preprocess_conditions(
        self,
        video: list[PipelineImageInput] | None = None,
        mask: list[PipelineImageInput] | None = None,
        reference_images: PIL.Image.Image | list[PIL.Image.Image] | list[list[PIL.Image.Image]] | None = None,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        if video is not None:
            base = self.vae_scale_factor_spatial * (
                self.transformer.config.patch_size[1]
                if self.transformer is not None
                else self.transformer_2.config.patch_size[1]
            )
            video_height, video_width = self.video_processor.get_default_height_width(video[0])

            if video_height * video_width > height * width:
                scale = min(width / video_width, height / video_height)
                video_height, video_width = int(video_height * scale), int(video_width * scale)

            if video_height % base != 0 or video_width % base != 0:
                logger.warning(
                    f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}."
                )
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base

            assert video_height * video_width <= height * width

            video = self.video_processor.preprocess_video(video, video_height, video_width)
            image_size = (video_height, video_width)
        else:
            video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
            image_size = (height, width)

        if mask is not None:
            mask = self.video_processor.preprocess_video(mask, image_size[0], image_size[1])
            mask = torch.clamp((mask + 1) / 2, min=0, max=1)
        else:
            mask = torch.ones_like(video)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        # Make a list of list of images where the outer list corresponds to video batch size and the inner list
        # corresponds to list of conditioning images per video
        if reference_images is None or isinstance(reference_images, PIL.Image.Image):
            reference_images = [[reference_images] for _ in range(video.shape[0])]
        elif isinstance(reference_images, (list, tuple)) and isinstance(next(iter(reference_images)), PIL.Image.Image):
            reference_images = [reference_images]
        elif (
            isinstance(reference_images, (list, tuple))
            and isinstance(next(iter(reference_images)), list)
            and isinstance(next(iter(reference_images[0])), PIL.Image.Image)
        ):
            reference_images = reference_images
        else:
            raise ValueError(
                "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
                f"`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
            )

        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` "
                f"{len(reference_images)} does not match."
            )

        ref_images_lengths = [len(batch) for batch in reference_images]
        if any(length != ref_images_lengths[0] for length in ref_images_lengths):
            raise ValueError(
                f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}."
            )

        reference_images_preprocessed = []
        for reference_images_batch in reference_images:
            preprocessed_images = []
            for image in reference_images_batch:
                if image is None:
                    continue
                image = self.video_processor.preprocess(image, None, None)
                img_height, img_width = image.shape[-2:]
                scale = min(image_size[0] / img_height, image_size[1] / img_width)
                new_height, new_width = int(img_height * scale), int(img_width * scale)
                resized_image = torch.nn.functional.interpolate(
                    image, size=(new_height, new_width), mode="bilinear", align_corners=False
                ).squeeze(0)  # [C, H, W]
                top = (image_size[0] - new_height) // 2
                left = (image_size[1] - new_width) // 2
                canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
                canvas[:, top : top + new_height, left : left + new_width] = resized_image
                preprocessed_images.append(canvas)
            reference_images_preprocessed.append(preprocessed_images)

        return video, mask, reference_images_preprocessed

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: list[list[torch.Tensor]] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self.device

        if isinstance(generator, list):
            raise ValueError("Passing a list of generators is not yet supported.")

        if reference_images is None:
            reference_images = [[None] for _ in range(video.shape[0])]
        else:
            if video.shape[0] != len(reference_images):
                raise ValueError(
                    f"Batch size of `video` {video.shape[0]} and length of `reference_images` "
                    f"{len(reference_images)} does not match."
                )

        if video.shape[0] != 1:
            raise ValueError("Generating with more than one video is not yet supported.")

        vae_dtype = self.pipeline_config.vae_dtype
        video = video.to(dtype=vae_dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        if mask is None:
            latents = retrieve_latents(self.vae.encode(video), generator, sample_mode="argmax").unbind(0)
            latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
        else:
            mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
            inactive = video * (1 - mask)
            reactive = video * mask
            inactive = retrieve_latents(self.vae.encode(inactive), generator, sample_mode="argmax")
            reactive = retrieve_latents(self.vae.encode(reactive), generator, sample_mode="argmax")
            inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
            reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
            latents = torch.cat([inactive, reactive], dim=1)

        latent_list = []
        for latent, reference_images_batch in zip(latents, reference_images):
            for reference_image in reference_images_batch:
                assert reference_image.ndim == 3
                reference_image = reference_image.to(dtype=vae_dtype)
                reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
                reference_latent = retrieve_latents(self.vae.encode(reference_image), generator, sample_mode="argmax")
                reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
                reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
            latent_list.append(latent)
        return torch.stack(latent_list)

    def prepare_masks(
        self,
        mask: torch.Tensor,
        reference_images: list[list[torch.Tensor]] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        if isinstance(generator, list):
            raise ValueError("Passing a list of generators is not yet supported.")

        if reference_images is None:
            reference_images = [[None] for _ in range(mask.shape[0])]
        else:
            if mask.shape[0] != len(reference_images):
                raise ValueError(
                    f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` "
                    f"{len(reference_images)} does not match."
                )

        if mask.shape[0] != 1:
            raise ValueError("Generating with more than one video is not yet supported.")

        transformer_patch_size = (
            self.transformer.config.patch_size[1]
            if self.transformer is not None
            else self.transformer_2.config.patch_size[1]
        )

        mask_list = []
        for mask_, reference_images_batch in zip(mask, reference_images):
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (num_frames + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
            new_height = height // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            new_width = width // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                num_frames, new_height, self.vae_scale_factor_spatial, new_width, self.vae_scale_factor_spatial
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
            mask_ = torch.nn.functional.interpolate(
                mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
            ).squeeze(0)
            num_ref_images = len(reference_images_batch)
            if num_ref_images > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
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
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        control_hidden_states: torch.Tensor,
        control_hidden_states_scale: torch.Tensor,
        attn_metadata,
        apply_cfg: bool,
        guidance_scale: float,
        use_cfg_parallel: bool,
        model: WanVACETransformer3DModel | None = None,
    ):
        """
        Predict noise with classifier-free guidance, supporting parallel CFG inference.

        Args:
            latent_model_input: The model input latents.
            timestep: Current timestep tensor.
            prompt_embeds: Positive prompt embeddings tensor.
            negative_prompt_embeds: Negative prompt embeddings tensor.
            control_hidden_states: VACE conditioning latents.
            control_hidden_states_scale: Per-layer scale for VACE conditioning.
            attn_metadata: Attention metadata for set_forward_context.
            apply_cfg: Whether to apply classifier-free guidance this step.
            guidance_scale: The CFG scale factor.
            use_cfg_parallel: Whether to use CFG parallelism across devices.
            model: The transformer model to use. If None, defaults to self.transformer.

        Returns:
            noise_pred: The predicted noise tensor.
        """
        if model is None:
            model = self.transformer

        if not apply_cfg:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred = model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    control_hidden_states=control_hidden_states,
                    control_hidden_states_scale=control_hidden_states_scale,
                    return_dict=False,
                )[0]
            return noise_pred.float()

        # CFG mode
        cfg_group, cfg_rank = None, None
        if use_cfg_parallel:
            if not is_cfg_group_initialized():
                raise RuntimeError("CFG group must be initialized when use_cfg_parallel=True")
            cfg_group = get_cfg_group()
            cfg_rank = cfg_group.rank_in_group

        noise_pred_pos = torch.zeros_like(latent_model_input, dtype=torch.float32)
        noise_pred_neg = torch.zeros_like(latent_model_input, dtype=torch.float32)

        # Positive prompt forward pass
        if not (use_cfg_parallel and cfg_rank != 0):
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_pos = model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    control_hidden_states=control_hidden_states,
                    control_hidden_states_scale=control_hidden_states_scale,
                    return_dict=False,
                )[0].float()

        # Negative prompt forward pass
        if not use_cfg_parallel or cfg_rank != 0:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_neg = model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    control_hidden_states=control_hidden_states,
                    control_hidden_states_scale=control_hidden_states_scale,
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
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        video: list[PipelineImageInput] | None = None,
        mask: list[PipelineImageInput] | None = None,
        reference_images: list[PipelineImageInput] | None = None,
        conditioning_scale: float | list[float] | torch.Tensor = 1.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
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
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to avoid during video generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            video (`list[PIL.Image.Image]`, *optional*):
                The input video or videos to be used as a starting point for the generation. The video should be a list
                of PIL images, a numpy array, or a torch tensor. Currently, the pipeline only supports generating one
                video at a time.
            mask (`list[PIL.Image.Image]`, *optional*):
                The input mask defines which video regions to condition on and which to generate. Black areas in the
                mask indicate conditioning regions, while white areas indicate regions for generation. The mask should
                be a list of PIL images, a numpy array, or a torch tensor. Currently supports generating a single video
                at a time.
            reference_images (`list[PIL.Image.Image]`, *optional*):
                A list of one or more reference images as extra conditioning for the generation. For example, if you
                are trying to inpaint a video to change the character, you can pass reference images of the new
                character here. Refer to the Diffusers [examples](https://github.com/huggingface/diffusers/pull/11582)
                and original [user
                guide](https://github.com/ali-vilab/VACE/blob/0897c6d055d7d9ea9e191dce763006664d9780f8/UserGuide.md)
                for a full list of supported tasks and use cases.
            conditioning_scale (`float`, `list[float]`, `torch.Tensor`, defaults to `1.0`):
                The conditioning scale to be applied when adding the control conditioning latent stream to the
                denoising latent stream in each control layer of the model. If a float is provided, it will be applied
                uniformly to all layers. If a list or tensor is provided, it should have the same length as the number
                of control layers in the model (`len(transformer.config.vace_layers)`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated video.
            width (`int`, defaults to `832`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
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
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            video,
            mask,
            reference_images,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
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

        transformer_dtype = self.pipeline_config.model_dtype

        vace_layers = (
            self.transformer.config.vace_layers
            if self.transformer is not None
            else self.transformer_2.config.vace_layers
        )
        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match "
                    f"number of layers {len(vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match "
                    f"number of layers {len(vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(device=device, dtype=transformer_dtype)

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

        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        video, mask, reference_images = self.preprocess_conditions(
            video,
            mask,
            reference_images,
            batch_size,
            height,
            width,
            num_frames,
            torch.float32,
            device,
        )
        num_reference_images = len(reference_images[0])

        conditioning_latents = self.prepare_video_latents(video, mask, reference_images, generator, device)
        mask = self.prepare_masks(mask, reference_images, generator)
        conditioning_latents = torch.cat([conditioning_latents, mask], dim=1)
        conditioning_latents = conditioning_latents.to(transformer_dtype)

        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames + num_reference_images * self.vae_scale_factor_temporal,
            torch.float32,
            device,
            generator,
            latents,
        )

        if conditioning_latents.shape[2] != latents.shape[2]:
            logger.warning(
                "The number of frames in the conditioning latents does not match the number of frames "
                "to be generated. Generation quality may be affected."
            )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.boundary_ratio is not None:
            boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                attn_metadata = self._build_attn_metadata(self.pipeline_config.attn_params)

                noise_pred = self._predict_noise_with_cfg(
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    control_hidden_states=conditioning_latents,
                    control_hidden_states_scale=conditioning_scale,
                    attn_metadata=attn_metadata,
                    apply_cfg=self.do_classifier_free_guidance,
                    guidance_scale=current_guidance_scale,
                    use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                    model=current_model,
                )

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

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents[:, :, num_reference_images:]
            latents = latents.to(self.pipeline_config.vae_dtype)
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
