# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py

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
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from accelerate import init_empty_weights
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

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


class WanImageToVideoPipeline(Pipeline):
    r"""
    Pipeline for image-to-video generation using Wan, adapted for DiffSynth-Engine.

    Args:
        pipeline_config (`WanPipelineConfig`):
            Configuration for the pipeline.
        tokenizer (`AutoTokenizer`):
            Tokenizer from T5, specifically the google/umt5-xxl variant.
        text_encoder (`UMT5EncoderModel`):
            T5 text encoder, specifically the google/umt5-xxl variant.
        image_encoder (`CLIPVisionModel`, *optional*):
            CLIP vision model for encoding input images.
        image_processor (`CLIPImageProcessor`, *optional*):
            CLIP image processor for preprocessing input images.
        vae (`AutoencoderKLWan`):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        transformer (`WanTransformer3DModel`, *optional*):
            Conditional Transformer to denoise the input latents.
        transformer_2 (`WanTransformer3DModel`, *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. If provided, enables
            two-stage denoising where `transformer` handles high-noise stages and `transformer_2` handles low-noise
            stages. If not provided, only `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
        expand_timesteps (`bool`, defaults to `False`):
            Whether to expand timesteps for Wan2.2 ti2v models.
    """

    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        pipeline_config: WanPipelineConfig,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: Optional[CLIPImageProcessor] = None,
        image_encoder: Optional[CLIPVisionModel] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__(pipeline_config)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.scheduler = scheduler
        self.boundary_ratio = boundary_ratio
        self.expand_timesteps = expand_timesteps

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if self.vae is not None else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if self.vae is not None else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        active_transformer = transformer if transformer is not None else transformer_2
        head_dim = active_transformer.config.attention_head_dim
        self.attn_backend = get_attn_backend(
            head_size=head_dim,
            attn_type=pipeline_config.attn_type,
        )

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | WanPipelineConfig):
        """
        Load a WanImageToVideoPipeline from a pretrained model path or config.

        Args:
            model_path_or_config: Either a string path to the model directory or a WanPipelineConfig instance.

        Returns:
            WanImageToVideoPipeline: The loaded pipeline.
        """
        if isinstance(model_path_or_config, str):
            pipeline_config = WanPipelineConfig(model_path=model_path_or_config)
        else:
            pipeline_config = model_path_or_config

        if not os.path.exists(pipeline_config.model_path):
            raise FileNotFoundError(f"Model path not found: {pipeline_config.model_path}")

        # Load model_index.json to read pipeline-level config and component declarations.
        model_index_path = os.path.join(pipeline_config.model_path, "model_index.json")
        model_index = {}
        boundary_ratio = None
        expand_timesteps = False
        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
            boundary_ratio = model_index.get("boundary_ratio", None)
            expand_timesteps = model_index.get("expand_timesteps", False)
            if boundary_ratio is not None:
                logger.info(f"Loaded boundary_ratio={boundary_ratio} from model_index.json")
            if expand_timesteps:
                logger.info(f"Loaded expand_timesteps={expand_timesteps} from model_index.json")

        # Load transformer (subfolder defaults to "transformer")
        transformer = cls.init_transformer(pipeline_config)

        # Load transformer_2 if declared in model_index.json.
        transformer_2 = None
        if "transformer_2" in model_index and model_index["transformer_2"] is not None:
            transformer_2_subfolder = "transformer_2"
            if os.path.isdir(os.path.join(pipeline_config.model_path, transformer_2_subfolder)):
                transformer_2 = cls.init_transformer(pipeline_config, subfolder=transformer_2_subfolder)
                logger.info(
                    f"Loaded transformer_2 from `{transformer_2_subfolder}` subfolder of {pipeline_config.model_path}."
                )
            else:
                logger.warning(
                    f"transformer_2 declared in model_index.json but subfolder "
                    f"'{transformer_2_subfolder}' not found in {pipeline_config.model_path}. Skipping."
                )

        # Load scheduler - auto-detect scheduler class from config
        scheduler_config_path = os.path.join(pipeline_config.model_path, "scheduler", SCHEDULER_CONFIG_NAME)
        scheduler_cls = FlowMatchEulerDiscreteScheduler
        if os.path.exists(scheduler_config_path):
            with open(scheduler_config_path, "r") as f:
                scheduler_config_dict = json.load(f)
            class_name = scheduler_config_dict.get("_class_name", None)
            if class_name is not None:
                try:
                    from diffusers import schedulers as schedulers_module

                    scheduler_cls = getattr(schedulers_module, class_name)
                    logger.info(f"Using scheduler class from config: {class_name}")
                except AttributeError:
                    logger.warning(
                        f"Scheduler class '{class_name}' not found in diffusers.schedulers, "
                        f"falling back to FlowMatchEulerDiscreteScheduler"
                    )
        scheduler = scheduler_cls.from_pretrained(
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
            transformer_2=transformer_2,
            scheduler=scheduler,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )

    @staticmethod
    def init_transformer(
        pipeline_config: WanPipelineConfig, empty_weights: bool = False, subfolder: str = "transformer"
    ):
        logger.info(f"Initializing transformer from subfolder={subfolder}...")
        with set_forward_context(attn_type=pipeline_config.attn_type):
            if empty_weights:
                with init_empty_weights():
                    config_dict = WanTransformer3DModel.load_config(
                        pipeline_config.model_path,
                        subfolder=subfolder,
                        local_files_only=True,
                    )
                    model = WanTransformer3DModel.from_config(config_dict)
            else:
                model = WanTransformer3DModel.from_pretrained(
                    pipeline_config.model_path,
                    subfolder=subfolder,
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

        if "shared.weight" in state_dict and "encoder.embed_tokens.weight" not in state_dict:
            state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

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

    @staticmethod
    def init_image_encoder(pipeline_config: WanPipelineConfig, empty_weights: bool = False):
        logger.info("Initializing image encoder...")
        image_encoder_path = os.path.join(pipeline_config.model_path, "image_encoder")
        if not os.path.isdir(image_encoder_path):
            logger.warning(f"image_encoder subfolder not found in {pipeline_config.model_path}. Skipping.")
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
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image,
        device: Optional[torch.device] = None,
    ):
        device = device or self.device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

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
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
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

        if self.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def prepare_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if self.expand_timesteps:
            video_condition = image
        elif last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.pipeline_config.vae_dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        if self.expand_timesteps:
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

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
        image_embeds: Optional[torch.Tensor],
        attn_metadata,
        apply_cfg: bool,
        guidance_scale: float,
        use_cfg_parallel: bool,
        batch_size: int,
        model: Optional[WanTransformer3DModel] = None,
    ):
        """
        Predict noise with optional classifier-free guidance and CFG parallelism.

        Args:
            latent_model_input: The model input (latents or latents + condition).
            timestep: Current timestep tensor.
            prompt_embeds: Positive prompt embeddings tensor.
            negative_prompt_embeds: Negative prompt embeddings tensor.
            image_embeds: Image embeddings tensor for I2V cross-attention.
            attn_metadata: Attention metadata for set_forward_context.
            apply_cfg: Whether to apply classifier-free guidance this step.
            guidance_scale: The CFG scale factor.
            use_cfg_parallel: Whether to use CFG parallelism across devices.
            batch_size: The actual batch size.
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
                    encoder_hidden_states_image=image_embeds,
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

        noise_pred_pos = torch.zeros_like(latent_model_input, dtype=torch.float32)
        noise_pred_neg = torch.zeros_like(latent_model_input, dtype=torch.float32)

        # Positive prompt forward pass
        if not (use_cfg_parallel and cfg_rank != 0):
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_pos = model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                )[0].float()

        # Negative prompt forward pass
        if not use_cfg_parallel or cfg_rank != 0:
            with set_forward_context(attn_metadata=attn_metadata):
                noise_pred_neg = model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
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
        image,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
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
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
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
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage when `boundary_ratio` is set. If `None` and
                `boundary_ratio` is not None, uses the same value as `guidance_scale`.
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
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings.
            last_image (`torch.Tensor`, *optional*):
                Optional last frame image for video generation with start and end frames.
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
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        patch_size = (
            self.transformer.config.patch_size if self.transformer is not None else self.transformer_2.config.patch_size
        )
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

        # Encode image embedding
        transformer_dtype = self.pipeline_config.model_dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )
        if self.expand_timesteps:
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.boundary_ratio is not None:
            boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        actual_batch_size = batch_size * num_videos_per_prompt

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Determine current model and guidance scale based on boundary_ratio
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                if self.expand_timesteps:
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                attn_metadata = self._build_attn_metadata(self.pipeline_config.attn_params)

                noise_pred = self._predict_noise_with_cfg(
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image_embeds=image_embeds,
                    attn_metadata=attn_metadata,
                    apply_cfg=self.do_classifier_free_guidance,
                    guidance_scale=current_guidance_scale,
                    use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                    batch_size=actual_batch_size,
                    model=current_model,
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

        if self.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        if not output_type == "latent":
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
