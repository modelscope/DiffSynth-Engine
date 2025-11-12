import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms.functional import pil_to_tensor
from typing import Callable, List, Optional
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.configs import WanSpeech2VideoPipelineConfig, WanS2VStateDicts
from diffsynth_engine.models.wan.wan_s2v_dit import WanS2VDiT
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_audio_encoder import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    get_audio_embed_bucket_fps,
    extract_audio_feat,
)
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.pipelines.wan_video import WanVideoPipeline
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.image import resize_and_center_crop
from diffsynth_engine.utils.video import read_n_frames
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging


logger = logging.get_logger(__name__)


def get_face_mask(
    ref_image: Image.Image,
    speaker_end_sec: List[List[int]],
    speaker_bbox: List[List[int]],
    num_frames_total: int,
    fps=16,
    temporal_scale=4,
    spatial_scale=16,
    dtype=torch.bfloat16,
):
    mask_height, mask_width = ref_image.height, ref_image.width
    ref_image = np.array(ref_image)
    face_mask = torch.zeros(
        [1, num_frames_total, mask_height, mask_width],
        dtype=dtype,
    )
    prev_time = 0
    for speaker_id, end_time in speaker_end_sec:
        start_frame = int(prev_time * fps)
        end_frame = int(end_time * fps)
        mask = torch.zeros(ref_image.shape[:2], dtype=dtype)
        for id, bbox in enumerate(speaker_bbox):
            if id == speaker_id:
                continue
            x_min, y_min, x_max, y_max = bbox
            y_min = max(0, y_min)
            y_max = min(mask.shape[0], y_max)
            mask[y_min:y_max, x_min:x_max] = 1
        face_mask[0, start_frame:end_frame] = mask[None]
        prev_time = end_time
        if end_frame > num_frames_total:
            break

    face_mask_resized = F.interpolate(
        face_mask[None],
        size=(
            num_frames_total // temporal_scale,
            mask_height // spatial_scale,
            mask_width // spatial_scale,
        ),
        mode="nearest",
    )[0]
    return 1 - face_mask_resized


def transform_bbox(
    bboxes: List[List[int]],
    original_height: int,
    original_width: int,
    target_height: int,
    target_width: int,
) -> Optional[List[float]]:
    transformed_bboxes = []
    for x_min, y_min, x_max, y_max in bboxes:
        # --- 1. The Resize Operation ---
        # The image is resized so its smaller edge is min(target_h, target_w).
        resize_size = min(target_height, target_width)

        # Determine the scaling factor.
        if original_width < original_height:
            # If width is the smaller edge
            scale_factor = resize_size / original_width
            resized_w = resize_size
            resized_h = int(original_height * scale_factor)
        else:
            # If height is the smaller edge or they are equal
            scale_factor = resize_size / original_height
            resized_h = resize_size
            resized_w = int(original_width * scale_factor)

        # Apply the scaling factor to the bbox coordinates.
        scaled_x_min = x_min * scale_factor
        scaled_y_min = y_min * scale_factor
        scaled_x_max = x_max * scale_factor
        scaled_y_max = y_max * scale_factor

        # --- 2. The Center Crop Operation ---
        # Calculate the top-left corner (offset) of the crop area.
        crop_offset_x = (resized_w - target_width) / 2.0
        crop_offset_y = (resized_h - target_height) / 2.0

        # Translate the bbox coordinates by subtracting the crop offset.
        # The new coordinate system's origin (0,0) is the top-left of the crop.
        final_x_min = scaled_x_min - crop_offset_x
        final_y_min = scaled_y_min - crop_offset_y
        final_x_max = scaled_x_max - crop_offset_x
        final_y_max = scaled_y_max - crop_offset_y

        # --- 3. Clipping ---
        # The bbox might now be partially or fully outside the crop.
        # Clip the coordinates to the crop dimensions [0, target_w] and [0, target_h].
        final_x_min = max(0, final_x_min)
        final_y_min = max(0, final_y_min)
        final_x_max = min(target_width, final_x_max)
        final_y_max = min(target_height, final_y_max)

        # Check if the bbox is still valid (has a positive area).
        if final_x_min >= final_x_max or final_y_min >= final_y_max:
            transformed_bboxes.append([0, 0, 0, 0])  # The bbox is completely outside the crop.
        else:
            # make bbox integer
            final_x_min = int(final_x_min)
            final_y_min = int(final_y_min)
            final_x_max = int(final_x_max)
            final_y_max = int(final_y_max)
            transformed_bboxes.append([final_x_min, final_y_min, final_x_max, final_y_max])

    return transformed_bboxes


def restrict_size_below_area(
    height: int | None, width: int | None, ref_image: Image.Image, target_area: int = 1024 * 704, divisor: int = 64
):
    if height is not None and width is not None:
        return height, width

    height, width = ref_image.height, ref_image.width
    if height * width <= target_area:
        # If the original image area is already less than or equal to the target,
        # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        # Resize to fit within the target area and then pad to multiples of `divisor`
        max_upper_area = target_area  # Maximum allowed total pixel count after padding
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area

        # Calculate scale boundaries using quadratic equation
        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)  # Scale when maximum padding is applied
        max_scale = math.sqrt(max_upper_area / (height * width))  # Scale without any padding

    # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height, new_width = int(height * scale), int(width * scale)

        # Pad to make dimensions divisible by 64
        pad_height = (64 - new_height % 64) % 64
        pad_width = (64 - new_width % 64) % 64
        padded_height, padded_width = new_height + pad_height, new_width + pad_width

        if padded_height * padded_width <= max_upper_area:
            return padded_height, padded_width

    # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
    aspect_ratio = width / height
    target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
    target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)

    # Ensure the result is not larger than the original resolution
    if target_width >= width or target_height >= height:
        target_width = int(width // divisor * divisor)
        target_height = int(height // divisor * divisor)

    return target_height, target_width


class WanSpeech2VideoPipeline(WanVideoPipeline):
    def __init__(
        self,
        config: WanSpeech2VideoPipelineConfig,
        tokenizer: WanT5Tokenizer,
        text_encoder: WanTextEncoder,
        audio_encoder: Wav2Vec2Model,
        dit: WanS2VDiT,
        vae: WanVideoVAE,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            dit2=None,
            vae=vae,
            image_encoder=None,
        )
        self.audio_encoder = audio_encoder
        self.model_names = ["audio_encoder", "text_encoder", "dit", "vae"]

    def encode_ref_and_motion(
        self,
        ref_image: Image.Image | None,
        height: int,
        width: int,
        num_motion_frames: int,
        ref_as_first_frame: bool,
    ):
        self.load_models_to_device(["vae"])

        ref_frame = self.preprocess_image(ref_image)
        ref_frame = torch.stack([ref_frame], dim=2).squeeze(0)
        ref_latents = self.encode_video([ref_frame]).to(dtype=self.dtype, device=self.device)

        # They fix channel and motion frame length.
        motion_frames = torch.zeros([1, 3, num_motion_frames, height, width], dtype=self.dtype, device=self.device)
        if ref_as_first_frame:
            motion_frames[:, :, -6:] = ref_frame
        motion_latents = self.encode_video(motion_frames).to(dtype=self.dtype, device=self.device)

        return ref_latents, motion_latents, motion_frames

    def encode_pose(
        self,
        pose_video: List[Image.Image],
        pose_video_fps: int,
        num_clips: int,
        num_frames_per_clip: int,
        height: int,
        width: int,
    ):
        self.load_models_to_device(["vae"])
        max_num_pose_frames = num_frames_per_clip * num_clips
        pose_video = read_n_frames(pose_video, pose_video_fps, max_num_pose_frames, target_fps=self.config.fps)
        pose_frames = torch.stack([pil_to_tensor(frame) for frame in pose_video])
        pose_frames = pose_frames / 255.0 * 2 - 1.0
        pose_frames = resize_and_center_crop(pose_frames, height, width).permute(1, 0, 2, 3)[None]
        pose_frames_padding = torch.zeros([1, 3, max_num_pose_frames - pose_frames.shape[2], height, width])
        pose_frames = torch.cat([pose_frames, pose_frames_padding], dim=2)
        pose_frames_all_clips = torch.chunk(pose_frames, num_clips, dim=2)

        pose_latents_all_clips = []
        for pose_frames_per_clip in pose_frames_all_clips:
            pose_frames_per_clip = torch.cat([pose_frames_per_clip[:, :, 0:1], pose_frames_per_clip], dim=2)
            pose_latents_per_clip = self.encode_video([pose_frames_per_clip.squeeze(0)])[:, :, 1:].cpu()
            pose_latents_all_clips.append(pose_latents_per_clip)
        return pose_latents_all_clips

    def encode_audio(self, audio: torch.Tensor, num_frames_per_clip: int, num_clips: int):
        self.load_models_to_device(["audio_encoder"])
        audio_embed_bucket, max_num_clips = get_audio_embed_bucket_fps(
            audio_embed=extract_audio_feat(audio, self.audio_encoder, device=self.device),
            num_frames_per_batch=num_frames_per_clip,
            fps=self.config.fps,
        )
        audio_embed_bucket = audio_embed_bucket[None].to(self.device, self.dtype)
        audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, min(max_num_clips, num_clips)

    def encode_void_audio(self, void_audio: torch.Tensor, num_frames_per_clip: int):
        self.load_models_to_device(["audio_encoder"])
        void_audio_embed_bucket, _ = get_audio_embed_bucket_fps(
            audio_embed=extract_audio_feat(void_audio, self.audio_encoder, device=self.device),
            num_frames_per_batch=num_frames_per_clip,
            fps=self.config.fps,
        )
        void_audio_embed_bucket = void_audio_embed_bucket[None].to(self.device, self.dtype)
        void_audio_embed_bucket = void_audio_embed_bucket.permute(0, 2, 3, 1)
        return void_audio_embed_bucket[..., :num_frames_per_clip]

    def predict_noise_with_cfg(
        self,
        model: WanS2VDiT,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
        pose_cond: torch.Tensor,
        audio_input: torch.Tensor,
        num_motion_frames: int,
        num_motion_latents: int,
        drop_motion_frames: bool,
        audio_mask: torch.Tensor | None,
        void_audio_input: torch.Tensor | None,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                model=model,
                latents=latents,
                timestep=timestep,
                context=positive_prompt_emb,
                ref_latents=ref_latents,
                motion_latents=motion_latents,
                pose_cond=pose_cond,
                audio_input=audio_input,
                num_motion_frames=num_motion_frames,
                num_motion_latents=num_motion_latents,
                drop_motion_frames=drop_motion_frames,
                audio_mask=audio_mask,
                void_audio_input=void_audio_input,
            )
        if not batch_cfg:
            positive_noise_pred = self.predict_noise(
                model=model,
                latents=latents,
                timestep=timestep,
                context=positive_prompt_emb,
                ref_latents=ref_latents,
                motion_latents=motion_latents,
                pose_cond=pose_cond,
                audio_input=audio_input,
                num_motion_frames=num_motion_frames,
                num_motion_latents=num_motion_latents,
                drop_motion_frames=drop_motion_frames,
                audio_mask=audio_mask,
                void_audio_input=void_audio_input,
            )
            negative_noise_pred = self.predict_noise(
                model=model,
                latents=latents,
                timestep=timestep,
                context=negative_prompt_emb,
                ref_latents=ref_latents,
                motion_latents=motion_latents,
                pose_cond=pose_cond,
                audio_input=0.0 * audio_input,
                num_motion_frames=num_motion_frames,
                num_motion_latents=num_motion_latents,
                drop_motion_frames=drop_motion_frames,
                audio_mask=audio_mask,
                void_audio_input=void_audio_input,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            audio_input = torch.cat([audio_input, 0.0 * audio_input], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                model=model,
                latents=latents,
                timestep=timestep,
                context=prompt_emb,
                ref_latents=ref_latents,
                motion_latents=motion_latents,
                pose_cond=pose_cond,
                audio_input=audio_input,
                num_motion_frames=num_motion_frames,
                num_motion_latents=num_motion_latents,
                drop_motion_frames=drop_motion_frames,
                audio_mask=audio_mask,
                void_audio_input=void_audio_input,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(
        self,
        model: WanS2VDiT,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
        pose_cond: torch.Tensor,
        audio_input: torch.Tensor,
        num_motion_frames: int,
        num_motion_latents: int,
        drop_motion_frames: bool,
        audio_mask: torch.Tensor | None = None,
        void_audio_input: torch.Tensor | None = None,
    ):
        latents = latents.to(dtype=self.config.model_dtype, device=self.device)
        attn_kwargs = self.get_attn_kwargs(latents)

        noise_pred = model(
            x=latents,
            context=context,
            timestep=timestep,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            pose_cond=pose_cond,
            audio_input=audio_input,
            num_motion_frames=num_motion_frames,
            num_motion_latents=num_motion_latents,
            drop_motion_frames=drop_motion_frames,
            audio_mask=audio_mask,
            void_audio_input=void_audio_input,
            attn_kwargs=attn_kwargs,
        )
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        audio: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames_per_clip: int = 80,
        ref_image: Image.Image | None = None,
        pose_video: List[Image.Image] | None = None,
        pose_video_fps: int | None = None,
        void_audio: torch.Tensor | None = None,
        num_clips: int = 1,
        ref_as_first_frame: bool = False,
        speaker_bbox: List[List[int]] = [],
        speaker_end_sec: List[List[int]] = [],
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        assert ref_image is not None, "ref_image must be provided"
        cfg_scale = self.config.cfg_scale if cfg_scale is None else cfg_scale
        num_inference_steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
        original_height, original_width = ref_image.height, ref_image.width
        height, width = restrict_size_below_area(height, width, ref_image)

        # Initialize noise
        if dist.is_initialized() and seed is None:
            raise ValueError("must provide a seed when parallelism is enabled")

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = self.encode_prompt(negative_prompt)

        # Encode ref image, previous video and audio
        num_motion_frames = 73
        num_motion_latents = (num_motion_frames + 3) // 4
        ref_image = resize_and_center_crop(ref_image, height, width)
        ref_latents, motion_latents, motion_frames = self.encode_ref_and_motion(
            ref_image, height, width, num_motion_frames, ref_as_first_frame
        )
        audio_emb, num_clips = self.encode_audio(audio, num_frames_per_clip, num_clips)
        if len(speaker_end_sec) > 0:
            void_audio_emb = self.encode_void_audio(void_audio, num_frames_per_clip)
            speaker_bbox = transform_bbox(
                speaker_bbox,
                original_height,
                original_width,
                height,
                width,
            )
            audio_mask = get_face_mask(
                ref_image=ref_image,
                speaker_end_sec=speaker_end_sec,
                speaker_bbox=speaker_bbox,
                num_frames_total=num_clips * num_frames_per_clip,
                fps=self.config.fps,
                dtype=self.dtype,
            ).to(self.device)
        if pose_video is not None:
            pose_latents_all_clips = self.encode_pose(
                pose_video, pose_video_fps, num_clips, num_frames_per_clip, height, width
            )

        output_frames_all_clips = []
        for clip_idx in range(num_clips):
            num_latents_per_clip = num_frames_per_clip // 4
            noise = self.generate_noise(
                (
                    1,
                    self.vae.z_dim,
                    num_latents_per_clip,
                    height // self.upsampling_factor,
                    width // self.upsampling_factor,
                ),
                seed=seed + clip_idx,
                device="cpu",
                dtype=torch.float32,
            ).to(self.device)
            _, latents, sigmas, timesteps = self.prepare_latents(
                latents=noise,
                input_video=None,
                denoising_strength=None,
                num_inference_steps=num_inference_steps,
            )
            # Initialize sampler
            self.sampler.initialize(sigmas=sigmas)

            # Index audio emb and pose latents
            audio_emb_curr_clip = audio_emb[
                ..., (clip_idx * num_frames_per_clip) : ((clip_idx + 1) * num_frames_per_clip)
            ]
            pose_latents_curr_clip = (
                pose_latents_all_clips[clip_idx] if pose_video is not None else torch.zeros_like(latents)
            )
            pose_latents_curr_clip = pose_latents_curr_clip.to(dtype=self.dtype, device=self.device)
            if len(speaker_end_sec) > 0:
                audio_mask_curr_clip = audio_mask[
                    None, :, (clip_idx * num_latents_per_clip) : ((clip_idx + 1) * num_latents_per_clip)
                ]
            else:
                audio_mask_curr_clip, void_audio_emb = None, None

            # Denoise
            drop_motion_frames = (not ref_as_first_frame) and clip_idx == 0
            hide_progress = dist.is_initialized() and dist.get_rank() != 0
            for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
                self.load_models_to_device(["dit"])

                timestep = timestep[None].to(dtype=self.dtype, device=self.device)
                # Classifier-free guidance
                noise_pred = self.predict_noise_with_cfg(
                    model=self.dit,
                    latents=latents,
                    timestep=timestep,
                    positive_prompt_emb=prompt_emb_posi,
                    negative_prompt_emb=prompt_emb_nega,
                    cfg_scale=cfg_scale,
                    batch_cfg=self.config.batch_cfg,
                    ref_latents=ref_latents,
                    motion_latents=motion_latents,
                    pose_cond=pose_latents_curr_clip,
                    audio_input=audio_emb_curr_clip,
                    num_motion_frames=num_motion_frames,
                    num_motion_latents=num_motion_latents,
                    drop_motion_frames=drop_motion_frames,
                    audio_mask=audio_mask_curr_clip,
                    void_audio_input=void_audio_emb,
                )
                # Scheduler
                latents = self.sampler.step(latents, noise_pred, i)
                if progress_callback is not None:
                    progress_callback(i + 1, len(timesteps), "DENOISING")

            if drop_motion_frames:
                decode_latents = torch.cat([ref_latents, latents], dim=2)
            else:
                decode_latents = torch.cat([motion_latents, latents], dim=2)
            self.load_models_to_device(["vae"])
            output_frames_curr_clip = torch.stack(
                self.decode_video(decode_latents, progress_callback=progress_callback)
            )
            output_frames_curr_clip = output_frames_curr_clip[:, :, -(num_frames_per_clip):]
            if drop_motion_frames:
                output_frames_curr_clip = output_frames_curr_clip[:, :, 3:]
            output_frames_all_clips.append(output_frames_curr_clip.cpu())

            if clip_idx < num_clips - 1:
                f = output_frames_curr_clip.shape[2]
                if f <= num_motion_frames:
                    motion_frames = torch.cat([motion_frames[:, :, f:], output_frames_curr_clip], dim=2)
                else:
                    motion_frames = output_frames_curr_clip[:, :, -num_motion_frames:]
                motion_latents = self.encode_video(motion_frames)

        output_frames_all_clips = torch.cat(output_frames_all_clips, dim=2)
        output_frames_all_clips = self.vae_output_to_image(output_frames_all_clips)
        return output_frames_all_clips

    @classmethod
    def from_pretrained(cls, model_path_or_config: WanSpeech2VideoPipelineConfig) -> "WanSpeech2VideoPipeline":
        if isinstance(model_path_or_config, str):
            config = WanSpeech2VideoPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"loading dit state dict from {config.model_path} ...")
        dit_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        if config.t5_path is None:
            config.t5_path = fetch_model("muse/wan2.1-umt5", path="umt5.safetensors")
        if config.vae_path is None:
            config.vae_path = fetch_model("muse/wan2.1-vae", path="vae.safetensors")
        if config.audio_encoder_path is None:
            config.audio_encoder_path = fetch_model(
                "Wan-AI/Wan2.2-S2V-14B", path="wav2vec2-large-xlsr-53-english/model.safetensors"
            )

        logger.info(f"loading t5 state dict from {config.t5_path} ...")
        t5_state_dict = cls.load_model_checkpoint(config.t5_path, device="cpu", dtype=config.t5_dtype)

        logger.info(f"loading vae state dict from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        logger.info(f"loading audio encoder state dict from {config.audio_encoder_path} ...")
        wav2vec_state_dict = cls.load_model_checkpoint(
            config.audio_encoder_path, device="cpu", dtype=config.audio_encoder_dtype
        )

        state_dicts = WanS2VStateDicts(
            model=dit_state_dict,
            t5=t5_state_dict,
            vae=vae_state_dict,
            audio_encoder=wav2vec_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(
        cls, state_dicts: WanS2VStateDicts, config: WanSpeech2VideoPipelineConfig
    ) -> "WanSpeech2VideoPipeline":
        if config.parallelism > 1:
            pipe = ParallelWrapper(
                cfg_degree=config.cfg_degree,
                sp_ulysses_degree=config.sp_ulysses_degree,
                sp_ring_degree=config.sp_ring_degree,
                tp_degree=config.tp_degree,
                use_fsdp=config.use_fsdp,
            )
            pipe.load_module(cls._from_state_dict, state_dicts=state_dicts, config=config)
        else:
            pipe = cls._from_state_dict(state_dicts, config)
        return pipe

    @classmethod
    def _from_state_dict(
        cls, state_dicts: WanS2VStateDicts, config: WanSpeech2VideoPipelineConfig
    ) -> "WanSpeech2VideoPipeline":
        # default params from model config
        vae_type = "wan2.1-vae"
        dit_type = "wan2.2-s2v-14b"
        vae_config: dict = WanVideoVAE.get_model_config(vae_type)
        model_config: dict = WanS2VDiT.get_model_config(dit_type)
        config.boundary = model_config.pop("boundary", -1.0)
        config.shift = model_config.pop("shift", 5.0)
        config.cfg_scale = model_config.pop("cfg_scale", 5.0)
        config.num_inference_steps = model_config.pop("num_inference_steps", 50)
        config.fps = model_config.pop("fps", 16)

        init_device = "cpu" if config.offload_mode is not None else config.device
        tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=512, clean="whitespace")
        text_encoder = WanTextEncoder.from_state_dict(state_dicts.t5, device=init_device, dtype=config.t5_dtype)
        vae = WanVideoVAE.from_state_dict(
            state_dicts.vae, config=vae_config, device=init_device, dtype=config.vae_dtype
        )
        audio_encoder = Wav2Vec2Model.from_state_dict(
            state_dicts.audio_encoder, config=Wav2Vec2Config(), device=init_device, dtype=config.audio_encoder_dtype
        )

        with LoRAContext():
            cls._auto_enable_vsa(state_dicts.model, config)

            dit = WanS2VDiT.from_state_dict(
                state_dicts.model,
                config=model_config,
                device=("cpu" if config.use_fsdp else init_device),
                dtype=config.model_dtype,
                use_vsa=(config.dit_attn_impl.value == "vsa"),
            )
            if config.use_fp8_linear:
                enable_fp8_linear(dit)

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            audio_encoder=audio_encoder,
        )
        pipe.eval()

        if config.offload_mode is not None:
            pipe.enable_cpu_offload(config.offload_mode)

        if config.model_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16  # compute dtype
            pipe.enable_fp8_autocast(
                model_names=["dit"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.t5_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16  # compute dtype
            pipe.enable_fp8_autocast(
                model_names=["text_encoder"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.use_torch_compile:
            pipe.compile()
        return pipe
