import math
from mediapipe.python.solutions.face_detection import FaceDetection

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Callable, List, Optional
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.configs import WanSound2VideoPipelineConfig
from diffsynth_engine.models.wan.wan_s2v_dit import WanS2VDiT
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_audio_encoder import Wav2Vec2Model, Wav2Vec2Config, get_audio_embed_bucket_fps, extract_audio_feat
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.pipelines.wan_video import WanVideoPipeline
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils.image import resize_and_center_crop
from diffsynth_engine.utils.video import read_n_frames
from diffsynth_engine.utils import logging


logger = logging.get_logger(__name__)


def face_bbox_detect(image: np.ndarray, bbox_scale=1.0):
    mp_face_detector = FaceDetection(min_detection_confidence=0.1)
    results = mp_face_detector.process(image)
    bboxes = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x_min = int(bboxC.xmin * iw)
            y_min = int(bboxC.ymin * ih)
            x_max = int((bboxC.xmin + bboxC.width) * iw)
            y_max = int((bboxC.ymin + bboxC.height) * ih)

            # Adjust bbox size
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(0, x_min - int((bbox_scale - 1.0) * width / 2))
            y_min = max(0, y_min - int((bbox_scale - 1.0) * height / 2))
            x_max = min(iw, x_max + int((bbox_scale - 1.0) * width / 2))
            y_max = min(ih, y_max + int((bbox_scale - 1.0) * height / 2))

            bboxes.append([x_min, y_min, x_max, y_max])

    if len(bboxes) == 0:
        mp_face_range_detector = FaceDetection(min_detection_confidence=0.1, model_selection=1)
        range_results = mp_face_range_detector.process(image)
        if range_results.detections:
            for detection in range_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x_min = int(bboxC.xmin * iw)
                y_min = int(bboxC.ymin * ih)
                x_max = int((bboxC.xmin + bboxC.width) * iw)
                y_max = int((bboxC.ymin + bboxC.height) * ih)

                # Adjust bbox size
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - int((bbox_scale - 1.0) * width / 2))
                y_min = max(0, y_min - int((bbox_scale - 1.0) * height / 2))
                x_max = min(iw, x_max + int((bbox_scale - 1.0) * width / 2))
                y_max = min(ih, y_max + int((bbox_scale - 1.0) * height / 2))

                bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes


def generate_bbox_mask(
    image: np.ndarray,
    bboxes: List[List[int]],
    skip_person_ids: List[int],
    y_offset=0.0
):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for id, bbox in enumerate(bboxes):
        if id in skip_person_ids:
            continue
        x_min, y_min, x_max, y_max = bbox
        height_offset = int((y_max - y_min) * y_offset)
        y_min = max(0, y_min + height_offset)
        y_max = min(mask.shape[0], y_max + height_offset)
        mask[y_min:y_max, x_min:x_max] = 255
    return mask


def get_face_mask(
    ref_image: Image.Image,
    speaking_duration: List[List[int]],
    num_frames_total: int,
    fps=16,
    bbox_scale=1.3,
    temporal_scale=4,
    spatial_scale=16,
    dtype=torch.bfloat16,
    device="cuda",
):
    ref_image = np.array(ref_image)
    mask_height, mask_width = ref_image.shape[0], ref_image.shape[1]
    face_mask = torch.zeros(
        [1, num_frames_total, mask_height, mask_width],
        dtype=dtype,
    )
    face_bboxs = face_bbox_detect(ref_image, bbox_scale=bbox_scale)
    prev_time = 0
    for person_id, end_time in speaking_duration:
        start_id = int(prev_time * fps)
        end_id = int(end_time * fps)
        mask_img = generate_bbox_mask(
            image=ref_image,
            bboxes=face_bboxs,
            skip_person_ids=[person_id],
        )
        face_mask[0, start_id:end_id] = torch.from_numpy(mask_img)[None].to(dtype=dtype, device=device) / 255
        prev_time = end_time
        if end_id > num_frames_total:
            break

    face_mask_resized = (
        F.interpolate(
            face_mask[None],
            size=(
                num_frames_total // temporal_scale,
                mask_height // spatial_scale,
                mask_width // spatial_scale,
            ),
            mode="nearest",
        )[0]
        .to(dtype=dtype, device=device)
    )
    return 1 - face_mask_resized


def get_size(height: int | None, width: int | None, ref_image: Image.Image, target_area: int = 1024 * 704, divisor: int = 64):
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
    # Use binary search-like iteration to find this scale
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


class WanSound2VideoPipeline(WanVideoPipeline):
    def __init__(
        self,
        config: WanSound2VideoPipelineConfig,
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
        self, pose_video_path: str, num_clips: int, num_frames_per_clip: int, height: int, width: int
    ):
        self.load_models_to_device(["vae"])
        max_num_pose_frames = num_frames_per_clip * num_clips
        pose_video = read_n_frames(pose_video_path, max_num_pose_frames, target_fps=self.config.fps)
        pose_frames = torch.from_numpy(pose_video)
        pose_frames = pose_frames.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
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

    def encode_audio(self, audio_path: str, num_frames_per_clip: int, num_clips: int):
        self.load_models_to_device(["audio_encoder"])
        audio_embed_bucket, max_num_clips = get_audio_embed_bucket_fps(
            audio_embed=extract_audio_feat(audio_path, self.audio_encoder, device=self.device),
            num_frames_per_batch=num_frames_per_clip,
            fps=self.config.fps,
        )
        audio_embed_bucket = audio_embed_bucket[None].to(self.device, self.dtype)
        audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, min(max_num_clips, num_clips)

    def encode_void_audio(self, void_audio_path: str, num_frames_per_clip: int):
        self.load_models_to_device(["audio_encoder"])
        void_audio_embed_bucket, _ = get_audio_embed_bucket_fps(
            audio_embed=extract_audio_feat(void_audio_path, self.audio_encoder, device=self.device),
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
        )
        return noise_pred 

    @torch.no_grad()
    def __call__(
        self,
        audio_path: str,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames_per_clip: int = 80,
        ref_image: Image.Image | None = None,
        pose_video_path: str | None = None,
        void_audio_path: str | None = None,
        num_clips: int = 1,
        ref_as_first_frame: bool = False,
        speaking_duration: List[List[int]] = [],
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        assert ref_image is not None, "ref_image must be provided"
        cfg_scale = self.config.cfg_scale if cfg_scale is None else cfg_scale
        num_inference_steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
        height, width = get_size(height, width, ref_image)

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
        audio_emb, num_clips = self.encode_audio(audio_path, num_frames_per_clip, num_clips)
        if len(speaking_duration) > 0:
            void_audio_emb = self.encode_void_audio(void_audio_path, num_frames_per_clip)
            audio_mask = get_face_mask(
                ref_image,
                speaking_duration,
                num_frames_total=num_clips * num_frames_per_clip,
                fps=self.config.fps,
                bbox_scale=1.3,
                device=self.device,
                dtype=self.dtype,
            )
        if pose_video_path is not None:
            pose_latents_all_clips = self.encode_pose(pose_video_path, num_clips, num_frames_per_clip, height, width)

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
            init_latents, latents, sigmas, timesteps = self.prepare_latents(
                latents=noise,
                input_video=None,
                denoising_strength=None,
                num_inference_steps=num_inference_steps,
            )
            # Initialize sampler
            self.sampler.initialize(sigmas=sigmas)

            # Index audio emb and pose latents
            audio_emb_curr_clip = audio_emb[..., (clip_idx * num_frames_per_clip) : ((clip_idx + 1) * num_frames_per_clip)]
            pose_latents_curr_clip = (
                pose_latents_all_clips[clip_idx] if pose_video_path is not None else torch.zeros_like(latents)
            )
            pose_latents_curr_clip = pose_latents_curr_clip.to(dtype=self.dtype, device=self.device)
            if len(speaking_duration) > 0:
                audio_mask_curr_clip = audio_mask[None, :, (clip_idx * num_latents_per_clip) : ((clip_idx + 1) * num_latents_per_clip)]
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
            output_frames_curr_clip = torch.stack(self.decode_video(decode_latents, progress_callback=progress_callback))
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
    def from_pretrained(cls, model_path_or_config: WanSound2VideoPipelineConfig) -> "WanSound2VideoPipeline":
        if isinstance(model_path_or_config, str):
            config = WanSound2VideoPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"loading dit state dict from {config.model_path} ...")
        dit_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)
        dit_type = "wan2.2-s2v-14b"

        if config.t5_path is None:
            config.t5_path = fetch_model("muse/wan2.1-umt5", path="umt5.safetensors")
        if config.vae_path is None:
            config.vae_path = fetch_model("muse/wan2.1-vae", path="vae.safetensors")
        if config.audio_encoder_path is None:
            config.audio_encoder_path = fetch_model("Wan-AI/Wan2.2-S2V-14B", path="wav2vec2-large-xlsr-53-english/model.safetensors")

        logger.info(f"loading t5 state dict from {config.t5_path} ...")
        t5_state_dict = cls.load_model_checkpoint(config.t5_path, device="cpu", dtype=config.t5_dtype)

        logger.info(f"loading vae state dict from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)
        vae_type = "wan2.1-vae"

        logger.info(f"loading audio encoder state dict from {config.audio_encoder_path} ...")
        wav2vec_state_dict = cls.load_model_checkpoint(config.audio_encoder_path, device="cpu", dtype=config.audio_encoder_dtype)

        # default params from model config
        vae_config: dict = WanVideoVAE.get_model_config(vae_type)
        model_config: dict = WanS2VDiT.get_model_config(dit_type)
        config.boundary = model_config.pop("boundary", -1.0)
        config.shift = model_config.pop("shift", 5.0)
        config.cfg_scale = model_config.pop("cfg_scale", 5.0)
        config.num_inference_steps = model_config.pop("num_inference_steps", 50)
        config.fps = model_config.pop("fps", 16)

        init_device = "cpu" if config.parallelism > 1 or config.offload_mode is not None else config.device
        tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=512, clean="whitespace")
        text_encoder = WanTextEncoder.from_state_dict(t5_state_dict, device=init_device, dtype=config.t5_dtype)
        vae = WanVideoVAE.from_state_dict(vae_state_dict, config=vae_config, device=init_device, dtype=config.vae_dtype)
        audio_encoder = Wav2Vec2Model.from_state_dict(wav2vec_state_dict, config=Wav2Vec2Config(), device=init_device, dtype=config.audio_encoder_dtype)

        with LoRAContext():
            attn_kwargs = {
                "attn_impl": config.dit_attn_impl,
                "sparge_smooth_k": config.sparge_smooth_k,
                "sparge_cdfthreshd": config.sparge_cdfthreshd,
                "sparge_simthreshd1": config.sparge_simthreshd1,
                "sparge_pvthreshd": config.sparge_pvthreshd,
            }
            dit = WanS2VDiT.from_state_dict(
                dit_state_dict,
                config=model_config,
                device=init_device,
                dtype=config.model_dtype,
                attn_kwargs=attn_kwargs,
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

        if config.parallelism > 1:
            return ParallelWrapper(
                pipe,
                cfg_degree=config.cfg_degree,
                sp_ulysses_degree=config.sp_ulysses_degree,
                sp_ring_degree=config.sp_ring_degree,
                tp_degree=config.tp_degree,
                use_fsdp=config.use_fsdp,
                device="cuda",
            )
        if config.use_torch_compile:
            pipe.compile()
        return pipe
