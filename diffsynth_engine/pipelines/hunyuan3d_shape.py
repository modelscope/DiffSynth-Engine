import torch
from typing import Optional, Callable
from tqdm import tqdm
from PIL import Image
from diffsynth_engine.algorithm.noise_scheduler.flow_match.recifited_flow import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler.flow_match.flow_match_euler import FlowMatchEulerSampler
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.configs.pipeline import HunyuanPipelineConfig
from diffsynth_engine.models.hunyuan3d.hunyuan3d_vae import ShapeVAEDecoder
from diffsynth_engine.models.hunyuan3d.dino_image_encoder import ImageEncoder
from diffsynth_engine.models.hunyuan3d.hunyuan3d_dit import HunYuan3DDiT
from diffsynth_engine.utils.download import fetch_model
import numpy as np
import trimesh
import logging
import cv2
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


def array_to_tensor(np_array):
    image_pt = torch.tensor(np_array).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts


class Hunyuan3DImageProcessor:
    def __init__(self, size=512):
        self.size = size

    def recenter(self, image, border_ratio: float = 0.2):
        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]

        H, W, C = image.shape

        size = max(H, W)
        result = np.zeros((size, size, C), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        if h == 0 or w == 0:
            raise ValueError("input image is empty")
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2

        y2_min = (size - w2) // 2
        y2_max = y2_min + w2

        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
        )

        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255

        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)

        mask = mask * 255
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 255).astype(np.uint8)
        return result, mask

    def load_image(self, image, border_ratio=0.15):
        image = image.convert("RGBA")
        image = np.asarray(image)
        image, mask = self.recenter(image, border_ratio=border_ratio)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]

        image = array_to_tensor(image)
        mask = array_to_tensor(mask)
        return image, mask

    def __call__(self, image, border_ratio=0.15):
        image, mask = self.load_image(image, border_ratio=border_ratio)
        outputs = {"image": image, "mask": mask}
        return outputs


def export_to_trimesh(mesh_output):
    mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
    mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
    return mesh_output


class Hunyuan3DShapePipeline(BasePipeline):
    def __init__(
        self,
        config: HunyuanPipelineConfig,
        dit: HunYuan3DDiT,
        vae_decoder: ShapeVAEDecoder,
        image_encoder: ImageEncoder,
    ):
        super().__init__(
            device=config.device,
            dtype=config.model_dtype,
        )
        self.config = config
        self.dit = dit
        self.vae_decoder = vae_decoder
        self.image_encoder = image_encoder
        self.noise_scheduler = RecifitedFlowScheduler(shift=1.0)
        self.sampler = FlowMatchEulerSampler()
        self.image_processor = Hunyuan3DImageProcessor()
        self.model_names = ["dit", "vae_decoder", "image_encoder"]

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | HunyuanPipelineConfig):
        if isinstance(model_path_or_config, str):
            config = HunyuanPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        if config.vae_path is None:
            config.vae_path = fetch_model("muse/Hunyuan3d-2.1-Shape", path="vae.safetensors")
        if config.image_encoder_path is None:
            config.image_encoder_path = fetch_model("muse/Hunyuan3d-2.1-Shape", path="image_encoder.safetensors")

        logger.info(f"loading state dict from {config.model_path} ...")
        dit_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)
        logger.info(f"loading state dict from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)
        logger.info(f"loading state dict from {config.image_encoder_path} ...")
        image_encoder_state_dict = cls.load_model_checkpoint(
            config.image_encoder_path, device="cpu", dtype=config.image_encoder_dtype
        )

        dit = HunYuan3DDiT.from_state_dict(dit_state_dict, device=config.device, dtype=config.model_dtype)
        vae_decoder = ShapeVAEDecoder.from_state_dict(vae_state_dict, device=config.device, dtype=config.vae_dtype)
        image_encoder = ImageEncoder.from_state_dict(
            image_encoder_state_dict, device=config.device, dtype=config.image_encoder_dtype
        )
        pipe = cls(config, dit, vae_decoder, image_encoder)
        pipe.eval()
        return pipe

    def encode_image(self, image: Image.Image):
        image = self.image_processor(image)["image"].to(device=self.device, dtype=self.dtype)
        image_emb = self.image_encoder(image)
        uncond_image_emb = torch.zeros_like(image_emb, device=self.device, dtype=self.dtype)
        return torch.cat([image_emb, uncond_image_emb], dim=0)

    def decode_latents(
        self, latents, box_v=1.01, mc_level=0.0, num_chunks=8000, octree_resolution=384, mc_algo=None, enable_pbar=True
    ):
        latents = latents / self.vae_decoder.scale_factor
        latents = self.vae_decoder(latents)
        outputs = self.vae_decoder.latents2mesh(
            latents,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )
        return export_to_trimesh(outputs)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        image_emb = self.encode_image(image)

        latents = self.generate_noise((1, 4096, 64), seed=seed, device=self.device, dtype=self.dtype)
        sigmas, timesteps = self.noise_scheduler.schedule(
            num_inference_steps, sigma_min=1.0, sigma_max=0.0, append_value=1.0
        )
        self.sampler.initialize(sigmas=sigmas)
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(device=self.device) / 1000
            model_outputs = self.dit(
                x=torch.cat([latents, latents]),
                t=torch.cat([timestep, timestep]),
                cond=image_emb,
            )
            noise_pred, noise_pred_uncond = model_outputs.chunk(2)
            model_outputs = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            latents = self.sampler.step(latents, model_outputs, i)
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")
        return self.decode_latents(latents)
