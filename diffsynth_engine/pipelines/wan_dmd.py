import torch
import torch.distributed as dist
from typing import Callable, List, Optional
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.pipelines.wan_video import WanVideoPipeline


class WanDMDPipeline(WanVideoPipeline):
    def prepare_latents(
        self,
        latents,
        denoising_step_list,
    ):
        height, width = latents.shape[-2:]
        height, width = height * self.upsampling_factor, width * self.upsampling_factor
        sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps=1000)
        sigmas = sigmas[[1000 - t for t in denoising_step_list] + [-1]]
        timesteps = timesteps[[1000 - t for t in denoising_step_list]]
        init_latents = latents.clone()

        return init_latents, latents, sigmas, timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        input_image: Image.Image | None = None,
        seed=None,
        height=480,
        width=832,
        num_frames=81,
        denoising_step_list: List[int] = None,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        denoising_step_list = [1000, 750, 500, 250] if denoising_step_list is None else denoising_step_list
        divisor = 32 if self.vae.z_dim == 48 else 16  # 32 for wan2.2 vae, 16 for wan2.1 vae
        assert height % divisor == 0 and width % divisor == 0, f"height and width must be divisible by {divisor}"
        assert (num_frames - 1) % 4 == 0, "num_frames must be 4X+1"

        # Initialize noise
        if dist.is_initialized() and seed is None:
            raise ValueError("must provide a seed when parallelism is enabled")
        noise = self.generate_noise(
            (
                1,
                self.vae.z_dim,
                (num_frames - 1) // 4 + 1,
                height // self.upsampling_factor,
                width // self.upsampling_factor,
            ),
            seed=seed,
            device="cpu",
            dtype=torch.float32,
        ).to(self.device)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(noise, denoising_step_list)
        mask = torch.ones((1, 1, *latents.shape[2:]), dtype=latents.dtype, device=latents.device)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = None

        # Encode image
        image_clip_feature = self.encode_clip_feature(input_image, height, width)
        image_y = self.encode_vae_feature(input_image, num_frames, height, width)
        image_latents = self.encode_image_latents(input_image, height, width)
        if image_latents is not None:
            latents[:, :, : image_latents.shape[2], :, :] = image_latents
            init_latents = latents.clone()
            mask[:, :, : image_latents.shape[2], :, :] = 0

        # Initialize sampler
        self.sampler.initialize(sigmas=sigmas)

        # Denoise
        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
            if timestep.item() / 1000 >= self.config.boundary:
                self.load_models_to_device(["dit"])
                model = self.dit
            else:
                self.load_models_to_device(["dit2"])
                model = self.dit2

            timestep = timestep * mask[:, :, :, ::2, ::2].flatten()  # seq_len
            timestep = timestep.to(dtype=self.dtype, device=self.device)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                model=model,
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=prompt_emb_posi,
                negative_prompt_emb=prompt_emb_nega,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                cfg_scale=1.0,
                batch_cfg=self.config.batch_cfg,
            )
            # Scheduler
            latents = self.sampler.step(latents, noise_pred, i)
            latents = latents * mask + init_latents * (1 - mask)
            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps), "DENOISING")

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, progress_callback=progress_callback)
        frames = self.vae_output_to_image(frames)
        return frames
