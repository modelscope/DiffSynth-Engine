import re
import torch
import numpy as np
from einops import repeat
from typing import Callable, Dict, Optional, List
from tqdm import tqdm
from PIL import Image, ImageOps

from diffsynth_engine.configs import SDXLPipelineConfig, ControlNetParams, SDXLStateDicts
from diffsynth_engine.models.base import split_suffix
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.models.basic.timestep import TemporalTimesteps
from diffsynth_engine.models.sdxl import (
    SDXLTextEncoder,
    SDXLTextEncoder2,
    SDXLVAEDecoder,
    SDXLVAEEncoder,
    SDXLUNet,
    sdxl_unet_config,
)
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import accumulate
from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler
from diffsynth_engine.algorithm.sampler import EulerSampler
from diffsynth_engine.utils.prompt import tokenize_long_prompt
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH, SDXL_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils.platform import empty_cache
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class SDXLLoRAConverter(LoRAStateDictConverter):
    def _replace_kohya_te1_key(self, key):
        key = key.replace("lora_te1_text_model_encoder_layers_", "encoders.")
        key = re.sub(r"(\d+)_", r"\1.", key)
        key = key.replace("mlp_fc1", "fc1")
        key = key.replace("mlp_fc2", "fc2")
        key = key.replace("self_attn_q_proj", "attn.to_q")
        key = key.replace("self_attn_k_proj", "attn.to_k")
        key = key.replace("self_attn_v_proj", "attn.to_v")
        key = key.replace("self_attn_out_proj", "attn.to_out")
        return key

    def _replace_kohya_te2_key(self, key):
        key = key.replace("lora_te2_text_model_encoder_layers_", "encoders.")
        key = re.sub(r"(\d+)_", r"\1.", key)
        key = key.replace("mlp_fc1", "fc1")
        key = key.replace("mlp_fc2", "fc2")
        key = key.replace("self_attn_q_proj", "attn.to_q")
        key = key.replace("self_attn_k_proj", "attn.to_k")
        key = key.replace("self_attn_v_proj", "attn.to_v")
        key = key.replace("self_attn_out_proj", "attn.to_out")
        return key

    def _replace_kohya_unet_key(self, key):
        rename_dict = sdxl_unet_config["civitai"]["rename_dict"]
        key = key.replace("lora_unet_", "model.diffusion_model.")
        key = key.replace("ff_net", "ff.net")
        key = re.sub(r"(\d+)_", r"\1.", key)
        key = re.sub(r"_(\d+)", r".\1", key)
        name, suffix = split_suffix(key)
        if name not in rename_dict:
            raise ValueError(f"Unsupported key: {key}, name: {name}, suffix: {suffix}")
        key = rename_dict[name] + suffix
        return key

    def _from_kohya(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        unet_dict = {}
        te1_dict = {}
        te2_dict = {}
        for key, param in lora_state_dict.items():
            lora_args = {}
            if ".alpha" not in key:
                continue
            lora_args["alpha"] = param
            lora_args["up"] = lora_state_dict[key.replace(".alpha", ".lora_up.weight")]
            lora_args["down"] = lora_state_dict[key.replace(".alpha", ".lora_down.weight")]
            lora_args["rank"] = lora_args["up"].shape[1]
            key = key.replace(".alpha", "")
            if "lora_te1" in key:
                key = self._replace_kohya_te1_key(key)
                te1_dict[key] = lora_args
            elif "lora_te2" in key:
                key = self._replace_kohya_te2_key(key)
                te2_dict[key] = lora_args
            elif "lora_unet" in key:
                key = self._replace_kohya_unet_key(key)
                unet_dict[key] = lora_args
            else:
                raise ValueError(f"Unsupported key: {key}")
        # clip skip
        te1_dict = {k: v for k, v in te1_dict.items() if not k.startswith("encoders.11")}
        return {"unet": unet_dict, "text_encoder": te1_dict, "text_encoder_2": te2_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if "lora_te1" in key or "lora_te2" in key or "lora_unet" in key:
            return self._from_kohya(lora_state_dict)
        raise ValueError(f"Unsupported key: {key}")


class SDXLImagePipeline(BasePipeline):
    lora_converter = SDXLLoRAConverter()

    def __init__(
        self,
        config: SDXLPipelineConfig,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        text_encoder: SDXLTextEncoder,
        text_encoder_2: SDXLTextEncoder2,
        unet: SDXLUNet,
        vae_decoder: SDXLVAEDecoder,
        vae_encoder: SDXLVAEEncoder,
    ):
        super().__init__(
            vae_tiled=config.vae_tiled,
            vae_tile_size=config.vae_tile_size,
            vae_tile_stride=config.vae_tile_stride,
            device=config.device,
            dtype=config.model_dtype,
        )
        self.config = config
        # sampler
        self.noise_scheduler = ScaledLinearScheduler()
        self.sampler = EulerSampler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.add_time_proj = TemporalTimesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            device=config.device,
            dtype=config.model_dtype,
        )
        self.model_names = ["text_encoder", "text_encoder_2", "unet", "vae_decoder", "vae_encoder"]

    @classmethod
    def from_pretrained(cls, model_path_or_config: SDXLPipelineConfig) -> "SDXLImagePipeline":
        if isinstance(model_path_or_config, str):
            config = SDXLPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"loading state dict from {config.model_path} ...")
        model_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        if config.vae_path is None:
            vae_state_dict = model_state_dict
        else:
            logger.info(f"loading state dict from {config.vae_path} ...")
            vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        if config.clip_l_path is None:
            clip_l_state_dict = model_state_dict
        else:
            logger.info(f"loading state dict from {config.clip_l_path} ...")
            clip_l_state_dict = cls.load_model_checkpoint(config.clip_l_path, device="cpu", dtype=config.clip_l_dtype)

        if config.clip_g_path is None:
            clip_g_state_dict = model_state_dict
        else:
            logger.info(f"loading state dict from {config.clip_g_path} ...")
            clip_g_state_dict = cls.load_model_checkpoint(config.clip_g_path, device="cpu", dtype=config.clip_g_dtype)

        state_dicts = SDXLStateDicts(
            model=model_state_dict,
            clip_l=clip_l_state_dict,
            clip_g=clip_g_state_dict,
            vae=vae_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: SDXLStateDicts, config: SDXLPipelineConfig) -> "SDXLImagePipeline":
        init_device = "cpu" if config.offload_mode is not None else config.device
        tokenizer = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONF_PATH)
        tokenizer_2 = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_2_CONF_PATH)
        with LoRAContext():
            text_encoder = SDXLTextEncoder.from_state_dict(
                state_dicts.clip_l, device=init_device, dtype=config.clip_l_dtype
            )
            text_encoder_2 = SDXLTextEncoder2.from_state_dict(
                state_dicts.clip_g, device=init_device, dtype=config.clip_g_dtype
            )
            unet = SDXLUNet.from_state_dict(state_dicts.model, device=init_device, dtype=config.model_dtype)
        vae_decoder = SDXLVAEDecoder.from_state_dict(
            state_dicts.vae, device=init_device, dtype=config.vae_dtype, attn_impl="sdpa"
        )
        vae_encoder = SDXLVAEEncoder.from_state_dict(
            state_dicts.vae, device=init_device, dtype=config.vae_dtype, attn_impl="sdpa"
        )

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
        )
        pipe.eval()

        if config.offload_mode is not None:
            pipe.enable_cpu_offload(config.offload_mode)
        return pipe

    def encode_prompt(self, prompt, clip_skip):
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(self.device)
        prompt_emb_1 = self.text_encoder(input_ids, clip_skip=clip_skip)

        input_ids_2 = tokenize_long_prompt(self.tokenizer_2, prompt).to(self.device)
        prompt_emb_2, add_text_embeds = self.text_encoder_2(input_ids_2, clip_skip=clip_skip)

        # Merge
        if prompt_emb_1.shape[0] != prompt_emb_2.shape[0]:
            max_batch_size = min(prompt_emb_1.shape[0], prompt_emb_2.shape[0])
            prompt_emb_1 = prompt_emb_1[:max_batch_size]
            prompt_emb_2 = prompt_emb_2[:max_batch_size]
        prompt_emb = torch.concatenate([prompt_emb_1, prompt_emb_2], dim=-1)

        # For very long prompt, we only use the first 77 tokens to compute `add_text_embeds`.
        add_text_embeds = add_text_embeds[0:1]
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0] * prompt_emb.shape[1], -1))

        return prompt_emb, add_text_embeds

    def preprocess_control_image(self, image: Image.Image, mode="RGB") -> torch.Tensor:
        image = image.convert(mode)
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]
        image = torch.Tensor(image_array / 255).permute(2, 0, 1).unsqueeze(0)
        return image

    def prepare_controlnet_params(self, controlnet_params: List[ControlNetParams], h, w):
        results = []
        for param in controlnet_params:
            condition = self.preprocess_control_image(param.image).to(device=self.device, dtype=self.dtype)
            results.append(
                ControlNetParams(
                    model=param.model, scale=param.scale, image=condition, processor_name=param.processor_name
                )
            )
        return results

    def prepare_add_time_id(self, latents):
        height, width = latents.shape[2] * 8, latents.shape[3] * 8
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=self.device).repeat(latents.shape[0])
        # original_size_as_tuple(height, width)
        # crop_coords_top_left(0, 0)
        # target_size_as_tuple(height, width)
        return add_time_id

    def prepare_add_embeds(self, add_text_embeds, add_time_id, dtype):
        time_embeds = self.add_time_proj(add_time_id)
        time_embeds = time_embeds.reshape((add_text_embeds.shape[0], -1))
        add_embeds = torch.concat([add_text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(dtype)
        return add_embeds

    def predict_multicontrolnet(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        add_text_embeds: torch.Tensor,
        add_time_id: torch.Tensor,
        controlnet_params: List[ControlNetParams],
        current_step: int,
        total_step: int,
    ):
        controlnet_res_stack = None
        for param in controlnet_params:
            current_scale = param.scale
            if not (
                current_step >= param.control_start * total_step and current_step <= param.control_end * total_step
            ):
                # if current_step is not in the control range
                # skip this controlnet
                continue
            if self.offload_mode is not None:
                empty_cache()
                param.model.to(self.device)
            if param.image.shape[0] != latents.shape[0]:
                param.image = repeat(param.image, "1 c h w -> b c h w", b=latents.shape[0])
            controlnet_res = param.model(
                latents,
                timestep,
                prompt_emb,
                param.image,
                param.processor_name,
                add_time_id,
                add_text_embeds,
            )
            controlnet_res = [res * current_scale for res in controlnet_res]
            if self.offload_mode is not None:
                param.model.to("cpu")
                empty_cache()
            controlnet_res_stack = accumulate(controlnet_res_stack, controlnet_res)
        return controlnet_res_stack

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        positive_add_text_embeds: torch.Tensor,
        negative_add_text_embeds: torch.Tensor,
        controlnet_params: List[ControlNetParams],
        current_step: int,
        total_step: int,
        add_time_id: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool = True,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                latents, timestep, positive_prompt_emb, add_time_id, controlnet_params, current_step, total_step
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents,
                timestep,
                positive_prompt_emb,
                positive_add_text_embeds,
                add_time_id,
                controlnet_params,
                current_step,
                total_step,
            )
            negative_noise_pred = self.predict_noise(
                latents,
                timestep,
                negative_prompt_emb,
                negative_add_text_embeds,
                add_time_id,
                controlnet_params,
                current_step,
                total_step,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            add_time_ids = torch.cat([add_time_id, add_time_id], dim=0)
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            add_text_embeds = torch.cat([positive_add_text_embeds, negative_add_text_embeds], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents,
                timestep,
                prompt_emb,
                add_text_embeds,
                add_time_ids,
                controlnet_params,
                current_step,
                total_step,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(
        self, latents, timestep, prompt_emb, add_text_embeds, add_time_id, controlnet_params, current_step, total_step
    ):
        y = self.prepare_add_embeds(add_text_embeds, add_time_id, self.dtype)
        controlnet_res_stack = self.predict_multicontrolnet(
            latents, timestep, prompt_emb, add_text_embeds, add_time_id, controlnet_params, current_step, total_step
        )

        noise_pred = self.unet(
            x=latents,
            timestep=timestep,
            y=y,
            context=prompt_emb,
            controlnet_res_stack=controlnet_res_stack,
            device=self.device,
        )
        return noise_pred

    def unload_loras(self):
        self.unet.unload_loras()
        self.text_encoder.unload_loras()
        self.text_encoder_2.unload_loras()

    def prepare_mask(
        self, input_image: Image.Image, mask_image: Image.Image, vae_scale_factor: int = 8, latent_channels=4
    ) -> torch.Tensor:
        height, width = mask_image.size
        # mask
        mask = torch.Tensor(np.array(mask_image) / 255).unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(height // vae_scale_factor, width // vae_scale_factor))
        mask = repeat(mask, "b 1 h w -> b c h w", c=latent_channels)
        mask = mask.to(self.device, self.dtype)
        # overlay_image
        overlay_image = Image.new("RGBa", (width, height))
        overlay_image.paste(input_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask_image.convert("L")))
        overlay_image = overlay_image.convert("RGBA")
        # mask: [1, 4, H, W]  = mask_image[1, 1, H//8, W//8] * 4
        # overlay_image: [1, 4, H, W]  = mask_image[1, 1, H, W] + input_image[1, 3, H, W]
        return mask, overlay_image

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 7.5,
        clip_skip: int = 2,
        input_image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        seed: int | None = None,
        controlnet_params: List[ControlNetParams] | ControlNetParams = [],
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        if not isinstance(controlnet_params, list):
            controlnet_params = [controlnet_params]

        if input_image is not None:
            width, height = input_image.size
        self.validate_image_size(height, width, minimum=64, multiple_of=8)
        noise = self.generate_noise((1, 4, height // 8, width // 8), seed=seed, device=self.device, dtype=self.dtype)

        init_latents, latents, sigmas, timesteps = self.prepare_latents(
            noise, input_image, denoising_strength, num_inference_steps
        )
        mask, overlay_image = None, None
        if mask_image is not None:
            mask, overlay_image = self.prepare_mask(input_image, mask_image, vae_scale_factor=8)
        # Initialize sampler
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas, mask=mask)

        # ControlNet
        controlnet_params = self.prepare_controlnet_params(controlnet_params, h=height, w=width)
        # Encode prompts
        self.load_models_to_device(["text_encoder", "text_encoder_2"])
        positive_prompt_emb, positive_add_text_embeds = self.encode_prompt(prompt, clip_skip=clip_skip)
        if negative_prompt != "":
            negative_prompt_emb, negative_add_text_embeds = self.encode_prompt(negative_prompt, clip_skip=clip_skip)
        else:
            # from automatic1111/stable-diffusion-webui
            negative_prompt_emb, negative_add_text_embeds = (
                torch.zeros_like(positive_prompt_emb),
                torch.zeros_like(positive_add_text_embeds),
            )

        # Prepare extra input
        add_time_id = self.prepare_add_time_id(latents)

        # Denoise
        self.load_models_to_device(["unet"])
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            positive_prompt_emb = positive_prompt_emb.to(self.dtype)
            negative_prompt_emb = negative_prompt_emb.to(self.dtype)
            positive_add_text_embeds = positive_add_text_embeds.to(self.dtype)
            negative_add_text_embeds = negative_add_text_embeds.to(self.dtype)
            add_time_id = add_time_id.to(self.dtype)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=positive_prompt_emb,
                negative_prompt_emb=negative_prompt_emb,
                positive_add_text_embeds=positive_add_text_embeds,
                negative_add_text_embeds=negative_add_text_embeds,
                add_time_id=add_time_id,
                cfg_scale=cfg_scale,
                controlnet_params=controlnet_params,
                current_step=i,
                total_step=len(timesteps),
                batch_cfg=self.config.batch_cfg,
            )
            # Denoise
            latents = self.sampler.step(latents, noise_pred, i)
            # UI
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")
        if mask_image is not None:
            latents = latents * mask + init_latents * (1 - mask)
        # Decode image
        self.load_models_to_device(["vae_decoder"])
        vae_output = self.decode_image(latents)
        image = self.vae_output_to_image(vae_output)

        if mask_image is not None:
            image = image.convert("RGBA")
            image.alpha_composite(overlay_image)
            image = image.convert("RGB")
        # offload all models
        self.load_models_to_device([])
        return image
