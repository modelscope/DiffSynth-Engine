import re
import torch
import numpy as np
from einops import repeat
from typing import Callable, Dict, Optional, List
from tqdm import tqdm
from PIL import Image, ImageOps

from diffsynth_engine.configs import SDPipelineConfig, ControlNetParams, SDStateDicts
from diffsynth_engine.models.base import split_suffix
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.models.sd import SDTextEncoder, SDVAEDecoder, SDVAEEncoder, SDUNet, sd_unet_config
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import accumulate
from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler
from diffsynth_engine.algorithm.sampler import EulerSampler
from diffsynth_engine.utils.prompt import tokenize_long_prompt
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.platform import empty_cache
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

re_compiled = {}
re_digits = re.compile(r"\d+")
suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    },
}


def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f"model.diffusion_model.input_blocks.0.0{m[0]}"

    if match(m, r"lora_unet_conv_out(.*)"):
        return f"model.diffusion_model.out.2{m[0]}"

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"model.diffusion_model.time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"model.diffusion_model.input_blocks.{1 + m[0] * 3 + m[2]}.{1 if m[1] == 'attentions' else 0}.{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"model.diffusion_model.middle_block.{1 if m[0] == 'attentions' else m[1] * 2}.{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"model.diffusion_model.output_blocks.{m[0] * 3 + m[2]}.{1 if m[1] == 'attentions' else 0}.{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"model.diffusion_model.input_blocks.{3 + m[0] * 3}.0.op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"model.diffusion_model.output_blocks.{2 + m[0] * 3}.{2 if m[0] > 0 else 1}.conv"
    return key


class SDLoRAConverter(LoRAStateDictConverter):
    def _replace_kohya_te_key(self, key):
        key = key.replace("lora_te_text_model_encoder_layers_", "encoders.")
        key = re.sub(r"(\d+)_", r"\1.", key)
        key = key.replace("mlp_fc1", "fc1")
        key = key.replace("mlp_fc2", "fc2")
        key = key.replace("self_attn_q_proj", "attn.to_q")
        key = key.replace("self_attn_k_proj", "attn.to_k")
        key = key.replace("self_attn_v_proj", "attn.to_v")
        key = key.replace("self_attn_out_proj", "attn.to_out")
        return key

    def _replace_kohya_unet_key(self, key):
        rename_dict = sd_unet_config["civitai"]["rename_dict"]
        key = convert_diffusers_name_to_compvis(key)
        key = re.sub(r"(\d+)_", r"\1.", key)
        key = re.sub(r"_(\d+)", r".\1", key)
        key = key.replace("ff_net", "ff.net")
        name, suffix = split_suffix(key)
        if name not in rename_dict:
            raise ValueError(f"Unsupported key: {key}, name: {name}, suffix: {suffix}")
        key = rename_dict[name] + suffix
        return key

    def _from_kohya(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        unet_dict = {}
        te_dict = {}
        for key, param in lora_state_dict.items():
            lora_args = {}
            if ".alpha" not in key:
                continue
            lora_args["alpha"] = param
            lora_args["up"] = lora_state_dict[key.replace(".alpha", ".lora_up.weight")].squeeze()
            lora_args["down"] = lora_state_dict[key.replace(".alpha", ".lora_down.weight")].squeeze()
            lora_args["rank"] = lora_args["up"].shape[1]
            key = key.replace(".alpha", "")
            if "lora_unet" in key:
                key = self._replace_kohya_unet_key(key)
                unet_dict[key] = lora_args
            elif "lora_te" in key:
                key = self._replace_kohya_te_key(key)
                te_dict[key] = lora_args
        return {"unet": unet_dict, "text_encoder": te_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if "lora_te" in key or "lora_unet" in key:
            return self._from_kohya(lora_state_dict)
        raise ValueError(f"Unsupported key: {key}")


class SDImagePipeline(BasePipeline):
    lora_converter = SDLoRAConverter()

    def __init__(
        self,
        config: SDPipelineConfig,
        tokenizer: CLIPTokenizer,
        text_encoder: SDTextEncoder,
        unet: SDUNet,
        vae_decoder: SDVAEDecoder,
        vae_encoder: SDVAEEncoder,
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
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.model_names = ["text_encoder", "unet", "vae_decoder", "vae_encoder"]

    @classmethod
    def from_pretrained(cls, model_path_or_config: SDPipelineConfig) -> "SDImagePipeline":
        if isinstance(model_path_or_config, str):
            config = SDPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"loading state dict from {config.model_path} ...")
        model_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        if config.vae_path is None:
            vae_state_dict = model_state_dict
        else:
            logger.info(f"loading state dict from {config.vae_path} ...")
            vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        if config.clip_path is None:
            clip_state_dict = model_state_dict
        else:
            logger.info(f"loading state dict from {config.clip_path} ...")
            clip_state_dict = cls.load_model_checkpoint(config.clip_path, device="cpu", dtype=config.clip_dtype)

        state_dicts = SDStateDicts(
            model=model_state_dict,
            vae=vae_state_dict,
            clip=clip_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: SDStateDicts, config: SDPipelineConfig) -> "SDImagePipeline":
        init_device = "cpu" if config.offload_mode is not None else config.device
        tokenizer = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONF_PATH)
        with LoRAContext():
            text_encoder = SDTextEncoder.from_state_dict(state_dicts.clip, device=init_device, dtype=config.clip_dtype)
            unet = SDUNet.from_state_dict(state_dicts.model, device=init_device, dtype=config.model_dtype)
        vae_decoder = SDVAEDecoder.from_state_dict(
            state_dicts.vae, device=init_device, dtype=config.vae_dtype, attn_impl="sdpa"
        )
        vae_encoder = SDVAEEncoder.from_state_dict(
            state_dicts.vae, device=init_device, dtype=config.vae_dtype, attn_impl="sdpa"
        )

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
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
        prompt_emb = self.text_encoder(input_ids, clip_skip=clip_skip)
        return prompt_emb

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
                    model=param.model,
                    scale=param.scale,
                    image=condition,
                )
            )
        return results

    def predict_multicontrolnet(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
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
            controlnet_res = param.model(latents, timestep, prompt_emb, param.image)
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
        controlnet_params: List[ControlNetParams],
        current_step: int,
        total_step: int,
        cfg_scale: float,
        batch_cfg: bool = True,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                latents, timestep, positive_prompt_emb, controlnet_params, current_step, total_step
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents, timestep, positive_prompt_emb, controlnet_params, current_step, total_step
            )
            negative_noise_pred = self.predict_noise(
                latents, timestep, negative_prompt_emb, controlnet_params, current_step, total_step
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents, timestep, prompt_emb, controlnet_params, current_step, total_step
            ).chunk(2)
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, latents, timestep, prompt_emb, controlnet_params, current_step, total_step):
        controlnet_res_stack = self.predict_multicontrolnet(
            latents, timestep, prompt_emb, controlnet_params, current_step, total_step
        )

        noise_pred = self.unet(
            x=latents,
            timestep=timestep,
            context=prompt_emb,
            controlnet_res_stack=controlnet_res_stack,
            device=self.device,
        )
        return noise_pred

    def unload_loras(self):
        self.unet.unload_loras()
        self.text_encoder.unload_loras()

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
        clip_skip: int = 1,
        input_image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
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
        self.load_models_to_device(["text_encoder"])
        positive_prompt_emb = self.encode_prompt(prompt, clip_skip=clip_skip)
        negative_prompt_emb = self.encode_prompt(negative_prompt, clip_skip=clip_skip)

        # Denoise
        self.load_models_to_device(["unet"])
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=positive_prompt_emb,
                negative_prompt_emb=negative_prompt_emb,
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
        # Paste Overlay Image
        if mask_image is not None:
            image = image.convert("RGBA")
            image.alpha_composite(overlay_image)
            image = image.convert("RGB")
        # offload all models
        self.load_models_to_device([])
        return image
