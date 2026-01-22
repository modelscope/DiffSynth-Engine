import torch
import math
import json
import torchvision
from typing import Callable, List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange

from diffsynth_engine.configs import (
    Flux2KleinPipelineConfig,
    Flux2StateDicts,
)
from diffsynth_engine.models.basic.lora import LoRAContext

from diffsynth_engine.models.flux2 import (
    Flux2DiT,
    Flux2VAE,
)
from diffsynth_engine.models.z_image import (
    Qwen3Model,
    Qwen3Config,
)
from transformers import AutoTokenizer
from diffsynth_engine.utils.constants import (
    Z_IMAGE_TEXT_ENCODER_CONFIG_FILE,
    Z_IMAGE_TOKENIZER_CONF_PATH,
    FLUX2_TEXT_ENCODER_8B_CONF_PATH,
)
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import calculate_shift
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model

logger = logging.get_logger(__name__)


class Flux2LoRAConverter(LoRAStateDictConverter):
    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            if "lora_A.weight" in key:
                lora_b_key = key.replace("lora_A.weight", "lora_B.weight")
                target_key = key.replace(".lora_A.weight", "").replace("diffusion_model.", "")

                up = lora_state_dict[lora_b_key]
                rank = up.shape[1]

                dit_dict[target_key] = {
                    "down": param,
                    "up": up,
                    "rank": rank,
                    "alpha": lora_state_dict.get(key.replace("lora_A.weight", "alpha"), rank),
                }

        return {"dit": dit_dict}

    def _from_diffsynth(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            if "lora_A.default.weight" in key:
                lora_b_key = key.replace("lora_A.default.weight", "lora_B.default.weight")
                target_key = key.replace(".lora_A.default.weight", "")

                up = lora_state_dict[lora_b_key]
                rank = up.shape[1]

                dit_dict[target_key] = {
                    "down": param,
                    "up": up,
                    "rank": rank,
                    "alpha": lora_state_dict.get(key.replace("lora_A.default.weight", "alpha"), rank),
                }

        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if key.startswith("diffusion_model."):
            return self._from_diffusers(lora_state_dict)
        else:
            return self._from_diffsynth(lora_state_dict)


def model_fn_flux2(
    dit: Flux2DiT,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    edit_latents=None,
    edit_image_ids=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    image_seq_len = latents.shape[1]
    if edit_latents is not None:
        latents = torch.concat([latents, edit_latents], dim=1)
        image_ids = torch.concat([image_ids, edit_image_ids], dim=1)
    embedded_guidance = torch.tensor([embedded_guidance], device=latents.device)
    model_output = dit(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=embedded_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    model_output = model_output[:, :image_seq_len]
    return model_output


class Flux2KleinPipeline(BasePipeline):
    lora_converter = Flux2LoRAConverter()

    def __init__(
        self,
        config: Flux2KleinPipelineConfig,
        tokenizer: AutoTokenizer,
        text_encoder: Qwen3Model,
        dit: Flux2DiT,
        vae: Flux2VAE,
    ):
        super().__init__(
            vae_tiled=config.vae_tiled,
            vae_tile_size=config.vae_tile_size,
            vae_tile_stride=config.vae_tile_stride,
            device=config.device,
            dtype=config.model_dtype,
        )
        self.config = config

        # Scheduler  
        self.noise_scheduler = RecifitedFlowScheduler(shift=1.0, use_dynamic_shifting=True, exponential_shift_mu=None)
        self.sampler = FlowMatchEulerSampler()
        self.tokenizer = tokenizer
        # Models
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae

        self.model_names = ["text_encoder", "dit", "vae"]

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | Flux2KleinPipelineConfig) -> "Flux2KleinPipeline":
        if isinstance(model_path_or_config, str):
            config = Flux2KleinPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"Loading state dict from {config.model_path} ...")

        model_state_dict = cls.load_model_checkpoint(
            config.model_path, device="cpu", dtype=config.model_dtype, convert_dtype=False
        )

        if config.vae_path is None:
            config.vae_path = fetch_model("black-forest-labs/FLUX.2-klein-4B", path="vae/*.safetensors")
        logger.info(f"Loading VAE from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        if config.encoder_path is None:
            if config.model_size == "4B":
                config.encoder_path = fetch_model("black-forest-labs/FLUX.2-klein-4B", path="text_encoder/*.safetensors")
            else:
                config.encoder_path = fetch_model("black-forest-labs/FLUX.2-klein-9B", path="text_encoder/*.safetensors")
        logger.info(f"Loading Text Encoder from {config.encoder_path} ...")
        text_encoder_state_dict = cls.load_model_checkpoint(
            config.encoder_path, device="cpu", dtype=config.encoder_dtype
        )

        state_dicts = Flux2StateDicts(
            model=model_state_dict,
            vae=vae_state_dict,
            encoder=text_encoder_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: Flux2StateDicts, config: Flux2KleinPipelineConfig) -> "Flux2KleinPipeline":
        assert config.parallelism <= 1, "Flux2 doesn't support parallelism > 1"
        pipe = cls._from_state_dict(state_dicts, config)
        return pipe

    @classmethod
    def _from_state_dict(cls, state_dicts: Flux2StateDicts, config: Flux2KleinPipelineConfig) -> "Flux2KleinPipeline":
        init_device = "cpu" if config.offload_mode is not None else config.device
        if config.model_size == "4B":
            with open(Z_IMAGE_TEXT_ENCODER_CONFIG_FILE, "r", encoding="utf-8") as f:
                qwen3_config = Qwen3Config(**json.load(f))
            dit_config = {}
        else:
            with open(FLUX2_TEXT_ENCODER_8B_CONF_PATH, "r", encoding="utf-8") as f:
                qwen3_config = Qwen3Config(**json.load(f))
                state_dicts.encoder.pop("lm_head.weight")
            dit_config = {"guidance_embeds": False, "joint_attention_dim": 12288, "num_attention_heads": 32, "num_layers": 8, "num_single_layers": 24}
        text_encoder = Qwen3Model.from_state_dict(
            state_dicts.encoder, config=qwen3_config, device=init_device, dtype=config.encoder_dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(Z_IMAGE_TOKENIZER_CONF_PATH, local_files_only=True)
        vae = Flux2VAE.from_state_dict(state_dicts.vae, device=init_device, dtype=config.vae_dtype)

        with LoRAContext():
            dit = Flux2DiT.from_state_dict(
                state_dicts.model,
                device=("cpu" if config.use_fsdp else init_device),
                dtype=config.model_dtype,
                **dit_config,
            )
            if config.use_fp8_linear:
                enable_fp8_linear(dit)

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
        )
        pipe.eval()

        if config.offload_mode is not None:
            pipe.enable_cpu_offload(config.offload_mode, config.offload_to_disk)

        if config.model_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16
            pipe.enable_fp8_autocast(
                model_names=["dit"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.use_torch_compile:
            pipe.compile()

        return pipe

    def update_weights(self, state_dicts: Flux2StateDicts) -> None:
        self.update_component(self.dit, state_dicts.model, self.config.device, self.config.model_dtype)
        self.update_component(
            self.text_encoder, state_dicts.encoder, self.config.device, self.config.encoder_dtype
        )
        self.update_component(self.vae, state_dicts.vae, self.config.device, self.config.vae_dtype)

    def compile(self):
        if hasattr(self.dit, "compile_repeated_blocks"):
            self.dit.compile_repeated_blocks()

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        assert self.config.tp_degree is None or self.config.tp_degree == 1, (
            "load LoRA is not allowed when tensor parallel is enabled; "
            "set tp_degree=None or tp_degree=1 during pipeline initialization"
        )
        assert not (self.config.use_fsdp and fused), (
            "load fused LoRA is not allowed when fully sharded data parallel is enabled; "
            "either load LoRA with fused=False or set use_fsdp=False during pipeline initialization"
        )
        super().load_loras(lora_list, fused, save_original_weight)

    def unload_loras(self):
        if hasattr(self.dit, "unload_loras"):
            self.dit.unload_loras()
        self.noise_scheduler.restore_config()

    def apply_scheduler_config(self, scheduler_config: Dict):
        self.noise_scheduler.update_config(scheduler_config)

    def prepare_latents(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
    ):
        # Compute dynamic shift length for FLUX.2 scheduler
        dynamic_shift_len = (height // 16) * (width // 16)
        
        # Match original FLUX.2 scheduler parameters
        sigma_min = 1.0 / num_inference_steps
        sigma_max = 1.0
        
        sigmas, timesteps = self.noise_scheduler.schedule(
            num_inference_steps, 
            sigma_min=sigma_min, 
            sigma_max=sigma_max,
            mu=self._compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        )

        # Apply denoising strength by truncating the schedule
        if denoising_strength < 1.0:
            num_actual_steps = max(1, int(num_inference_steps * denoising_strength))
            sigmas = sigmas[:num_actual_steps + 1]
            timesteps = timesteps[:num_actual_steps]

        sigmas = sigmas.to(device=self.device, dtype=self.dtype)
        timesteps = timesteps.to(device=self.device, dtype=self.dtype)
        latents = latents.to(device=self.device, dtype=self.dtype)

        return latents, sigmas, timesteps

    def _compute_empirical_mu(self, image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 scheduler (matching original implementation)"""
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.device)

        # Forward pass through the model
        with torch.inference_mode():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Use outputs from intermediate layers (9, 18, 27) for Qwen3 (matching original behavior)
        hidden_states = output["hidden_states"] if isinstance(output, dict) else output.hidden_states
        out = torch.stack([hidden_states[k] for k in (9, 18, 27)], dim=1)
        out = out.to(dtype=self.dtype, device=self.device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        
        # Prepare text IDs
        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(self.device)
        
        return prompt_embeds, text_ids

    def prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def prepare_image_ids(self, height, width):
        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        image_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        image_ids = image_ids.unsqueeze(0).expand(1, -1, -1)

        return image_ids

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        embedded_guidance: float = 4.0,
        edit_latents: torch.Tensor = None,
        edit_image_ids: torch.Tensor = None,
    ):
        self.load_models_to_device(["dit"])
        
        # Handle edit images by concatenating latents and image IDs
        if edit_latents is not None and edit_image_ids is not None:
            latents = torch.concat([latents, edit_latents], dim=1)
            image_ids = torch.concat([image_ids, edit_image_ids], dim=1)
        
        embedded_guidance_tensor = torch.tensor([embedded_guidance], device=latents.device)
        
        noise_pred = self.dit(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=embedded_guidance_tensor,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
        )
        
        # Return only the original image sequence length if edit images were used
        if edit_latents is not None:
            noise_pred = noise_pred[:, :image_ids.shape[1] - edit_image_ids.shape[1]]
            
        return noise_pred

    def encode_edit_image(
        self,
        edit_image: Union[Image.Image, List[Image.Image]],
        edit_image_auto_resize: bool = True,
    ):
        """Encode edit image(s) to latents for FLUX.2 pipeline"""
        if edit_image is None:
            return None, None
            
        self.load_models_to_device(["vae"])
        
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]
            
        resized_edit_image, edit_latents = [], []
        for image in edit_image:
            # Preprocess
            if edit_image_auto_resize:
                image = self.edit_image_auto_resize(image)
            resized_edit_image.append(image)
            # Encode
            image_tensor = self.preprocess_image(image).to(dtype=self.dtype, device=self.device)
            latents = self.vae.encode(image_tensor)
            edit_latents.append(latents)
            
        edit_image_ids = self.process_edit_image_ids(edit_latents)
        edit_latents = torch.concat([rearrange(latents, "B C H W -> B (H W) C") for latents in edit_latents], dim=1)
        
        return edit_latents, edit_image_ids

    def edit_image_auto_resize(self, edit_image):
        """Auto resize edit image to optimal dimensions"""
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return self.crop_and_resize(edit_image, calculated_height, calculated_width)

    def crop_and_resize(self, image, target_height, target_width):
        """Crop and resize image to target dimensions"""
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def process_edit_image_ids(self, image_latents, scale=10):
        """Process image IDs for edit images"""
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        denoising_strength: float = 1.0,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        # Edit image parameters
        edit_image: Union[Image.Image, List[Image.Image]] = None,
        edit_image_auto_resize: bool = True,
    ):
        self.validate_image_size(height, width, multiple_of=16)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_embeds, text_ids = self.encode_prompt(prompt)
        if negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(negative_prompt)
        else:
            negative_prompt_embeds, negative_text_ids = None, None
        self.model_lifecycle_finish(["text_encoder"])

        # Encode edit images if provided
        edit_latents, edit_image_ids = None, None
        if edit_image is not None:
            edit_latents, edit_image_ids = self.encode_edit_image(edit_image, edit_image_auto_resize)
            if edit_latents is not None:
                edit_latents = edit_latents.to(device=self.device, dtype=self.dtype)
                edit_image_ids = edit_image_ids.to(device=self.device, dtype=self.dtype)

        # Generate initial noise
        noise = self.generate_noise((1, 128, height // 16, width // 16), seed=seed, device="cpu", dtype=self.dtype).to(
            device=self.device
        )
        noise = noise.reshape(1, 128, height//16 * width//16).permute(0, 2, 1)
        
        # Prepare latents with noise scheduling
        latents, sigmas, timesteps = self.prepare_latents(noise, num_inference_steps, denoising_strength, height, width)

        self.sampler.initialize(sigmas=sigmas)

        # Prepare image IDs
        image_ids = self.prepare_image_ids(height // 16, width // 16).to(self.device)

        # Denoising loop
        self.load_models_to_device(["dit"])
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            
            if cfg_scale > 1.0 and negative_prompt_embeds is not None:
                # CFG prediction
                latents_input = torch.cat([latents] * 2, dim=0)
                timestep_input = torch.cat([timestep] * 2, dim=0)
                prompt_embeds_input = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
                text_ids_input = torch.cat([text_ids, negative_text_ids], dim=0)
                image_ids_input = torch.cat([image_ids] * 2, dim=0)
                
                # Handle edit images for CFG
                edit_latents_input = None
                edit_image_ids_input = None
                if edit_latents is not None:
                    edit_latents_input = torch.cat([edit_latents] * 2, dim=0)
                    edit_image_ids_input = torch.cat([edit_image_ids] * 2, dim=0)
                
                noise_pred = self.predict_noise(
                    latents=latents_input,
                    timestep=timestep_input,
                    prompt_embeds=prompt_embeds_input,
                    text_ids=text_ids_input,
                    image_ids=image_ids_input,
                    embedded_guidance=embedded_guidance,
                    edit_latents=edit_latents_input,
                    edit_image_ids=edit_image_ids_input,
                )
                
                # Split predictions and apply CFG
                noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)
            else:
                # Non-CFG prediction
                noise_pred = self.predict_noise(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_ids=image_ids,
                    embedded_guidance=embedded_guidance,
                    edit_latents=edit_latents,
                    edit_image_ids=edit_image_ids,
                )
            
            latents = self.sampler.step(latents, noise_pred, i)
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")

        self.model_lifecycle_finish(["dit"])

        # Decode final latents
        self.load_models_to_device(["vae"])
        latents = rearrange(latents, "B (H W) C -> B C H W", H=height//16, W=width//16)
        vae_output = self.vae.decode(latents)
        image = self.vae_output_to_image(vae_output)
        
        # Offload all models
        self.load_models_to_device([])
        return image
