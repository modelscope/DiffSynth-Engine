import torch
import torch.distributed as dist
import math
import json
from typing import Callable, List, Dict, Tuple, Optional, Union
from tqdm import tqdm

from diffsynth_engine.configs import (
    ZImagePipelineConfig,
    ZImageStateDicts,
)
from diffsynth_engine.models.basic.lora import LoRAContext

from diffsynth_engine.models.z_image import (
    ZImageDiT,
    Qwen3Model,
    Qwen3Config,
)
from diffsynth_engine.tokenizers.qwen2 import Qwen2TokenizerFast
from diffsynth_engine.utils.constants import (
    Z_IMAGE_TEXT_ENCODER_CONFIG_FILE,
    Z_IMAGE_TOKENIZER_CONF_PATH,
)
from diffsynth_engine.models.flux import FluxVAEDecoder
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import calculate_shift
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model

logger = logging.get_logger(__name__)


class ZImageLoRAConverter(LoRAStateDictConverter):
    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            if "lora_A.weight" in key:
                lora_b_key = key.replace("lora_A.weight", "lora_B.weight")
                target_key = key.replace(".lora_A.weight", "").replace("diffusion_model.", "")

                if "attention.to_out.0" in target_key:
                    target_key = target_key.replace("attention.to_out.0", "attention.to_out")
                if "adaLN_modulation.0" in target_key:
                    target_key = target_key.replace("adaLN_modulation.0", "adaLN_modulation")

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

                if "attention.to_out.0" in target_key:
                    target_key = target_key.replace("attention.to_out.0", "attention.to_out")

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


class ZImagePipeline(BasePipeline):
    lora_converter = ZImageLoRAConverter()

    def __init__(
        self,
        config: ZImagePipelineConfig,
        tokenizer: Qwen2TokenizerFast,
        text_encoder: Qwen3Model,
        dit: ZImageDiT,
        vae_decoder: FluxVAEDecoder,
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
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0, use_dynamic_shifting=True)
        self.sampler = FlowMatchEulerSampler()
        self.tokenizer = tokenizer
        # Models
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae_decoder = vae_decoder

        self.model_names = ["text_encoder", "dit", "vae_decoder"]

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | ZImagePipelineConfig) -> "ZImagePipeline":
        if isinstance(model_path_or_config, str):
            config = ZImagePipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"Loading state dict from {config.model_path} ...")

        model_state_dict = cls.load_model_checkpoint(
            config.model_path, device="cpu", dtype=config.model_dtype, convert_dtype=False
        )

        if config.vae_path is None:
            config.vae_path = fetch_model(config.model_path, path="vae/diffusion_pytorch_model.safetensors")
        logger.info(f"Loading VAE from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        if config.encoder_path is None:
            config.encoder_path = fetch_model(config.model_path, path="text_encoder")
        logger.info(f"Loading Text Encoder from {config.encoder_path} ...")
        text_encoder_state_dict = cls.load_model_checkpoint(
            config.encoder_path, device="cpu", dtype=config.encoder_dtype
        )

        state_dicts = ZImageStateDicts(
            model=model_state_dict,
            vae=vae_state_dict,
            encoder=text_encoder_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: ZImageStateDicts, config: ZImagePipelineConfig) -> "ZImagePipeline":
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
    def _from_state_dict(cls, state_dicts: ZImageStateDicts, config: ZImagePipelineConfig) -> "ZImagePipeline":
        init_device = "cpu" if config.offload_mode is not None else config.device
        with open(Z_IMAGE_TEXT_ENCODER_CONFIG_FILE, "r", encoding="utf-8") as f:
            qwen3_config = Qwen3Config(**json.load(f))
        text_encoder = Qwen3Model.from_state_dict(
            state_dicts.encoder, config=qwen3_config, device=init_device, dtype=config.encoder_dtype
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(Z_IMAGE_TOKENIZER_CONF_PATH)
        vae_decoder = FluxVAEDecoder.from_state_dict(state_dicts.vae, device=init_device, dtype=config.vae_dtype)

        with LoRAContext():
            dit = ZImageDiT.from_state_dict(
                state_dicts.model,
                device=("cpu" if config.use_fsdp else init_device),
                dtype=config.model_dtype,
            )
            if config.use_fp8_linear:
                enable_fp8_linear(dit)

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            vae_decoder=vae_decoder,
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

    def update_weights(self, state_dicts: ZImageStateDicts) -> None:
        self.update_component(self.dit, state_dicts.model, self.config.device, self.config.model_dtype)
        self.update_component(
            self.text_encoder, state_dicts.encoder, self.config.device, self.config.encoder_dtype
        )
        self.update_component(self.vae_decoder, state_dicts.vae, self.config.device, self.config.vae_dtype)

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
        mu: float,
    ):
        sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps, mu=mu, sigma_min=0, sigma_max=1.0)

        sigmas = sigmas.to(device=self.device, dtype=self.dtype)
        timesteps = timesteps.to(device=self.device, dtype=self.dtype)
        latents = latents.to(device=self.device, dtype=self.dtype)

        return latents, sigmas, timesteps

    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
    ):
        if prompt is None:
            return None
        template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        txt = [template.format(prompt)]
        text_inputs = self.tokenizer(
            txt,
            max_length=max_sequence_length,
            padding_strategy="max_length",
        )

        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device).bool()
        # Encoder forward
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        prompt_embeds = outputs["hidden_states"][-2]
        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][attention_mask[i]])
        return embeddings_list

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: List[torch.Tensor],
        negative_prompt_emb: List[torch.Tensor],
        cfg_scale: float = 5.0,
        cfg_truncation: float = 1.0,
        cfg_normalization: float = 0.0,  # 0.0 means disabled
        batch_cfg: bool = False,
    ):
        t = timestep.expand(latents.shape[0])
        t = (1000 - t) / 1000
        progress = t[0].item()

        current_cfg_scale = cfg_scale
        if cfg_truncation <= 1.0 and progress > cfg_truncation:
            current_cfg_scale = 0.0

        do_cfg = current_cfg_scale > 0 and negative_prompt_emb is not None

        if not do_cfg:
            comb_pred = self.predict_noise(latents, t, prompt_emb)[0]
        else:
            if not batch_cfg:
                positive_noise_pred = self.predict_noise(latents, t, prompt_emb)[0]
                negative_noise_pred = self.predict_noise(latents, t, negative_prompt_emb)[0]
            else:
                latents_input = torch.cat([latents, latents], dim=0)
                t = torch.cat([t, t], dim=0)
                prompt_input = prompt_emb + negative_prompt_emb

                noise_pred = self.predict_noise(latents_input, t, prompt_input)

                positive_noise_pred, negative_noise_pred = noise_pred[0], noise_pred[1]

            comb_pred = positive_noise_pred + current_cfg_scale * (positive_noise_pred - negative_noise_pred)

            if cfg_normalization is not None and cfg_normalization > 0:
                cond_norm = torch.linalg.vector_norm(positive_noise_pred)
                new_norm = torch.linalg.vector_norm(comb_pred)
                max_allowed_norm = cond_norm * cfg_normalization
                new_norm = torch.where(new_norm < 1e-6, torch.ones_like(new_norm), new_norm)
                scale_factor = max_allowed_norm / new_norm
                scale_factor = torch.clamp(scale_factor, max=1.0)
                comb_pred = comb_pred * scale_factor

        comb_pred = -comb_pred.squeeze(1).unsqueeze(0)
        return comb_pred

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: List[torch.Tensor],
    ):
        self.load_models_to_device(["dit"])

        latents_list = list(latents.unsqueeze(2).unbind(dim=0))

        noise_pred = self.dit(
            image=latents_list,
            timestep=timestep,
            cap_feats=prompt_emb,
        )
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.validate_image_size(height, width, multiple_of=16)

        self.load_models_to_device(["text_encoder"])
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt), self.encode_prompt(negative_prompt)
        self.model_lifecycle_finish(["text_encoder"])

        noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device="cpu", dtype=self.dtype).to(
            device=self.device
        )
        image_seq_len = math.ceil(height // 16) * math.ceil(width // 16)

        mu = calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)

        latents, sigmas, timesteps = self.prepare_latents(noise, num_inference_steps, mu)

        self.sampler.initialize(sigmas=sigmas)

        self.load_models_to_device(["dit"])
        hide_progress = dist.is_initialized() and dist.get_rank() != 0

        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_embeds,
                negative_prompt_emb=negative_prompt_embeds,
                batch_cfg=self.config.batch_cfg,
                cfg_scale=cfg_scale,
                cfg_truncation=cfg_truncation,
                cfg_normalization=cfg_normalization,
            )
            latents = self.sampler.step(latents, noise_pred, i)
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")

        self.model_lifecycle_finish(["dit"])

        self.load_models_to_device(["vae_decoder"])
        vae_output = self.decode_image(latents)
        image = self.vae_output_to_image(vae_output)
        # Offload all models
        self.load_models_to_device([])
        return image
