import json
import torch
import math
from typing import Callable, List, Tuple, Optional, Union, Dict
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist

from diffsynth_engine.configs import QwenImagePipelineConfig, QwenImageStateDicts
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.models.qwen_image import (
    QwenImageDiT,
    QwenImageDiTFBCache,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLConfig,
)
from diffsynth_engine.models.qwen_image import QwenImageVAE
from diffsynth_engine.tokenizers import Qwen2TokenizerFast
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import calculate_shift
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.constants import (
    QWEN_IMAGE_TOKENIZER_CONF_PATH,
    QWEN_IMAGE_CONFIG_FILE,
    QWEN_IMAGE_VISION_CONFIG_FILE,
    QWEN_IMAGE_VAE_CONFIG_FILE,
)
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model


logger = logging.get_logger(__name__)


class QwenImageLoRAConverter(LoRAStateDictConverter):
    def _from_diffsynth(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            lora_a_suffix = None
            if "lora_A.default.weight" in key:
                lora_a_suffix = "lora_A.default.weight"
            elif "lora_A.weight" in key:
                lora_a_suffix = "lora_A.weight"

            if lora_a_suffix is None:
                continue

            lora_args = {}
            lora_args["down"] = param

            lora_b_suffix = lora_a_suffix.replace("lora_A", "lora_B")
            lora_args["up"] = lora_state_dict[origin_key.replace(lora_a_suffix, lora_b_suffix)]

            lora_args["rank"] = lora_args["up"].shape[1]
            alpha_key = origin_key.replace("lora_up", "lora_A").replace(lora_a_suffix, "alpha")

            if alpha_key in lora_state_dict:
                alpha = lora_state_dict[alpha_key]
            else:
                alpha = lora_args["rank"]
            lora_args["alpha"] = alpha

            key = key.replace(f".{lora_a_suffix}", "")

            if key.startswith("transformer") and "attn.to_out.0" in key:
                key = key.replace("attn.to_out.0", "attn.to_out")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._from_diffsynth(lora_state_dict)


class QwenImagePipeline(BasePipeline):
    lora_converter = QwenImageLoRAConverter()

    def __init__(
        self,
        config: QwenImagePipelineConfig,
        tokenizer: Qwen2TokenizerFast,
        encoder: Qwen2_5_VLForConditionalGeneration,
        dit: QwenImageDiT,
        vae: QwenImageVAE,
    ):
        super().__init__(
            vae_tiled=config.vae_tiled,
            vae_tile_size=config.vae_tile_size,
            vae_tile_stride=config.vae_tile_stride,
            device=config.device,
            dtype=config.model_dtype,
        )
        self.config = config
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        # sampler
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0, use_dynamic_shifting=True)
        self.sampler = FlowMatchEulerSampler()
        # models
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.dit = dit
        self.vae = vae
        self.model_names = [
            "encoder",
            "dit",
            "vae",
        ]

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | QwenImagePipelineConfig) -> "QwenImagePipeline":
        if isinstance(model_path_or_config, str):
            config = QwenImagePipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        return cls.from_state_dict(QwenImageStateDicts(), config)

    @classmethod
    def from_state_dict(cls, state_dicts: QwenImageStateDicts, config: QwenImagePipelineConfig) -> "QwenImagePipeline":
        if state_dicts.model is None:
            if config.model_path is None:
                raise ValueError("`model_path` cannot be empty")
            logger.info(f"loading state dict from {config.model_path} ...")
            state_dicts.model = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        if state_dicts.vae is None:
            if config.vae_path is None:
                config.vae_path = fetch_model(
                    "MusePublic/Qwen-image", revision="v1", path="vae/diffusion_pytorch_model.safetensors"
                )
            logger.info(f"loading state dict from {config.vae_path} ...")
            state_dicts.vae = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        if state_dicts.encoder is None:
            if config.encoder_path is None:
                config.encoder_path = fetch_model(
                    "MusePublic/Qwen-image",
                    revision="v1",
                    path=[
                        "text_encoder/model-00001-of-00004.safetensors",
                        "text_encoder/model-00002-of-00004.safetensors",
                        "text_encoder/model-00003-of-00004.safetensors",
                        "text_encoder/model-00004-of-00004.safetensors",
                    ],
                )
            logger.info(f"loading state dict from {config.encoder_path} ...")
            state_dicts.encoder = cls.load_model_checkpoint(
                config.encoder_path, device="cpu", dtype=config.encoder_dtype
            )

        init_device = "cpu" if config.parallelism > 1 or config.offload_mode is not None else config.device
        tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_IMAGE_TOKENIZER_CONF_PATH)
        with open(QWEN_IMAGE_VISION_CONFIG_FILE, "r") as f:
            vision_config = Qwen2_5_VLVisionConfig(**json.load(f))
        with open(QWEN_IMAGE_CONFIG_FILE, "r") as f:
            text_config = Qwen2_5_VLConfig(**json.load(f))
        encoder = Qwen2_5_VLForConditionalGeneration.from_state_dict(
            state_dicts.encoder,
            vision_config=vision_config,
            config=text_config,
            device=init_device,
            dtype=config.encoder_dtype,
        )
        with open(QWEN_IMAGE_VAE_CONFIG_FILE, "r") as f:
            vae_config = json.load(f)
        vae = QwenImageVAE.from_state_dict(
            state_dicts.vae, config=vae_config, device=init_device, dtype=config.vae_dtype
        )

        with LoRAContext():
            attn_kwargs = {
                "attn_impl": config.dit_attn_impl,
                "sparge_smooth_k": config.sparge_smooth_k,
                "sparge_cdfthreshd": config.sparge_cdfthreshd,
                "sparge_simthreshd1": config.sparge_simthreshd1,
                "sparge_pvthreshd": config.sparge_pvthreshd,
            }
            if config.use_fbcache:
                dit = QwenImageDiTFBCache.from_state_dict(
                    state_dicts.model,
                    device=init_device,
                    dtype=config.model_dtype,
                    attn_kwargs=attn_kwargs,
                    relative_l1_threshold=config.fbcache_relative_l1_threshold,
                )
            else:
                dit = QwenImageDiT.from_state_dict(
                    state_dicts.model,
                    device=init_device,
                    dtype=config.model_dtype,
                    attn_kwargs=attn_kwargs,
                )
            if config.use_fp8_linear:
                enable_fp8_linear(dit)

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            encoder=encoder,
            dit=dit,
            vae=vae,
        )
        pipe.eval()

        if config.offload_mode is not None:
            pipe.enable_cpu_offload(config.offload_mode, config.offload_to_disk)
        
        if config.model_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16  # compute dtype
            pipe.enable_fp8_autocast(
                model_names=["dit"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.encoder_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16  # compute dtype
            pipe.enable_fp8_autocast(
                model_names=["encoder"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.parallelism > 1:
            pipe = ParallelWrapper(
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

    def compile(self):
        self.dit.compile()

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
        self.dit.unload_loras()

    def prepare_latents(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        mu: float,
    ):
        sigmas, timesteps = self.noise_scheduler.schedule(
            num_inference_steps, mu=mu, sigma_min=1 / num_inference_steps, sigma_max=1.0
        )
        init_latents = latents.clone()
        sigmas, timesteps = (
            sigmas.to(device=self.device, dtype=self.dtype),
            timesteps.to(device=self.device, dtype=self.dtype),
        )
        init_latents, latents = (
            init_latents.to(device=self.device, dtype=self.dtype),
            latents.to(device=self.device, dtype=self.dtype),
        )
        return init_latents, latents, sigmas, timesteps

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt)
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        texts = [template.format(txt) for txt in prompt]
        outputs = self.tokenizer(texts, max_length=max_sequence_length + drop_idx)
        input_ids, attention_mask = outputs["input_ids"].to(self.device), outputs["attention_mask"].to(self.device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["hidden_states"]
        prompt_embeds = hidden_states[:, drop_idx:]
        prompt_embeds_mask = attention_mask[:, drop_idx:]
        seq_len = prompt_embeds.shape[1]

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool = False,
    ):
        if cfg_scale <= 1.0 or negative_prompt_emb is None:
            return self.predict_noise(
                latents,
                timestep,
                prompt_emb,
                prompt_embeds_mask,
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            h, w = latents.shape[-2:]
            positive_noise_pred = self.predict_noise(
                latents,
                timestep,
                prompt_emb,
                prompt_embeds_mask,
            )
            negative_noise_pred = self.predict_noise(
                latents,
                timestep,
                negative_prompt_emb,
                negative_prompt_embeds_mask,
            )
            comb_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            cond_norm = torch.norm(self.dit.patchify(positive_noise_pred), dim=-1, keepdim=True)
            noise_norm = torch.norm(self.dit.patchify(comb_pred), dim=-1, keepdim=True)
            noise_pred = self.dit.unpatchify(self.dit.patchify(comb_pred) * (cond_norm / noise_norm), h, w)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            bs, _, h, w = latents.shape
            prompt_emb = torch.cat([prompt_emb, negative_prompt_emb], dim=0)
            prompt_embeds_mask = torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            noise_pred = self.predict_noise(
                latents,
                timestep,
                prompt_emb,
                prompt_embeds_mask,
            )
            positive_noise_pred, negative_noise_pred = noise_pred[:bs], noise_pred[bs:]
            comb_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            cond_norm = torch.norm(self.dit.patchify(positive_noise_pred), dim=-1, keepdim=True)
            noise_norm = torch.norm(self.dit.patchify(comb_pred), dim=-1, keepdim=True)
            noise_pred = self.dit.unpatchify(self.dit.patchify(comb_pred) * (cond_norm / noise_norm), h, w)
            return noise_pred

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
    ):
        self.load_models_to_device(["dit"])

        noise_pred = self.dit(
            image=latents,
            text=prompt_emb,
            timestep=timestep,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1),
        )
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,  # true cfg
        height: int = 1328,
        width: int = 1328,
        num_inference_steps: int = 50,
        seed: int | None = None,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device="cpu", dtype=self.dtype).to(
            device=self.device
        )
        # dynamic shift
        image_seq_len = math.ceil(height // 16) * math.ceil(width // 16)
        mu = calculate_shift(image_seq_len, max_shift=0.9, max_seq_len=8192)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(noise, num_inference_steps, mu)
        # Initialize sampler
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas)

        self.load_models_to_device(["encoder"])
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, 1, 4096)
        if cfg_scale > 1.0 and negative_prompt != "":
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(negative_prompt, 1, 4096)
        else:
            negative_prompt_embeds, negative_prompt_embeds_mask = None, None
        self.model_lifecycle_finish(["encoder"])

        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_embeds,
                negative_prompt_emb=negative_prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                cfg_scale=cfg_scale,
                batch_cfg=self.config.batch_cfg,
            )
            # Denoise
            latents = self.sampler.step(latents, noise_pred, i)
            # UI
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")
        self.model_lifecycle_finish(["dit"])
        # Decode image
        self.load_models_to_device(["vae"])
        latents = rearrange(latents, "B C H W -> B C 1 H W")
        vae_output = rearrange(
            self.vae.decode(
                latents.to(self.vae.model.encoder.conv1.weight.dtype), device=self.vae.model.encoder.conv1.weight.device
            )[0],
            "C B H W -> B C H W",
        )
        image = self.vae_output_to_image(vae_output)
        # Offload all models
        self.model_lifecycle_finish(["vae"])        
        self.load_models_to_device([])
        return image
