import json
import torch
import torch.distributed as dist
import math
import sys
from typing import Callable, List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from einops import rearrange
from PIL import Image

from diffsynth_engine.configs import (
    QwenImagePipelineConfig,
    QwenImageStateDicts,
    QwenImageControlNetParams,
    QwenImageControlType,
)
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.models.qwen_image import (
    QwenImageDiT,
    QwenImageDiTFBCache,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLConfig,
)
from diffsynth_engine.models.qwen_image import QwenImageVAE
from diffsynth_engine.tokenizers import Qwen2TokenizerFast, Qwen2VLProcessor
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import calculate_shift, pad_and_concat
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.constants import (
    QWEN_IMAGE_TOKENIZER_CONF_PATH,
    QWEN_IMAGE_PROCESSOR_CONFIG_FILE,
    QWEN_IMAGE_CONFIG_FILE,
    QWEN_IMAGE_VISION_CONFIG_FILE,
    QWEN_IMAGE_VAE_CONFIG_FILE,
)
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.flag import NUNCHAKU_AVAILABLE


logger = logging.get_logger(__name__)



class QwenImageLoRAConverter(LoRAStateDictConverter):
    def _from_diffsynth(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            lora_a_suffix = None
            if "lora_A.default.weight" in key:
                lora_a_suffix = "lora_A.default.weight"
                lora_b_suffix = "lora_B.default.weight"
            elif "lora_A.weight" in key:
                lora_a_suffix = "lora_A.weight"
                lora_b_suffix = "lora_B.weight"
            elif "lora_down.weight" in key:
                lora_a_suffix = "lora_down.weight"
                lora_b_suffix = "lora_up.weight"

            if lora_a_suffix is None:
                continue

            lora_args = {}
            lora_args["down"] = param
            lora_args["up"] = lora_state_dict[origin_key.replace(lora_a_suffix, lora_b_suffix)]

            lora_args["rank"] = lora_args["up"].shape[1]
            alpha_key = origin_key.replace(lora_a_suffix, "alpha")

            if alpha_key in lora_state_dict:
                alpha = lora_state_dict[alpha_key]
            else:
                alpha = lora_args["rank"]
            lora_args["alpha"] = alpha

            key = key.replace(f".{lora_a_suffix}", "")
            key = key.replace("base_model.model.", "")
            key = key.replace("transformer.", "")

            if key.startswith("transformer") and "attn.to_out.0" in key:
                key = key.replace("attn.to_out.0", "attn.to_out")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            lora_a_suffix = None
            if "lora_A.weight" in key:
                lora_a_suffix = "lora_A.weight"
                lora_b_suffix = "lora_B.weight"

            if lora_a_suffix is None:
                continue

            lora_args = {}
            lora_args["down"] = param
            lora_args["up"] = lora_state_dict[origin_key.replace(lora_a_suffix, lora_b_suffix)]
            lora_args["rank"] = lora_args["up"].shape[1]
            alpha_key = origin_key.replace(lora_a_suffix, "alpha")

            if alpha_key in lora_state_dict:
                alpha = lora_state_dict[alpha_key]
            else:
                alpha = lora_args["rank"]
            lora_args["alpha"] = alpha

            key = key.replace(f".{lora_a_suffix}", "")
            key = key.replace("diffusion_model.", "")

            if key.startswith("transformer") and "attn.to_out.0" in key:
                key = key.replace("attn.to_out.0", "attn.to_out")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if key.startswith("diffusion_model."):
            return self._from_diffusers(lora_state_dict)
        else:
            return self._from_diffsynth(lora_state_dict)


class QwenImagePipeline(BasePipeline):
    lora_converter = QwenImageLoRAConverter()

    def __init__(
        self,
        config: QwenImagePipelineConfig,
        tokenizer: Qwen2TokenizerFast,
        processor: Qwen2VLProcessor,
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
        # qwen image
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        # qwen image edit
        self.edit_system_prompt = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        self.edit_prompt_template_encode = (
            "<|im_start|>system\n"
            + self.edit_system_prompt
            + "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        )
        # qwen image edit plus
        self.edit_plus_prompt_template_encode = (
            "<|im_start|>system\n"
            + self.edit_system_prompt
            + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )

        self.edit_prompt_template_encode_start_idx = 64

        # sampler
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0, use_dynamic_shifting=True)
        self.sampler = FlowMatchEulerSampler()
        # models
        self.tokenizer = tokenizer
        self.processor = processor
        self.encoder = encoder
        self.dit = dit
        self.vae = vae
        self.model_names = [
            "encoder",
            "dit",
            "vae",
        ]

    @classmethod
    def _setup_nunchaku_config(
        cls, model_state_dict: Dict[str, torch.Tensor], config: QwenImagePipelineConfig
    ) -> QwenImagePipelineConfig:
        is_nunchaku_model = any("qweight" in key for key in model_state_dict)

        if is_nunchaku_model:
            logger.info("Nunchaku quantized model detected. Configuring for nunchaku.")
            config.use_nunchaku = True
            config.nunchaku_rank = model_state_dict["transformer_blocks.0.img_mlp.net.0.proj.proj_up"].shape[1]

            if "transformer_blocks.0.img_mod.1.qweight" in model_state_dict:
                config.use_nunchaku_awq = True
                logger.info("Enable nunchaku AWQ.")
            else:
                config.use_nunchaku_awq = False
                logger.info("Disable nunchaku AWQ.")

            if "transformer_blocks.0.attn.to_qkv.qweight" in model_state_dict:
                config.use_nunchaku_attn = True
                logger.info("Enable nunchaku attention quantization.")
            else:
                config.use_nunchaku_attn = False
                logger.info("Disable nunchaku attention quantization.")
        
        else:
            config.use_nunchaku = False

        return config

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | QwenImagePipelineConfig) -> "QwenImagePipeline":
        if isinstance(model_path_or_config, str):
            config = QwenImagePipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"loading state dict from {config.model_path} ...")
        model_state_dict = cls.load_model_checkpoint(
            config.model_path, device="cpu", dtype=config.model_dtype, convert_dtype=False
        )

        config = cls._setup_nunchaku_config(model_state_dict, config)

        # for svd quant model fp4/int4 linear layers, do not convert dtype here
        if not config.use_nunchaku:
            for key, value in model_state_dict.items():
                model_state_dict[key] = value.to(config.model_dtype)

        if config.vae_path is None:
            config.vae_path = fetch_model(
                "MusePublic/Qwen-image", revision="v1", path="vae/diffusion_pytorch_model.safetensors"
            )
        logger.info(f"loading state dict from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        encoder_state_dict = None
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
        if config.load_encoder:
            logger.info(f"loading state dict from {config.encoder_path} ...")
            encoder_state_dict = cls.load_model_checkpoint(
                config.encoder_path, device="cpu", dtype=config.encoder_dtype
            )

        state_dicts = QwenImageStateDicts(
            model=model_state_dict,
            vae=vae_state_dict,
            encoder=encoder_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: QwenImageStateDicts, config: QwenImagePipelineConfig) -> "QwenImagePipeline":
        config = cls._setup_nunchaku_config(state_dicts.model, config)

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
    def _from_state_dict(cls, state_dicts: QwenImageStateDicts, config: QwenImagePipelineConfig) -> "QwenImagePipeline":
        init_device = "cpu" if config.offload_mode is not None else config.device
        tokenizer, processor, encoder = None, None, None
        if config.load_encoder:
            tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_IMAGE_TOKENIZER_CONF_PATH)
            processor = Qwen2VLProcessor.from_pretrained(
                tokenizer_config_path=QWEN_IMAGE_TOKENIZER_CONF_PATH,
                image_processor_config_path=QWEN_IMAGE_PROCESSOR_CONFIG_FILE,
            )
            with open(QWEN_IMAGE_VISION_CONFIG_FILE, "r", encoding="utf-8") as f:
                vision_config = Qwen2_5_VLVisionConfig(**json.load(f))
            with open(QWEN_IMAGE_CONFIG_FILE, "r", encoding="utf-8") as f:
                text_config = Qwen2_5_VLConfig(**json.load(f))
            encoder = Qwen2_5_VLForConditionalGeneration.from_state_dict(
                state_dicts.encoder,
                vision_config=vision_config,
                config=text_config,
                device=("cpu" if config.use_fsdp else init_device),
                dtype=config.encoder_dtype,
            )

        with open(QWEN_IMAGE_VAE_CONFIG_FILE, "r", encoding="utf-8") as f:
            vae_config = json.load(f)
        vae = QwenImageVAE.from_state_dict(
            state_dicts.vae, config=vae_config, device=init_device, dtype=config.vae_dtype
        )

        with LoRAContext():
            if config.use_fbcache:
                dit = QwenImageDiTFBCache.from_state_dict(
                    state_dicts.model,
                    device=("cpu" if config.use_fsdp else init_device),
                    dtype=config.model_dtype,
                    relative_l1_threshold=config.fbcache_relative_l1_threshold,
                )
            elif config.use_nunchaku:
                if not NUNCHAKU_AVAILABLE:
                    from diffsynth_engine.utils.flag import NUNCHAKU_IMPORT_ERROR
                    raise ImportError(NUNCHAKU_IMPORT_ERROR)

                from diffsynth_engine.models.qwen_image import QwenImageDiTNunchaku
                from diffsynth_engine.models.basic.lora_nunchaku import patch_nunchaku_model_for_lora

                dit = QwenImageDiTNunchaku.from_state_dict(
                    state_dicts.model,
                    device=init_device,
                    dtype=config.model_dtype,
                    use_nunchaku_awq=config.use_nunchaku_awq,
                    use_nunchaku_attn=config.use_nunchaku_attn,
                    nunchaku_rank=config.nunchaku_rank,
                )
                patch_nunchaku_model_for_lora(dit)
            else:
                dit = QwenImageDiT.from_state_dict(
                    state_dicts.model,
                    device=("cpu" if config.use_fsdp else init_device),
                    dtype=config.model_dtype,
                )
            if config.use_fp8_linear and not config.use_nunchaku:
                enable_fp8_linear(dit)

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
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

        if config.use_torch_compile:
            pipe.compile()
        return pipe

    def update_weights(self, state_dicts: QwenImageStateDicts) -> None:
        self.update_component(self.dit, state_dicts.model, self.config.device, self.config.model_dtype)
        self.update_component(self.encoder, state_dicts.encoder, self.config.device, self.config.encoder_dtype)
        self.update_component(self.vae, state_dicts.vae, self.config.device, self.config.vae_dtype)

    def compile(self):
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
        prompt_emb = hidden_states[:, drop_idx:]
        prompt_emb_mask = attention_mask[:, drop_idx:]
        seq_len = prompt_emb.shape[1]

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_emb = prompt_emb.repeat(1, num_images_per_prompt, 1)
        prompt_emb = prompt_emb.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_emb_mask = prompt_emb_mask.repeat(1, num_images_per_prompt, 1)
        prompt_emb_mask = prompt_emb_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_emb, prompt_emb_mask

    def encode_prompt_with_image(
        self,
        prompt: Union[str, List[str]],
        vae_image: List[torch.Tensor],
        condition_image: List[torch.Tensor],  # edit plus
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        is_edit_plus: bool = True,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt)
        template = self.edit_prompt_template_encode
        drop_idx = self.edit_prompt_template_encode_start_idx
        if not is_edit_plus:
            template = self.edit_prompt_template_encode
            texts = [template.format(txt) for txt in prompt]
            image = vae_image
        else:
            template = self.edit_plus_prompt_template_encode
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(condition_image))])
            texts = [template.format(img_prompt + e) for e in prompt]
            image = condition_image

        model_inputs = self.processor(text=texts, images=image, max_length=max_sequence_length + drop_idx)
        input_ids, attention_mask, pixel_values, image_grid_thw = (
            model_inputs["input_ids"].to(self.device),
            model_inputs["attention_mask"].to(self.device),
            model_inputs["pixel_values"].to(self.device),
            model_inputs["image_grid_thw"].to(self.device),
        )
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        hidden_states = outputs["hidden_states"]
        prompt_emb = hidden_states[:, drop_idx:]
        prompt_emb_mask = attention_mask[:, drop_idx:]
        seq_len = prompt_emb.shape[1]

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_emb = prompt_emb.repeat(1, num_images_per_prompt, 1)
        prompt_emb = prompt_emb.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_emb_mask = prompt_emb_mask.repeat(1, num_images_per_prompt, 1)
        prompt_emb_mask = prompt_emb_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_emb, prompt_emb_mask

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        negative_prompt_emb_mask: torch.Tensor,
        # in_context
        context_latents: torch.Tensor = None,
        # eligen
        entity_prompt_embs: Optional[List[torch.Tensor]] = None,
        entity_prompt_emb_masks: Optional[List[torch.Tensor]] = None,
        negative_entity_prompt_embs: Optional[List[torch.Tensor]] = None,
        negative_entity_prompt_emb_masks: Optional[List[torch.Tensor]] = None,
        entity_masks: Optional[List[torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        batch_cfg: bool = False,
    ):
        if cfg_scale <= 1.0 or negative_prompt_emb is None:
            return self.predict_noise(
                latents,
                image_latents,
                timestep,
                prompt_emb,
                prompt_emb_mask,
                context_latents=context_latents,
                entity_prompt_embs=entity_prompt_embs,
                entity_prompt_emb_masks=entity_prompt_emb_masks,
                entity_masks=entity_masks,
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            h, w = latents.shape[-2:]
            positive_noise_pred = self.predict_noise(
                latents,
                image_latents,
                timestep,
                prompt_emb,
                prompt_emb_mask,
                context_latents=context_latents,
                entity_prompt_embs=entity_prompt_embs,
                entity_prompt_emb_masks=entity_prompt_emb_masks,
                entity_masks=entity_masks,
            )
            negative_noise_pred = self.predict_noise(
                latents,
                image_latents,
                timestep,
                negative_prompt_emb,
                negative_prompt_emb_mask,
                context_latents=context_latents,
                entity_prompt_embs=negative_entity_prompt_embs,
                entity_prompt_emb_masks=negative_entity_prompt_emb_masks,
                entity_masks=entity_masks,
            )
            comb_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            cond_norm = torch.norm(self.dit.patchify(positive_noise_pred), dim=-1, keepdim=True)
            noise_norm = torch.norm(self.dit.patchify(comb_pred), dim=-1, keepdim=True)
            noise_pred = self.dit.unpatchify(self.dit.patchify(comb_pred) * (cond_norm / noise_norm), h, w)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            bs, _, h, w = latents.shape
            prompt_emb = pad_and_concat(prompt_emb, negative_prompt_emb)
            prompt_emb_mask = pad_and_concat(prompt_emb_mask, negative_prompt_emb_mask)
            if entity_prompt_embs is not None:
                entity_prompt_embs = [
                    torch.cat([x, y], dim=0) for x, y in zip(entity_prompt_embs, negative_entity_prompt_embs)
                ]
                entity_prompt_emb_masks = [
                    torch.cat([x, y], dim=0) for x, y in zip(entity_prompt_emb_masks, negative_entity_prompt_emb_masks)
                ]
                entity_masks = [torch.cat([mask, mask], dim=0) for mask in entity_masks]
            latents = torch.cat([latents, latents], dim=0)
            if image_latents is not None:
                image_latents = [torch.cat([image_latent, image_latent], dim=0) for image_latent in image_latents]
            if context_latents is not None:
                context_latents = torch.cat([context_latents, context_latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            noise_pred = self.predict_noise(
                latents,
                image_latents,
                timestep,
                prompt_emb,
                prompt_emb_mask,
                context_latents=context_latents,
                entity_prompt_embs=entity_prompt_embs,
                entity_prompt_emb_masks=entity_prompt_emb_masks,
                entity_masks=entity_masks,
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
        image_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        # in_context
        context_latents: torch.Tensor = None,
        # eligen
        entity_prompt_embs: Optional[List[torch.Tensor]] = None,
        entity_prompt_emb_masks: Optional[List[torch.Tensor]] = None,
        entity_masks: Optional[List[torch.Tensor]] = None,
    ):
        self.load_models_to_device(["dit"])
        attn_kwargs = self.get_attn_kwargs(latents)
        noise_pred = self.dit(
            image=latents,
            edit=image_latents,
            timestep=timestep,
            text=prompt_emb,
            text_seq_lens=prompt_emb_mask.sum(dim=1),
            context_latents=context_latents,
            entity_text=entity_prompt_embs,
            entity_seq_lens=[mask.sum(dim=1) for mask in entity_prompt_emb_masks] if entity_prompt_emb_masks else None,
            entity_masks=entity_masks,
            attn_kwargs=attn_kwargs,
        )
        return noise_pred

    def prepare_image_latents(self, input_image: Image.Image):
        image = self.preprocess_image(input_image).to(dtype=self.config.vae_dtype)
        image = image.unsqueeze(2)
        image_latents = self.vae.encode(
            image,
            device=self.device,
            tiled=self.vae_tiled,
            tile_size=self.vae_tile_size,
            tile_stride=self.vae_tile_stride,
        )
        image_latents = image_latents.squeeze(2).to(device=self.device)
        return image_latents

    def prepare_eligen(self, entity_prompts, entity_masks, width, height):
        entity_masks = [mask.resize((width // 8, height // 8), resample=Image.NEAREST) for mask in entity_masks]
        entity_masks = [self.preprocess_image(mask).mean(dim=1, keepdim=True) > 0 for mask in entity_masks]
        entity_masks = [mask.to(device=self.device, dtype=self.dtype) for mask in entity_masks]
        prompt_embs, prompt_emb_masks = [], []
        negative_prompt_embs, negative_prompt_emb_masks = [], []
        for entity_prompt in entity_prompts:
            prompt_emb, prompt_emb_mask = self.encode_prompt(entity_prompt, 1, 512)
            prompt_embs.append(prompt_emb)
            prompt_emb_masks.append(prompt_emb_mask)
            negative_prompt_embs.append(torch.zeros_like(prompt_emb))
            negative_prompt_emb_masks.append(torch.zeros_like(prompt_emb_mask))
        return prompt_embs, prompt_emb_masks, negative_prompt_embs, negative_prompt_emb_masks, entity_masks

    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        # single image for edit, list for edit plus(QwenImageEdit2509)
        input_image: List[Image.Image] | Image.Image | None = None,
        cfg_scale: float = 4.0,  # true cfg
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        seed: int | None = None,
        controlnet_params: List[QwenImageControlNetParams] | QwenImageControlNetParams = [],
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
        # eligen
        entity_prompts: Optional[List[str]] = None,
        entity_masks: Optional[List[Image.Image]] = None,
    ):
        assert (height is None) == (width is None), "height and width should be set together"
        is_edit_plus = isinstance(input_image, list)

        if input_image is not None:
            if not isinstance(input_image, list):
                input_image = [input_image]
            condition_images = []
            vae_images = []
            for img in input_image:
                img_width, img_height = img.size
                condition_width, condition_height = self.calculate_dimensions(384 * 384, img_width / img_height)
                vae_width, vae_height = self.calculate_dimensions(1024 * 1024, img_width / img_height)
                condition_images.append(img.resize((condition_width, condition_height), Image.LANCZOS))
                vae_images.append(img.resize((vae_width, vae_height), Image.LANCZOS))
            if width is None and height is None:
                width, height = vae_images[-1].size

        if width is None and height is None:
            width, height = 1328, 1328
        self.validate_image_size(height, width, minimum=64, multiple_of=16)

        if not isinstance(controlnet_params, list):
            controlnet_params = [controlnet_params]

        context_latents = None
        for param in controlnet_params:
            self.load_lora(param.model, param.scale, fused=False, save_original_weight=False)
            if param.control_type == QwenImageControlType.in_context:
                width, height = param.image.size
                self.validate_image_size(height, width, minimum=64, multiple_of=16)
                context_latents = self.prepare_image_latents(param.image.resize((width, height), Image.LANCZOS))

        noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device="cpu", dtype=self.dtype).to(
            device=self.device
        )
        # dynamic shift
        image_seq_len = math.ceil(height // 16) * math.ceil(width // 16)
        mu = calculate_shift(image_seq_len, max_shift=0.9, max_seq_len=8192)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(noise, num_inference_steps, mu)
        # Initialize sampler
        self.sampler.initialize(sigmas=sigmas)

        self.load_models_to_device(["vae"])
        if input_image:
            image_latents = [self.prepare_image_latents(img) for img in vae_images]
        else:
            image_latents = None

        self.load_models_to_device(["encoder"])
        if image_latents is not None:
            prompt_emb, prompt_emb_mask = self.encode_prompt_with_image(
                prompt, vae_images, condition_images, 1, 4096, is_edit_plus
            )
            if cfg_scale > 1.0 and negative_prompt != "":
                negative_prompt_emb, negative_prompt_emb_mask = self.encode_prompt_with_image(
                    negative_prompt, vae_images, condition_images, 1, 4096, is_edit_plus
                )
            else:
                negative_prompt_emb, negative_prompt_emb_mask = None, None
        else:
            prompt_emb, prompt_emb_mask = self.encode_prompt(prompt, 1, 4096)
            if cfg_scale > 1.0 and negative_prompt != "":
                negative_prompt_emb, negative_prompt_emb_mask = self.encode_prompt(negative_prompt, 1, 4096)
            else:
                negative_prompt_emb, negative_prompt_emb_mask = None, None

        entity_prompt_embs, entity_prompt_emb_masks = None, None
        negative_entity_prompt_embs, negative_entity_prompt_emb_masks = None, None
        if entity_prompts is not None and entity_masks is not None:
            assert len(entity_prompts) == len(entity_masks), "entity_prompts and entity_masks must have the same length"
            (
                entity_prompt_embs,
                entity_prompt_emb_masks,
                negative_entity_prompt_embs,
                negative_entity_prompt_emb_masks,
                entity_masks,
            ) = self.prepare_eligen(entity_prompts, entity_masks, width, height)

        self.model_lifecycle_finish(["encoder"])

        self.load_models_to_device(["dit"])
        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                image_latents=image_latents,
                timestep=timestep,
                prompt_emb=prompt_emb,
                negative_prompt_emb=negative_prompt_emb,
                prompt_emb_mask=prompt_emb_mask,
                negative_prompt_emb_mask=negative_prompt_emb_mask,
                context_latents=context_latents,
                entity_prompt_embs=entity_prompt_embs,
                entity_prompt_emb_masks=entity_prompt_emb_masks,
                negative_entity_prompt_embs=negative_entity_prompt_embs,
                negative_entity_prompt_emb_masks=negative_entity_prompt_emb_masks,
                entity_masks=entity_masks,
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
                latents.to(self.vae.model.encoder.conv1.weight.dtype),
                device=self.vae.model.encoder.conv1.weight.device,
                tiled=self.vae_tiled,
                tile_size=self.vae_tile_size,
                tile_stride=self.vae_tile_stride,
            )[0],
            "C B H W -> B C H W",
        )
        image = self.vae_output_to_image(vae_output)
        # Offload all models
        self.model_lifecycle_finish(["vae"])
        self.load_models_to_device([])
        return image
