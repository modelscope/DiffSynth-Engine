from typing import Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Callable, List, Optional
from tqdm import tqdm

from diffsynth_engine.configs import ACEStepPipelineConfig, ACEStateDicts
from diffsynth_engine.models.ace_step.ace_dit import ACEStepDiT
from diffsynth_engine.models.ace_step.vae.music_dcae import MusicDCAE
from diffsynth_engine.models.ace_step.vae.dcae import DCAE
from diffsynth_engine.models.ace_step.vae.hifi_gan import ADaMoSHiFiGANV1
from diffsynth_engine.models.ace_step.lyric_tokenizer.lyric_tokenizer import (
    VoiceBpeTokenizer,
    SUPPORT_LANGUAGES,
    structure_pattern,
)
from diffsynth_engine.models.ace_step.lyric_tokenizer.lang_segment import LangSegment, default
from diffsynth_engine.models.ace_step.ace_text_encoder import ACETextEncoder
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.algorithm.noise_scheduler.flow_match import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils import logging


logger = logging.get_logger(__name__)


def fwd_with_temperature(
    inputs, model_fwd_func, get_hooked_layer_func, layer_start_idx, layer_end_idx, temperature=0.01
):
    def hook(module, input, output):
        output[:] *= temperature
        return output

    handlers = []
    for i in range(layer_start_idx, layer_end_idx):
        layer = get_hooked_layer_func(i)
        if isinstance(layer, list):
            for sub_layer in layer:
                handler = sub_layer.register_forward_hook(hook)
                handlers.append(handler)
        else:
            handler = layer.register_forward_hook(hook)
            handlers.append(handler)

    with torch.no_grad():
        prompt_emb = model_fwd_func(**inputs)

    for handler in handlers:
        handler.remove()
    return prompt_emb


class MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
    dims=[-1, -2],
):
    if v0.device.type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()

    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(v0), v0_orthogonal.to(v0)


def apg_forward(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=[-1, -2],
):
    diff = pred_cond - pred_uncond
    momentum_buffer.update(diff)
    diff = momentum_buffer.running_average

    diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
    scale_factor = torch.minimum(torch.ones_like(diff), norm_threshold / diff_norm)
    diff *= scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


class ACEStepMusicPipeline(BasePipeline):
    def __init__(
        self,
        config: ACEStepPipelineConfig,
        tokenizer: WanT5Tokenizer,
        text_encoder: ACETextEncoder,
        dit: ACEStepDiT,
        vae: MusicDCAE,
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
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0)
        self.sampler = FlowMatchEulerSampler()
        # models
        self.lyric_tokenizer = VoiceBpeTokenizer()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae
        self.model_names = ["text_encoder", "dit", "vae"]
        # language segment
        self.lang_segment = LangSegment()
        self.lang_segment.setfilters(default)

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        # assert self.config.tp_degree is None or self.config.tp_degree == 1, (
        #     "load LoRA is not allowed when tensor parallel is enabled; "
        #     "set tp_degree=None or tp_degree=1 during pipeline initialization"
        # )
        # assert not (self.config.use_fsdp and fused), (
        #     "load fused LoRA is not allowed when fully sharded data parallel is enabled; "
        #     "either load LoRA with fused=False or set use_fsdp=False during pipeline initialization"
        # )
        super().load_loras(lora_list, fused, save_original_weight)

    def unload_loras(self):
        self.dit.unload_loras()
        self.text_encoder.unload_loras()

    def encode_prompt(self, prompt):
        self.load_models_to_device(["text_encoder"])
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        prompt_emb = self.text_encoder(ids, mask)
        return prompt_emb, mask

    def encode_prompt_null(self, prompt):
        self.load_models_to_device(["text_encoder"])
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        prompt_emb = fwd_with_temperature(
            inputs=(ids, mask),
            model_fwd_func=self.text_encoder,
            get_hooked_layer_func=lambda i: self.text_encoder.blocks[i].attn.q,
            layer_start_idx=4,
            layer_end_idx=6,
        )
        return prompt_emb, mask

    def tokenize_lyric(self, lyrics: str):
        lyric_token_idx = [261]
        for line in lyrics.split("\n"):
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            try:
                self.lang_segment.getTexts(line)
                langCounts = self.lang_segment.getCounts()
                language = langCounts[0][0]
                if len(langCounts) > 1 and language == "en":
                    language = langCounts[1][0]
            except Exception:
                language = "en"

            if language not in SUPPORT_LANGUAGES:
                language = "en"
            if "zh" in language:
                language = "zh"
            if "spa" in language:
                language = "es"

            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, language)
                lyric_token_idx += token_idx + [2]
            except Exception as e:
                logger.warning("tokenize error", e, "for line", line, "major_language", language)
        lyric_mask = torch.tensor([1] * len(lyric_token_idx), device=self.device)[None]
        lyric_token_idx = torch.tensor(lyric_token_idx, device=self.device)[None]
        return lyric_token_idx, lyric_mask

    def predict_noise_with_cfg(
        self,
        model: ACEStepDiT,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_null: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_mask_ctx: torch.Tensor,
        attn_mask_ctx_null: torch.Tensor,
        cfg_scale: float,
        momentum_buffer: MomentumBuffer,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                model=model,
                latents=latents,
                timestep=timestep,
                context=context,
                attn_mask=attn_mask,
                attn_mask_ctx=attn_mask_ctx,
            )
        # cfg by predict noise one by one
        positive_noise_pred = self.predict_noise(
            model=model,
            latents=latents,
            timestep=timestep,
            context=context,
            attn_mask=attn_mask,
            attn_mask_ctx=attn_mask_ctx,
        )
        negative_noise_pred = fwd_with_temperature(
            inputs={
                "model": model,
                "latents": latents,
                "timestep": timestep,
                "context": context_null,
                "attn_mask": attn_mask,
                "attn_mask_ctx": attn_mask_ctx_null,
            },
            model_fwd_func=self.predict_noise,
            get_hooked_layer_func=lambda i: [
                model.transformer_blocks[i].attn.q,
                model.transformer_blocks[i].cross_attn.q,
            ],
            layer_start_idx=15,
            layer_end_idx=20,
        )
        noise_pred = apg_forward(
            pred_cond=positive_noise_pred,
            pred_uncond=negative_noise_pred,
            guidance_scale=cfg_scale,
            momentum_buffer=momentum_buffer,
        )
        return noise_pred

    def predict_noise(self, model, latents, timestep, context, attn_mask, attn_mask_ctx):
        latents = latents.to(dtype=self.config.model_dtype, device=self.device)

        noise_pred = model(
            x=latents,
            timestep=timestep,
            context=context,
            attn_mask=attn_mask,
            attn_mask_ctx=attn_mask_ctx,
        )
        return noise_pred

    def decode_audio(self, latents: torch.Tensor, sample_rate=48000) -> List[torch.Tensor]:
        self.load_models_to_device(["vae"])
        latents = latents.to(dtype=self.config.vae_dtype, device=self.device)
        audios = self.vae.decode(latents, sr=sample_rate)
        return audios

    @torch.no_grad()
    def text2audio(
        self,
        prompt: str,
        audio_duration: float,
        lyrics: str = "",
        cfg_scale: int = 15,
        omega_scale: int = 10.0,
        num_inference_steps: int = 60,
        seed=None,
        guidance_interval: float = 0.5,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        prompt_emb, prompt_attn_mask = self.encode_prompt(prompt)
        prompt_emb_null, prompt_attn_mask_null = self.encode_prompt_null(prompt)
        if len(lyrics.strip()) > 0:
            lyric_token_idx, lyric_mask = self.tokenize_lyric(lyrics)
        else:
            lyric_token_idx = torch.zeros((1, 1), device=self.device, dtype=torch.long)
            lyric_mask = torch.zeros((1, 1), device=self.device, dtype=torch.long)

        num_frames = int(audio_duration * 44100 / 512 / 8)
        noise = self.generate_noise((1, 8, 16, num_frames), seed=seed, device="cpu", dtype=torch.float32).to(
            self.device
        )
        attn_mask = torch.ones(1, num_frames, device=self.device, dtype=self.dtype)
        _, latents, sigmas, timesteps = self.prepare_latents(
            latents=noise,
            input_video=None,
            denoising_strength=None,
            num_inference_steps=num_inference_steps,
        )
        # Initialize sampler
        self.sampler.initialize(sigmas=sigmas)
        # guidance interval
        cfg_start_step = int(num_inference_steps * ((1 - guidance_interval) / 2))
        cfg_end_step = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        momentum_buffer = MomentumBuffer()

        context, context_mask = self.dit.encode(prompt_emb, lyric_token_idx, prompt_attn_mask, lyric_mask)
        context_null, context_mask_null = fwd_with_temperature(
            inputs={
                "context_prompt": prompt_emb_null,
                "context_lyric": lyric_token_idx,
                "attn_mask_prompt": prompt_attn_mask_null,
                "attn_mask_lyric": lyric_mask,
            },
            model_fwd_func=self.dit.encode,
            get_hooked_layer_func=lambda i: self.dit.lyric_encoder.encoders[i].self_attn.linear_q,
            layer_start_idx=4,
            layer_end_idx=6,
        )

        self.load_models_to_device(["dit"])
        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
            timestep = timestep.to(dtype=self.dtype, device=self.device)
            # Classifier-free guidance
            if cfg_start_step <= i < cfg_end_step:
                noise_pred = self.predict_noise_with_cfg(
                    model=self.dit,
                    latents=latents,
                    timestep=timestep,
                    context=context,
                    context_null=context_null,
                    attn_mask=attn_mask,
                    attn_mask_ctx=context_mask,
                    attn_mask_ctx_null=context_mask_null,
                    cfg_scale=cfg_scale,
                    momentum_buffer=momentum_buffer,
                )
            else:
                noise_pred = self.predict_noise(
                    model=self.dit.decode,
                    latents=latents,
                    timestep=timestep,
                    context=context,
                    attn_mask=attn_mask,
                    attn_mask_ctx=context_mask,
                )
            # Scheduler
            dx: torch.Tensor = noise_pred * (self.sampler.sigmas[i + 1] - self.sampler.sigmas[i])
            dx_mean = dx.mean(dim=(1, 2, 3), keepdim=True)
            latents = latents.to(dtype=torch.float32)
            latents += (dx - dx_mean) * omega_scale + dx
            latents = latents.to(dtype=noise_pred.dtype)
            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps), "DENOISING")

        # Decode
        return self.decode_audio(latents)

    def audio2audio(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_path_or_config: ACEStepPipelineConfig) -> "ACEStepMusicPipeline":
        if isinstance(model_path_or_config, str):
            config = ACEStepPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        dit_state_dict = None
        if dit_state_dict is None:
            logger.info(f"loading dit state dict from {config.model_path} ...")
            dit_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        if config.t5_path is None:
            config.t5_path = fetch_model("ACE-Step/ACE-Step-v1-3.5B", path="umt5-base/model.safetensors")
        logger.info(f"loading t5 state dict from {config.t5_path} ...")
        t5_state_dict = cls.load_model_checkpoint(config.t5_path, device="cpu", dtype=config.t5_dtype)

        if config.dcae_path is None:
            config.dcae_path = fetch_model(
                "ACE-Step/ACE-Step-v1-3.5B", path="music_dcae_f8c8/diffusion_pytorch_model.safetensors"
            )
        logger.info(f"loading vae/dcae state dict from {config.dcae_path} ...")
        dcae_state_dict = cls.load_model_checkpoint(config.dcae_path, device="cpu", dtype=config.vae_dtype)

        if config.vocoder_path is None:
            config.vocoder_path = fetch_model(
                "ACE-Step/ACE-Step-v1-3.5B", path="music_vocoder/diffusion_pytorch_model.safetensors"
            )
        logger.info(f"loading vae/vocoder state dict from {config.vocoder_path} ...")
        vocoder_state_dict = cls.load_model_checkpoint(config.vocoder_path, device="cpu", dtype=config.vae_dtype)

        state_dicts = ACEStateDicts(
            model=dit_state_dict,
            t5=t5_state_dict,
            dcae=dcae_state_dict,
            vocoder=vocoder_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: ACEStateDicts, config: ACEStepPipelineConfig) -> "ACEStepMusicPipeline":
        # if config.parallelism > 1:
        #     pipe = ParallelWrapper(
        #         cfg_degree=config.cfg_degree,
        #         sp_ulysses_degree=config.sp_ulysses_degree,
        #         sp_ring_degree=config.sp_ring_degree,
        #         tp_degree=config.tp_degree,
        #         use_fsdp=config.use_fsdp,
        #     )
        #     pipe.load_module(cls._from_state_dict, state_dicts=state_dicts, config=config)
        # else:
        pipe = cls._from_state_dict(state_dicts, config)
        return pipe

    @classmethod
    def _from_state_dict(cls, state_dicts: ACEStateDicts, config: ACEStepPipelineConfig) -> "ACEStepMusicPipeline":
        # default params from model config
        dcae_config: dict = DCAE.get_model_config()
        vocoder_config: dict = ADaMoSHiFiGANV1.get_model_config()
        dit_config: dict = ACEStepDiT.get_model_config()
        t5_config: dict = ACETextEncoder.get_model_config()
        config.shift = dit_config.pop("shift", 3.0)
        config.cfg_scale = dit_config.pop("cfg_scale", 15)
        config.num_inference_steps = dit_config.pop("num_inference_steps", 60)

        init_device = "cpu" if config.offload_mode is not None else config.device
        tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=256, clean="whitespace")
        text_encoder = ACETextEncoder.from_state_dict(state_dicts.t5, config=t5_config, device=init_device, dtype=config.t5_dtype)
        dcae = DCAE.from_state_dict(state_dicts.dcae, config=dcae_config, device=init_device, dtype=config.vae_dtype)
        hifi_gan = ADaMoSHiFiGANV1.from_state_dict(state_dicts.vocoder, config=vocoder_config, device=init_device, dtype=config.vae_dtype)
        vae = MusicDCAE(dcae=dcae, vocoder=hifi_gan)

        with LoRAContext():
            attn_kwargs = {
                "attn_impl": config.dit_attn_impl,
                "sparge_smooth_k": config.sparge_smooth_k,
                "sparge_cdfthreshd": config.sparge_cdfthreshd,
                "sparge_simthreshd1": config.sparge_simthreshd1,
                "sparge_pvthreshd": config.sparge_pvthreshd,
            }
            dit = ACEStepDiT.from_state_dict(
                state_dicts.model,
                config=dit_config,
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

    def compile(self):
        self.dit.compile_repeated_blocks(dynamic=True)
