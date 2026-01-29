import torch
import torch.nn as nn
import math
import numpy as np
from typing import Literal, Optional, Dict
from copy import deepcopy
from PIL import Image
from einops import rearrange, repeat
from contextlib import contextmanager

from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.pipelines.qwen_image import QwenImagePipeline
from diffsynth_engine.models.qwen_image import QwenImageVAE
from diffsynth_engine.models.basic.lora import LoRALinear
from diffsynth_engine.models.qwen_image.qwen_image_dit import QwenImageTransformerBlock, QwenEmbedRope
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.image import adain_color_fix, wavelet_color_fix

logger = logging.get_logger(__name__)


@contextmanager
def odtsr_forward():
    """
    Context manager for ODTSR forward pass optimization.

    Replaces two methods:
    1. LoRALinear.forward - to support batch CFG with dual outputs
    2. QwenImageTransformerBlock._modulate - optimized version without repeat_interleave
    """
    original_lora_forward = LoRALinear.forward
    original_modulate = QwenImageTransformerBlock._modulate
    original_rope_forward = QwenEmbedRope.forward

    def lora_batch_cfg_forward(self, x):
        y = nn.Linear.forward(self, x)
        if len(self._lora_dict) < 1:
            return y
        if x.ndim == 2:
            y2 = y.clone()
            for name, lora in self._lora_dict.items():
                y2 += lora(x)
            return torch.stack([y, y2], dim=1)
        else:
            L2 = x.shape[1]
            L = L2 // 2
            x2 = x[:, L:, :]
            for name, lora in self._lora_dict.items():
                y[:, L:] += lora(x2)
            return y

    def optimized_rope_forward(self, video_fhw, txt_length, device):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        idx = 0
        for fhw in video_fhw:
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + txt_length, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


    def optimized_modulate(self, x, mod_params, index=None):
        if mod_params.ndim == 2:
            shift, scale, gate = mod_params.chunk(3, dim=-1)
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
        else:
            B, L2, C = x.shape
            L = L2 // 2
            shift, scale, gate = mod_params.chunk(3, dim=-1)  # Each: [B, 2, dim]

            result = torch.empty_like(x)
            gate_result = torch.empty(B, L2, gate.shape[-1], dtype=x.dtype, device=x.device)

            result[:, :L] = x[:, :L] * (1 + scale[:, 0:1]) + shift[:, 0:1]
            gate_result[:, :L] = gate[:, 0:1].expand(-1, L, -1)

            result[:, L:] = x[:, L:] * (1 + scale[:, 1:2]) + shift[:, 1:2]
            gate_result[:, L:] = gate[:, 1:2].expand(-1, L, -1)

            return result, gate_result

    LoRALinear.forward = lora_batch_cfg_forward
    QwenImageTransformerBlock._modulate = optimized_modulate
    QwenEmbedRope.forward = optimized_rope_forward

    try:
        yield
    finally:
        LoRALinear.forward = original_lora_forward
        QwenImageTransformerBlock._modulate = original_modulate
        QwenEmbedRope.forward = original_rope_forward


class QwenImageUpscalerTool:
    """
    Tool for ODTSR (One-step Diffusion Transformer Super Resolution) image upscaling.
    https://huggingface.co/double8fun/ODTSR
    """

    def __init__(
        self,
        pipeline: QwenImagePipeline,
        odtsr_weight_path: Optional[str] = None,
    ):
        self.pipe = pipeline
        self.device = self.pipe.device
        self.dtype = self.pipe.dtype

        # to avoid "small grid" artifacts in generated images
        self._convert_dit_part_linear_weight()

        if not odtsr_weight_path:
            odtsr_weight_path = fetch_model("muse/ODTSR", revision="master", path="weight.safetensors")
        odtsr_state_dict = load_file(odtsr_weight_path)
        lora_state_dict = self._convert_odtsr_lora(odtsr_state_dict)
        lora_state_dict_list = [(lora_state_dict, 1.0, odtsr_weight_path)]
        self.pipe._load_lora_state_dicts(lora_state_dict_list, fused=False, save_original_weight=False)

        self.new_vae = deepcopy(self.pipe.vae)
        self._load_vae_encoder_weights(odtsr_state_dict)

        sigmas = torch.linspace(1.0, 0.0, 1000 + 1)[:-1]
        mu = 0.8
        shift_terminal = 0.02
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        one_minus_sigmas = 1 - sigmas
        scale_factor = one_minus_sigmas[-1] / (1 - shift_terminal)
        self.sigmas = 1 - (one_minus_sigmas / scale_factor)
        self.sigmas = self.sigmas.to(device=self.device)
        self.timesteps = self.sigmas * self.pipe.noise_scheduler.num_train_timesteps
        self.timesteps = self.timesteps.to(device=self.device)
        self.start_timestep = 750
        self.fixed_timestep = self.timesteps[self.start_timestep].to(device=self.device)
        self.one_step_sigma = self.sigmas[self.start_timestep].to(device=self.device)

        self.prompt = "High Contrast, hyper detailed photo, 2k UHD"
        self.prompt_emb, self.prompt_emb_mask = self.pipe.encode_prompt(self.prompt, 1, 4096)

    @classmethod
    def from_pretrained(
        cls,
        qwen_model_path: str,
        odtsr_weight_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        config = QwenImagePipelineConfig(
            model_path=qwen_model_path,
            model_dtype=dtype,
            device=device,
            load_encoder=True,
        )
        pipe = QwenImagePipeline.from_pretrained(config)
        return cls(pipe, odtsr_weight_path)

    def _convert_dit_part_linear_weight(self):
        """
        Perform dtype conversion on weights of specific Linear layers in the DIT model.

        This is an important trick: for Linear layers NOT in the patterns list, convert their weights
        to float8_e4m3fn first, then convert back to the original dtype (typically bfloat16). This operation
        matches the weight processing method used during training to avoid "small grid" artifacts in generated images.

        Layers in the patterns list (such as LoRA-related layers) are skipped and their original weights remain unchanged.
        """
        patterns = [
            "img_in",
            "img_mod.1",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "to_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ]

        def _convert_weight(parent: nn.Module, name_prefix: str = ""):
            for name, module in list(parent.named_children()):
                full_name = f"{name_prefix}{name}"
                if isinstance(module, torch.nn.Linear):
                    if not any(p in full_name for p in patterns):
                        origin_dtype = module.weight.data.dtype
                        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                        module.weight.data = module.weight.data.to(origin_dtype)
                        if module.bias is not None:
                            module.bias.data = module.bias.data.to(torch.float8_e4m3fn)
                            module.bias.data = module.bias.data.to(origin_dtype)
                else:
                    _convert_weight(module, name_prefix=full_name + ".")

        _convert_weight(self.pipe.dit)

    def _convert_odtsr_lora(self, odtsr_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for key, param in odtsr_state_dict.items():
            if "lora_A2" in key:
                lora_b_key = key.replace("lora_A2", "lora_B2")
                lora_b_param = odtsr_state_dict[lora_b_key]

                lora_a_key = key.replace("lora_A2", "lora_A").replace("pipe.dit.", "")
                lora_b_key = lora_b_key.replace("lora_B2", "lora_B").replace("pipe.dit.", "")
                state_dict[lora_a_key] = param
                state_dict[lora_b_key] = lora_b_param

        return state_dict

    def _load_vae_encoder_weights(self, state_dict: Dict[str, torch.Tensor]):
        try:
            vae_state_dict = {}
            for k, v in state_dict.items():
                if 'pipe.new_vae.' in k:
                    new_key = k.replace('pipe.new_vae.', '')
                    vae_state_dict[new_key] = v
            if vae_state_dict:
                self.new_vae.load_state_dict(vae_state_dict, strict=False)
                logger.info(f"Loaded {len(vae_state_dict)} trained VAE encoder parameters")
            else:
                logger.warning(f"No 'pipe.new_vae.' weights found, using original VAE")
        except Exception as e:
            logger.error(f"Failed to load VAE encoder weights: {e}")
            raise e

    
    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * sample + sigma * noise
        return sample

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=self.dtype, device=self.device)
        image = image * (2 / 255) - 1
        image = repeat(image, f"H W C -> B C H W", **({"B": 1}))
        return image

    def _prepare_condition_latents(self, image: Image.Image, vae: QwenImageVAE, vae_tiled: bool) -> torch.Tensor:
        image_tensor = self.preprocess_image(image).to(dtype=self.pipe.config.vae_dtype)
        image_tensor = image_tensor.unsqueeze(2)

        latents = vae.encode(
            image_tensor,
            device=self.device,
            tiled=vae_tiled,
            tile_size=self.pipe.vae_tile_size,
            tile_stride=self.pipe.vae_tile_stride,
        )
        latents = latents.squeeze(2).to(device=self.device, dtype=self.dtype)
        return latents

    def _single_step_denoise(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        noise: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        fidelity: float,
    ) -> torch.Tensor:
        fidelity_timestep_id = int(self.start_timestep + fidelity * (1000 - self.start_timestep) + 0.5)
        if fidelity_timestep_id != 1000:
            fidelity_timestep = self.timesteps[fidelity_timestep_id].to(device=self.device)
            image_latents = self.add_noise(image_latents, noise, fidelity_timestep)

        latents = self.add_noise(latents, noise, self.fixed_timestep)

        with odtsr_forward():
            noise_pred = self.pipe.predict_noise_with_cfg(
                latents=latents,
                image_latents=[image_latents],
                timestep=self.fixed_timestep.unsqueeze(0),
                prompt_emb=prompt_emb,
                prompt_emb_mask=prompt_emb_mask,
                negative_prompt_emb=None,
                negative_prompt_emb_mask=None,
                context_latents=None,
                entity_prompt_embs=None,
                entity_prompt_emb_masks=None,
                negative_entity_prompt_embs=None,
                negative_entity_prompt_emb_masks=None,
                entity_masks=None,
                cfg_scale=1.0,
                batch_cfg=self.pipe.config.batch_cfg,
            )

        denoised = latents + (0 - self.one_step_sigma) * noise_pred
        return denoised

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        scale: int = 2,
        prompt: str = "High Contrast, hyper detailed photo, 2k UHD",
        fidelity: float = 1.0,
        align_method: Literal["none", "adain", "wavelet"] = "none",
    ) -> Image.Image:
        width, height = image.size
        target_width, target_height = width * scale, height * scale
        target_width_round = target_width // 16 * 16
        target_height_round = target_height // 16 * 16
        logger.info(f"Upscaling image from {width}x{height} to {target_width}x{target_height}")
        vae_tiled = (target_width_round * target_height_round > 2048 * 2048)

        resized_image = image.resize((target_width_round, target_height_round), Image.BICUBIC)

        condition_latents = self._prepare_condition_latents(resized_image, self.pipe.vae, vae_tiled)
        latents = self._prepare_condition_latents(resized_image, self.new_vae, vae_tiled)

        noise = self.pipe.generate_noise(
            (1, 16, target_height_round // 8, target_width_round // 8),
            seed=42,
            device=self.device,
            dtype=self.dtype
        )

        prompt_emb, prompt_emb_mask = self.prompt_emb, self.prompt_emb_mask
        if prompt != self.prompt:
            prompt_emb, prompt_emb_mask = self.pipe.encode_prompt(prompt, 1, 4096)
            
        denoised_latents = self._single_step_denoise(
            latents=latents,
            noise=noise,
            image_latents=condition_latents,
            prompt_emb=prompt_emb,
            prompt_emb_mask=prompt_emb_mask,
            fidelity=fidelity,
        )

        # Decode
        denoised_latents = rearrange(denoised_latents, "B C H W -> B C 1 H W")
        vae_output = rearrange(
            self.pipe.vae.decode(
                denoised_latents.to(self.pipe.vae.model.encoder.conv1.weight.dtype),
                device=self.pipe.vae.model.encoder.conv1.weight.device,
                tiled=vae_tiled,
                tile_size=self.pipe.vae_tile_size,
                tile_stride=self.pipe.vae_tile_stride,
            )[0],
            "C B H W -> B C H W",
        )
        result_image = self.pipe.vae_output_to_image(vae_output)
        self.pipe.model_lifecycle_finish(["vae"])

        if align_method == "adain":
            result_image = adain_color_fix(target=result_image, source=resized_image)
        elif align_method == "wavelet":
            result_image = wavelet_color_fix(target=result_image, source=resized_image)

        result_image = result_image.resize((target_width, target_height), Image.BICUBIC)
        return result_image
