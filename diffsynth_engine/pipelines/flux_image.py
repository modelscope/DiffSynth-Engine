import re
import os
import torch
import torch.nn as nn
import math
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
from diffsynth_engine.models.flux import (
    FluxTextEncoder1,
    FluxTextEncoder2,
    FluxVAEDecoder,
    FluxVAEEncoder,
    FluxDiT,
    flux_dit_config,
    flux_text_encoder_config,
)
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH, FLUX_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils.parallel import ParallelModel, shard_model
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.platform import empty_cache

from einops import rearrange

logger = logging.get_logger(__name__)


class FluxLoRAConverter(LoRAStateDictConverter):
    def _from_kohya(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        flux_dim = 3072
        dit_rename_dict = flux_dit_config["civitai"]["rename_dict"]
        dit_suffix_rename_dict = flux_dit_config["civitai"]["suffix_rename_dict"]
        clip_rename_dict = flux_text_encoder_config["diffusers"]["rename_dict"]
        clip_attn_rename_dict = flux_text_encoder_config["diffusers"]["attn_rename_dict"]

        dit_dict = {}
        te_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            if ".alpha" not in key:
                continue
            if "lora_unet" in key:
                key = key.replace("lora_unet_", "")
                key = re.sub(r"(\d+)_", r"\1.", key)
                key = re.sub(r"_(\d+)", r".\1", key)
                key = key.replace("modulation_lin", "modulation.lin")
                key = key.replace("mod_lin", "mod.lin")
                key = key.replace("attn_qkv", "attn.qkv")
                key = key.replace("attn_proj", "attn.proj")
                key = key.replace(".alpha", ".weight")
                names = key.split(".")
                if key in dit_rename_dict:
                    rename = dit_rename_dict[key]
                elif names[0] == "double_blocks":
                    rename = f"blocks.{names[1]}." + dit_suffix_rename_dict[".".join(names[2:])]
                elif names[0] == "single_blocks":
                    if ".".join(names[2:]) in dit_suffix_rename_dict:
                        rename = f"single_blocks.{names[1]}." + dit_suffix_rename_dict[".".join(names[2:])]
                    else:
                        raise ValueError(f"Unsupported key: {key}")
                if "to_qkv_mlp" in rename:
                    rename = rename.replace(".weight", "")
                    qkv_lora_args = {}
                    qkv_lora_args["alpha"] = param
                    qkv_lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")][
                        : 3 * flux_dim
                    ]
                    qkv_lora_args["rank"] = qkv_lora_args["up"].shape[1]
                    qkv_lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    dit_dict[rename.replace("to_qkv_mlp", "attn.to_qkv")] = qkv_lora_args

                    mlp_lora_args = {}
                    mlp_lora_args["alpha"] = param
                    mlp_lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")][
                        3 * flux_dim :
                    ]
                    mlp_lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    mlp_lora_args["rank"] = mlp_lora_args["up"].shape[1]
                    dit_dict[rename.replace("to_qkv_mlp", "mlp.0")] = mlp_lora_args
                elif "linear_a" in rename:
                    rename = rename.replace(".weight", "")
                    attn_lora_args: dict = {}
                    attn_lora_args["alpha"] = param
                    attn_lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")][
                        : 3 * flux_dim
                    ]
                    attn_lora_args["rank"] = attn_lora_args["up"].shape[1]
                    attn_lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    dit_dict[rename.replace("linear_a", "norm_msa_a.linear")] = attn_lora_args

                    mlp_lora_args = {}
                    mlp_lora_args["alpha"] = param
                    mlp_lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")][
                        3 * flux_dim :
                    ]
                    mlp_lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    mlp_lora_args["rank"] = mlp_lora_args["up"].shape[1]
                    dit_dict[rename.replace("linear_a", "norm_mlp_a.linear")] = mlp_lora_args
                elif "linear_b" in rename:
                    rename = rename.replace(".weight", "")
                    attn_lora_args: dict = {}
                    attn_lora_args["alpha"] = param
                    attn_lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")][
                        : 3 * flux_dim
                    ]
                    attn_lora_args["rank"] = attn_lora_args["up"].shape[1]
                    attn_lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    dit_dict[rename.replace("linear_b", "norm_msa_b.linear")] = attn_lora_args

                    mlp_lora_args = {}
                    mlp_lora_args["alpha"] = param
                    mlp_lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")][
                        3 * flux_dim :
                    ]
                    mlp_lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    mlp_lora_args["rank"] = mlp_lora_args["up"].shape[1]
                    dit_dict[rename.replace("linear_b", "norm_mlp_b.linear")] = mlp_lora_args
                else:
                    lora_args = {}
                    lora_args["alpha"] = param
                    lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                    lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                    lora_args["rank"] = lora_args["up"].shape[1]
                    rename = rename.replace(".weight", "")
                    dit_dict[rename] = lora_args
            elif "lora_te" in key:
                name = key.replace("lora_te1", "text_encoder")
                name = name.replace("text_model_encoder_layers", "text_model.encoder.layers")
                name = name.replace(".alpha", ".weight")
                rename = ""
                if name in clip_rename_dict:
                    if name == "text_model.embeddings.position_embedding.weight":
                        param = param.reshape((1, param.shape[0], param.shape[1]))
                    rename = clip_rename_dict[name]
                elif name.startswith("text_model.encoder.layers."):
                    names = name.split(".")
                    layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                    rename = ".".join(["encoders", layer_id, clip_attn_rename_dict[layer_type], tail])
                else:
                    raise ValueError(f"Unsupported key: {key}")
                lora_args = {}
                lora_args["alpha"] = param
                lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                lora_args["rank"] = lora_args["up"].shape[1]
                rename = rename.replace(".weight", "")
                te_dict[rename] = lora_args
            else:
                raise ValueError(f"Unsupported key: {key}")
        return {"dit": dit_dict, "text_encoder_1": te_dict}

    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            if ".alpha" not in key:
                continue
            key = key.replace(".alpha", ".weight")
            key = key.replace("transformer.", "")
            if "single_transformer_blocks" in key:  # transformer.single_transformer_blocks.0.attn.to_k.weight
                key = key.replace(
                    "single_transformer_blocks", "single_blocks"
                )  # single_transformer_blocks.0.attn.to_k.weight
                key = key.replace("norm.linear", "modulation.lin")
                key = key.replace("proj_out", "linear2")
                key = key.replace("attn", "linear1")  # linear1 = [to_q, to_k, to_v, mlp]
                key = key.replace("proj_mlp", "linear1.mlp")
            elif "transformer_blocks" in key:
                key = key.replace("transformer_blocks", "double_blocks")
                key = key.replace("attn.add_k_proj", "txt_attn.qkv.to_k")
                key = key.replace("attn.add_q_proj", "txt_attn.qkv.to_q")
                key = key.replace("attn.add_v_proj", "txt_attn.qkv.to_v")
                key = key.replace("attn.to_add_out", "txt_attn.qkv.to_out")
                key = key.replace("attn.to_k", "img_attn.qkv.to_k")
                key = key.replace("attn.to_q", "img_attn.qkv.to_q")
                key = key.replace("attn.to_v", "img_attn.qkv.to_v")
                key = key.replace("attn.to_out", "img_attn.qkv.to_out")
                key = key.replace("ff.net", "img_mlp")
                key = key.replace("ff_context.net", "txt_mlp")
                key = key.replace("norm1.linear", "img_mod.lin")
                key = key.replace("norm1_context.linear", "txt_mod.lin")
                key = key.replace(".0.proj", ".0")
            elif "context_embedder" in key:
                key = key.replace("context_embedder", "txt_in")
            elif "x_embedder" in key and "_x_embedder" not in key:
                key = key.replace("x_embedder", "img_in")
            elif "time_text_embed" in key:
                key = key.replace("time_text_embed.", "")
                key = key.replace("timestep_embedder", "time_in")
                key = key.replace("guidance_embedder", "guidance_in")
                key = key.replace("text_embedder", "vector_in")
                key = key.replace("linear_1", "in_layer")
                key = key.replace("linear_2", "out_layer")
            elif "norm_out.linear" in key:
                key = key.replace("norm_out.linear", "final_layer.adaLN_modulation.1")
            elif "proj_out" in key:
                key = key.replace("proj_out", "final_layer.linear")
            else:
                raise ValueError(f"Unsupported key: {key}")
            lora_args = {}
            lora_args["alpha"] = param
            lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
            lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
            lora_args["rank"] = lora_args["up"].shape[1]
            key = key.replace(".weight", "")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if "lora_te" in key or "lora_unet" in key:
            return self._from_kohya(lora_state_dict)
        elif key.startswith("transformer"):
            return self._from_diffusers(lora_state_dict)
        raise ValueError(f"Unsupported key: {key}")


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def accumulate(result, new_item):
    if result is None:
        return new_item
    for i, item in enumerate(new_item):
        result[i] += item
    return result


ImageType = Union[Image.Image, torch.Tensor, List[Image.Image], List[torch.Tensor]]


class ControlType(Enum):
    normal = "normal"
    bfl_control = "bfl_control"
    bfl_fill = "bfl_fill"

    def get_in_channel(self):
        if self == ControlType.normal:
            return 64
        elif self == ControlType.bfl_control:
            return 128
        elif self == ControlType.bfl_fill:
            return 384


@dataclass
class ControlNetParams:
    scale: float
    image: ImageType
    model: Optional[nn.Module] = None
    mask: Optional[ImageType] = None
    control_start: float = 0
    control_end: float = 1


@dataclass
class FluxModelConfig:
    dit_path: str | os.PathLike
    clip_path: Optional[str | os.PathLike] = None
    t5_path: Optional[str | os.PathLike] = None
    vae_path: Optional[str | os.PathLike] = None

    dit_dtype: torch.dtype = torch.bfloat16
    clip_dtype: torch.dtype = torch.bfloat16
    t5_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16

    dit_attn_impl: Optional[str] = "auto"
    use_fp8_linear: bool = False

    sp_ulysses_degree: Optional[int] = None
    sp_ring_degree: Optional[int] = None
    tp_degree: Optional[int] = None
    use_fsdp: bool = False


class FluxImagePipeline(BasePipeline):
    lora_converter = FluxLoRAConverter()

    def __init__(
        self,
        config: FluxModelConfig,
        tokenizer: CLIPTokenizer,
        tokenizer_2: T5TokenizerFast,
        text_encoder_1: FluxTextEncoder1,
        text_encoder_2: FluxTextEncoder2,
        dit: FluxDiT,
        vae_decoder: FluxVAEDecoder,
        vae_encoder: FluxVAEEncoder,
        load_text_encoder: bool = True,
        use_cfg: bool = False,
        batch_cfg: bool = False,
        vae_tiled: bool = False,
        vae_tile_size: int = 256,
        vae_tile_stride: int = 256,
        control_type: ControlType = ControlType.normal,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            vae_tiled=vae_tiled,
            vae_tile_size=vae_tile_size,
            vae_tile_stride=vae_tile_stride,
            device=device,
            dtype=dtype,
        )
        self.config = config
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0, use_dynamic_shifting=True)
        self.sampler = FlowMatchEulerSampler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.dit = dit
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.load_text_encoder = load_text_encoder
        self.use_cfg = use_cfg
        self.batch_cfg = batch_cfg
        self.ip_adapter = None
        self.redux = None
        self.control_type = control_type
        self.model_names = [
            "text_encoder_1",
            "text_encoder_2",
            "dit",
            "vae_decoder",
            "vae_encoder",
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_config: str | os.PathLike | FluxModelConfig,
        control_type: ControlType = ControlType.normal,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: str | None = None,
        load_text_encoder: bool = True,
        parallelism: int = 1,
        use_cfg_parallel: bool = False,
    ) -> "FluxImagePipeline":
        cls.validate_offload_mode(offload_mode)

        model_config = (
            model_path_or_config
            if isinstance(model_path_or_config, FluxModelConfig)
            else FluxModelConfig(dit_path=model_path_or_config, dit_dtype=dtype, t5_dtype=dtype, clip_dtype=dtype)
        )
        if model_config.vae_path is None:
            model_config.vae_path = fetch_model("muse/flux_vae", revision="20241015120836", path="ae.safetensors")

        if model_config.clip_path is None and load_text_encoder:
            model_config.clip_path = fetch_model(
                "muse/flux_clip_l", revision="20241209", path="clip_l_bf16.safetensors"
            )
        if model_config.t5_path is None and load_text_encoder:
            model_config.t5_path = fetch_model(
                "muse/google_t5_v1_1_xxl", revision="20241024105236", path="t5xxl_v1_1_bf16.safetensors"
            )

        logger.info(f"loading state dict from {model_config.dit_path} ...")
        dit_state_dict = cls.load_model_checkpoint(model_config.dit_path, device="cpu", dtype=dtype)
        logger.info(f"loading state dict from {model_config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(model_config.vae_path, device="cpu", dtype=dtype)
        if load_text_encoder:
            logger.info(f"loading state dict from {model_config.clip_path} ...")
            clip_state_dict = cls.load_model_checkpoint(model_config.clip_path, device="cpu", dtype=dtype)
            logger.info(f"loading state dict from {model_config.t5_path} ...")
            t5_state_dict = cls.load_model_checkpoint(model_config.t5_path, device="cpu", dtype=dtype)

        init_device = "cpu" if offload_mode else device
        if load_text_encoder:
            tokenizer = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)
            tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)

        if parallelism > 1:
            parallel_config = cls.init_parallel_config(parallelism, use_cfg_parallel, model_config)
            cfg_degree = parallel_config["cfg_degree"]
            sp_ulysses_degree = parallel_config["sp_ulysses_degree"]
            sp_ring_degree = parallel_config["sp_ring_degree"]
            tp_degree = parallel_config["tp_degree"]
            use_fsdp = parallel_config["use_fsdp"]
            with LoRAContext():
                dit = FluxDiT.from_state_dict(
                    dit_state_dict,
                    device=init_device,
                    dtype=model_config.dit_dtype,
                    in_channel=control_type.get_in_channel(),
                    attn_impl=model_config.dit_attn_impl,
                    use_usp=(sp_ulysses_degree * sp_ring_degree > 1),
                )
                if model_config.use_fp8_linear:
                    enable_fp8_linear(dit)
                dit = ParallelModel(
                    dit,
                    cfg_degree=cfg_degree,
                    sp_ulysses_degree=sp_ulysses_degree,
                    sp_ring_degree=sp_ring_degree,
                    tp_degree=tp_degree,
                    shard_fn=partial(shard_model, wrap_module_names=["blocks", "single_blocks"]) if use_fsdp else None,
                    device="cuda",
                )
        else:
            with LoRAContext():
                dit = FluxDiT.from_state_dict(
                    dit_state_dict,
                    device=init_device,
                    dtype=model_config.dit_dtype,
                    in_channel=control_type.get_in_channel(),
                    attn_impl=model_config.dit_attn_impl,
                )
                if model_config.use_fp8_linear:
                    enable_fp8_linear(dit)

        if load_text_encoder:
            with LoRAContext():
                text_encoder_1 = FluxTextEncoder1.from_state_dict(
                    clip_state_dict, device=init_device, dtype=model_config.clip_dtype
                )
            text_encoder_2 = FluxTextEncoder2.from_state_dict(
                t5_state_dict, device=init_device, dtype=model_config.t5_dtype
            )

        vae_decoder = FluxVAEDecoder.from_state_dict(vae_state_dict, device=init_device, dtype=model_config.vae_dtype)
        vae_encoder = FluxVAEEncoder.from_state_dict(vae_state_dict, device=init_device, dtype=model_config.vae_dtype)

        pipe = cls(
            config=model_config,
            tokenizer=tokenizer if load_text_encoder else None,
            tokenizer_2=tokenizer_2 if load_text_encoder else None,
            text_encoder_1=text_encoder_1 if load_text_encoder else None,
            text_encoder_2=text_encoder_2 if load_text_encoder else None,
            dit=dit,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            load_text_encoder=load_text_encoder,
            control_type=control_type,
            device=device,
            dtype=dtype,
        )
        if offload_mode == "cpu_offload":
            pipe.enable_cpu_offload()
        elif offload_mode == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload()
        return pipe

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        assert self.config.tp_degree is None, (
            "load LoRA is not allowed when tensor parallel is enabled; "
            "set tp_degree=None during pipeline initialization"
        )
        assert not (self.config.use_fsdp and fused), (
            "load fused LoRA is not allowed when fully sharded data parallel is enabled; "
            "either load LoRA with fused=False or set use_fsdp=False during pipeline initialization"
        )
        super().load_loras(lora_list, fused, save_original_weight)

    def unload_loras(self):
        self.dit.unload_loras()
        self.text_encoder_1.unload_loras()
        self.text_encoder_2.unload_loras()

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16
    ) -> "FluxImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, clip_skip: int = 2):
        if (
            self.tokenizer is None
            or self.tokenizer_2 is None
            or self.text_encoder_1 is None
            or self.text_encoder_2 is None
        ):
            return torch.zeros((1, 512, 4096), device=self.device, dtype=self.dtype), torch.zeros(
                (1, 768), device=self.device, dtype=self.dtype
            )

        input_ids = self.tokenizer(prompt, max_length=77)["input_ids"].to(device=self.device)
        _, add_text_embeds = self.text_encoder_1(input_ids, clip_skip=clip_skip)

        input_ids = self.tokenizer_2(prompt, max_length=512)["input_ids"].to(device=self.device)
        prompt_emb = self.text_encoder_2(input_ids)

        return prompt_emb, add_text_embeds

    def prepare_extra_input(self, latents, positive_prompt_emb, guidance=1.0):
        image_ids = FluxDiT.prepare_image_ids(latents)
        guidance = torch.tensor([guidance] * latents.shape[0], device=latents.device, dtype=latents.dtype)
        text_ids = torch.zeros(positive_prompt_emb.shape[0], positive_prompt_emb.shape[1], 3).to(
            device=self.device, dtype=positive_prompt_emb.dtype
        )
        return image_ids, text_ids, guidance

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        positive_add_text_embeds: torch.Tensor,
        negative_add_text_embeds: torch.Tensor,
        image_emb: torch.Tensor | None,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        cfg_scale: float,
        guidance: torch.Tensor,
        controlnet_params: List[ControlNetParams],
        current_step: int,
        total_step: int,
        use_cfg: bool = False,
        batch_cfg: bool = False,
    ):
        if cfg_scale <= 1.0 or not use_cfg:
            return self.predict_noise(
                latents,
                timestep,
                positive_prompt_emb,
                positive_add_text_embeds,
                image_emb,
                image_ids,
                text_ids,
                guidance,
                controlnet_params,
                current_step,
                total_step,
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents,
                timestep,
                positive_prompt_emb,
                positive_add_text_embeds,
                image_emb,
                image_ids,
                text_ids,
                guidance,
                controlnet_params,
                current_step,
                total_step,
            )
            negative_noise_pred = self.predict_noise(
                latents,
                timestep,
                negative_prompt_emb,
                negative_add_text_embeds,
                image_emb,
                image_ids,
                text_ids,
                guidance,
                controlnet_params,
                current_step,
                total_step,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            add_text_embeds = torch.cat([positive_add_text_embeds, negative_add_text_embeds], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents,
                timestep,
                prompt_emb,
                add_text_embeds,
                image_emb,
                image_ids,
                text_ids,
                guidance,
                controlnet_params,
                current_step,
                total_step,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        add_text_embeds: torch.Tensor,
        image_emb: torch.Tensor | None,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: float,
        controlnet_params: List[ControlNetParams],
        current_step: int,
        total_step: int,
    ):
        if self.control_type != ControlType.normal:
            controlnet_param = controlnet_params[0]
            latents = torch.cat((latents, controlnet_param.image * controlnet_param.scale), dim=1)
            latents = latents.to(self.dtype)
            controlnet_params = []

        double_block_output, single_block_output = self.predict_multicontrolnet(
            latents=latents,
            timestep=timestep,
            prompt_emb=prompt_emb,
            add_text_embeds=add_text_embeds,
            guidance=guidance,
            text_ids=text_ids,
            image_ids=image_ids,
            controlnet_params=controlnet_params,
            current_step=current_step,
            total_step=total_step,
        )

        self.load_models_to_device(["dit"])

        noise_pred = self.dit(
            hidden_states=latents,
            timestep=timestep,
            prompt_emb=prompt_emb,
            pooled_prompt_emb=add_text_embeds,
            image_emb=image_emb,
            guidance=guidance,
            text_ids=text_ids,
            image_ids=image_ids,
            controlnet_double_block_output=double_block_output,
            controlnet_single_block_output=single_block_output,
        )
        return noise_pred

    def prepare_latents(
        self,
        latents: torch.Tensor,
        input_image: Optional[Image.Image],
        denoising_strength: float,
        num_inference_steps: int,
        mu: float,
    ):
        # Prepare scheduler
        if input_image is not None:
            self.load_models_to_device(["vae_encoder"])
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(
                total_steps, mu=mu, sigma_min=1 / total_steps, sigma_max=1.0
            )
            t_start = max(total_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]
            noise = latents
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image)
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(
                num_inference_steps, mu=mu, sigma_min=1 / num_inference_steps, sigma_max=1.0
            )
            init_latents = latents.clone()
        sigmas, timesteps = sigmas.to(device=self.device), timesteps.to(self.device)
        return init_latents, latents, sigmas, timesteps

    def prepare_masked_latent(self, image: Image.Image, mask: Image.Image | None, height: int, width: int):
        self.load_models_to_device(["vae_encoder"])
        if mask is None:
            image = image.resize((width, height))
            image = self.preprocess_image(image).to(device=self.device, dtype=self.dtype)
            latent = self.encode_image(image)
        else:
            if self.control_type == ControlType.normal:
                image = image.resize((width, height))
                mask = mask.resize((width, height))
                image = self.preprocess_image(image).to(device=self.device, dtype=self.dtype)
                mask = self.preprocess_mask(mask).to(device=self.device, dtype=self.dtype)
                masked_image = image.clone()
                masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1
                latent = self.encode_image(masked_image)
                mask = torch.nn.functional.interpolate(mask, size=(latent.shape[2], latent.shape[3]))
                mask = 1 - mask
                latent = torch.cat([latent, mask], dim=1)
            elif self.control_type == ControlType.bfl_fill:
                image = image.resize((width, height))
                mask = mask.resize((width, height))
                image = self.preprocess_image(image).to(device=self.device, dtype=self.dtype)
                mask = self.preprocess_mask(mask).to(device=self.device, dtype=self.dtype)
                image = image * (1 - mask)
                image = self.encode_image(image)
                mask = rearrange(mask, "b 1 (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
                latent = torch.cat((image, mask), dim=1)
            else:
                raise ValueError(f"Unsupported mask latent prepare for controlnet type: {self.control_type}")
        return latent

    def prepare_controlnet_params(self, controlnet_params: List[ControlNetParams], h, w):
        results = []
        for param in controlnet_params:
            condition = self.prepare_masked_latent(param.image, param.mask, h, w)
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
        add_text_embeds: torch.Tensor,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: float,
        controlnet_params: List[ControlNetParams],
        current_step: int,
        total_step: int,
    ):
        double_block_output_results, single_block_output_results = None, None
        if len(controlnet_params) > 0:
            self.load_models_to_device([])
        for param in controlnet_params:
            current_scale = param.scale
            if not (
                current_step >= param.control_start * total_step and current_step <= param.control_end * total_step
            ):
                # if current_step is not in the control range
                # skip thie controlnet
                continue
            if self.offload_mode == "sequential_cpu_offload" or self.offload_mode == "cpu_offload":
                empty_cache()
                param.model.to(self.device)
            double_block_output, single_block_output = param.model(
                latents,
                param.image,
                current_scale,
                timestep,
                prompt_emb,
                add_text_embeds,
                guidance,
                image_ids,
                text_ids,
            )
            if self.offload_mode == "sequential_cpu_offload" or self.offload_mode == "cpu_offload":
                empty_cache()
                param.model.to("cpu")
            double_block_output_results = accumulate(double_block_output_results, double_block_output)
            single_block_output_results = accumulate(single_block_output_results, single_block_output)
        return double_block_output_results, single_block_output_results

    def load_ip_adapter(self, ip_adapter):
        self.ip_adapter = ip_adapter
        self.ip_adapter.inject(self.dit)

    def unload_ip_adapter(self):
        if self.ip_adapter is not None:
            self.ip_adapter.remove(self.dit)
            self.ip_adapter = None

    def load_redux(self, redux):
        self.redux = redux

    def unload_redux(self):
        self.redux = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        ref_image: Image.Image | None = None,  # use for redux, ip-adapter, instance-id
        cfg_scale: float = 1.0,  # true cfg
        clip_skip: int = 2,
        input_image: Image.Image | None = None,  # use for img2img
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        seed: int | None = None,
        flux_guidance_scale=3.5,  # use for flux guidance, not true cfg
        controlnet_params: List[ControlNetParams] | ControlNetParams = [],
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        if self.control_type != ControlType.normal:
            assert controlnet_params and len(controlnet_params) == 1, "bfl_controlnet must have one controlnet"

        if input_image is not None:
            width, height = input_image.size
        if not isinstance(controlnet_params, list):
            controlnet_params = [controlnet_params]
        self.validate_image_size(height, width, minimum=64, multiple_of=16)

        noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device="cpu", dtype=self.dtype).to(
            device=self.device
        )
        # dynamic shift
        image_seq_len = math.ceil(height // 16) * math.ceil(width // 16)
        mu = calculate_shift(image_seq_len)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(
            noise, input_image, denoising_strength, num_inference_steps, mu
        )
        # Initialize sampler
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas)

        # Encode prompts
        self.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        positive_prompt_emb, positive_add_text_embeds = self.encode_prompt(prompt, clip_skip=clip_skip)
        if self.use_cfg and cfg_scale > 1:
            negative_prompt_emb, negative_add_text_embeds = self.encode_prompt(negative_prompt, clip_skip=clip_skip)
        else:
            negative_prompt_emb, negative_add_text_embeds = None, None

        # ControlNet
        controlnet_params = self.prepare_controlnet_params(controlnet_params, h=height, w=width)

        # image_emb
        image_emb = None
        if self.ip_adapter is not None and self.redux is not None:
            raise Exception("ip-adapter and flux redux cannot be used at the same time")
        elif self.ip_adapter is not None:
            image_emb = self.ip_adapter(ref_image)
        elif self.redux is not None:
            image_prompt_embeds = self.redux(ref_image)
            positive_prompt_emb = torch.cat([positive_prompt_emb, image_prompt_embeds], dim=1)

        # Extra input
        image_ids, text_ids, guidance = self.prepare_extra_input(
            latents, positive_prompt_emb, guidance=flux_guidance_scale
        )

        # Denoise
        self.load_models_to_device([])
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=positive_prompt_emb,
                negative_prompt_emb=negative_prompt_emb,
                positive_add_text_embeds=positive_add_text_embeds,
                negative_add_text_embeds=negative_add_text_embeds,
                image_emb=image_emb,
                image_ids=image_ids,
                text_ids=text_ids,
                cfg_scale=cfg_scale,
                guidance=guidance,
                controlnet_params=controlnet_params,
                current_step=i,
                total_step=len(timesteps),
                use_cfg=self.use_cfg,
                batch_cfg=self.batch_cfg,
            )
            # Denoise
            latents = self.sampler.step(latents, noise_pred, i)
            # UI
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")
        # Decode image
        self.load_models_to_device(["vae_decoder"])
        vae_output = self.decode_image(latents)
        image = self.vae_output_to_image(vae_output)
        # Offload all models
        self.load_models_to_device([])
        return image
