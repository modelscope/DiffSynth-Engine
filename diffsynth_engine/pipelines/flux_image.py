import re
import json
import torch
import torch.distributed as dist
import math
from einops import rearrange
from typing import Callable, Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from PIL import Image
from diffsynth_engine.models.flux import (
    FluxTextEncoder1,
    FluxTextEncoder2,
    FluxVAEDecoder,
    FluxVAEEncoder,
    FluxDiT,
    FluxDiTFBCache,
    flux_dit_config,
    flux_text_encoder_config,
)
from diffsynth_engine.configs import (
    FluxPipelineConfig,
    FluxStateDicts,
    ControlType,
    ControlNetParams,
)
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.pipelines.utils import accumulate, calculate_shift
from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH, FLUX_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.platform import empty_cache
from diffsynth_engine.utils.constants import FLUX_DIT_CONFIG_FILE

logger = logging.get_logger(__name__)

with open(FLUX_DIT_CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

PREFERRED_KONTEXT_RESOLUTIONS = config["preferred_kontext_resolutions"]


class FluxLoRAConverter(LoRAStateDictConverter):
    def _from_kohya(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        flux_dim = 3072
        dit_rename_dict = flux_dit_config["civitai"]["rename_dict"]
        dit_suffix_rename_dict = flux_dit_config["civitai"]["suffix_rename_dict"]
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
            elif "lora_te1_text_model_encoder_layers_" in key:
                name = key.replace("lora_te1_text_model_encoder_layers_", "")
                name = name.replace(".alpha", "")
                layer_id, layer_type = name.split("_", 1)
                layer_type = layer_type.replace("self_attn_", "self_attn.").replace("mlp_", "mlp.")
                rename = ".".join(["encoders", layer_id, clip_attn_rename_dict[layer_type]])

                lora_args = {}
                lora_args["alpha"] = param
                lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                lora_args["rank"] = lora_args["up"].shape[1]
                te_dict[rename] = lora_args
            else:
                raise ValueError(f"Unsupported key: {key}")
        return {"dit": dit_dict, "text_encoder_1": te_dict}

    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        rename_dict = config["diffusers"]["rename_dict"]
        lora_state_dict_ = {}
        dim = 3072
        for key, param in list(lora_state_dict.items()):
            origin_key = key
            if "lora_A.weight" not in key:
                continue
            key = key.replace("transformer.", "")
            if key == "proj_out.lora_A.weight":
                key = key.replace("proj_out", "final_proj_out")
                lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                    origin_key.replace("lora_A", "lora_B")
                ]
                lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))

            elif "single_transformer_blocks" in key:  # transformer.single_transformer_blocks.0.attn.to_k.weight
                key = key.replace(
                    "single_transformer_blocks", "single_blocks"
                )  # single_transformer_blocks.0.attn.to_k.weight
                if "attn.to_q.lora_A" in key:
                    A_q = param
                    A_k = lora_state_dict[origin_key.replace("to_q", "to_k")]
                    A_v = lora_state_dict[origin_key.replace("to_q", "to_v")]
                    A_qkv = torch.cat([A_q, A_k, A_v], dim=0)

                    B_q = lora_state_dict[origin_key.replace("to_q", "to_q").replace("lora_A", "lora_B")]
                    B_k = lora_state_dict[origin_key.replace("to_q", "to_k").replace("lora_A", "lora_B")]
                    B_v = lora_state_dict[origin_key.replace("to_q", "to_v").replace("lora_A", "lora_B")]
                    B_qkv = torch.block_diag(B_q, B_k, B_v)

                    lora_state_dict_[key.replace("to_q.lora_A", "to_qkv.lora_A")] = A_qkv
                    lora_state_dict_[key.replace("to_q.lora_A", "to_qkv.lora_B")] = B_qkv

                    lora_state_dict.pop(origin_key.replace("to_q", "to_k"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_v"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_q"))

                    lora_state_dict.pop(origin_key.replace("to_q", "to_q").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_k").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_v").replace("lora_A", "lora_B"))

                elif "norm.linear.lora_A" in key:
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]

                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                elif "proj_mlp.lora_A" in key:
                    key = key.replace("proj_mlp", "mlp.0")
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]

                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                elif "proj_out.lora_A" in key:
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
            elif "transformer_blocks" in key:
                key = key.replace("transformer_blocks", "blocks")
                if "attn.to_q.lora_A" in key:
                    A_q = param
                    A_k = lora_state_dict[origin_key.replace("to_q", "to_k")]
                    A_v = lora_state_dict[origin_key.replace("to_q", "to_v")]
                    A_qkv = torch.cat([A_q, A_k, A_v], dim=0)

                    B_q = lora_state_dict[origin_key.replace("to_q", "to_q").replace("lora_A", "lora_B")]
                    B_k = lora_state_dict[origin_key.replace("to_q", "to_k").replace("lora_A", "lora_B")]
                    B_v = lora_state_dict[origin_key.replace("to_q", "to_v").replace("lora_A", "lora_B")]
                    B_qkv = torch.block_diag(B_q, B_k, B_v)

                    lora_state_dict_[key.replace("to_q.lora_A", "a_to_qkv.lora_A")] = A_qkv
                    lora_state_dict_[key.replace("to_q.lora_A", "a_to_qkv.lora_B")] = B_qkv

                    lora_state_dict.pop(origin_key.replace("to_q", "to_k"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_v"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_q"))

                    lora_state_dict.pop(origin_key.replace("to_q", "to_q").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_k").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_v").replace("lora_A", "lora_B"))

                    lora_state_dict_[key.replace("to_q.lora_A", "a_to_out.lora_A")] = lora_state_dict[
                        origin_key.replace("to_q.lora_A", "to_out.0.lora_A")
                    ]
                    lora_state_dict_[key.replace("to_q.lora_A", "a_to_out.lora_B")] = lora_state_dict[
                        origin_key.replace("to_q.lora_A", "to_out.0.lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("to_q", "to_out.0").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("to_q", "to_out.0").replace("lora_A", "lora_A"))
                elif "attn.add_q_proj.lora_A" in key:
                    A_q = param
                    A_k = lora_state_dict[origin_key.replace("add_q_proj", "add_k_proj")]
                    A_v = lora_state_dict[origin_key.replace("add_q_proj", "add_v_proj")]
                    A_qkv = torch.cat([A_q, A_k, A_v], dim=0)

                    B_q = lora_state_dict[origin_key.replace("add_q_proj", "add_q_proj").replace("lora_A", "lora_B")]
                    B_k = lora_state_dict[origin_key.replace("add_q_proj", "add_k_proj").replace("lora_A", "lora_B")]
                    B_v = lora_state_dict[origin_key.replace("add_q_proj", "add_v_proj").replace("lora_A", "lora_B")]
                    B_qkv = torch.block_diag(B_q, B_k, B_v)

                    lora_state_dict_[key.replace("add_q_proj.lora_A", "b_to_qkv.lora_A")] = A_qkv
                    lora_state_dict_[key.replace("add_q_proj.lora_A", "b_to_qkv.lora_B")] = B_qkv

                    lora_state_dict.pop(origin_key.replace("add_q_proj", "add_q_proj"))
                    lora_state_dict.pop(origin_key.replace("add_q_proj", "add_k_proj"))
                    lora_state_dict.pop(origin_key.replace("add_q_proj", "add_v_proj"))

                    lora_state_dict.pop(origin_key.replace("add_q_proj", "add_q_proj").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("add_q_proj", "add_k_proj").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("add_q_proj", "add_v_proj").replace("lora_A", "lora_B"))

                    lora_state_dict_[key.replace("add_q_proj.lora_A", "b_to_out.lora_A")] = lora_state_dict[
                        origin_key.replace("add_q_proj.lora_A", "to_add_out.lora_A")
                    ]
                    lora_state_dict_[key.replace("add_q_proj.lora_A", "b_to_out.lora_B")] = lora_state_dict[
                        origin_key.replace("add_q_proj.lora_A", "to_add_out.lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("add_q_proj", "to_add_out").replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("add_q_proj", "to_add_out").replace("lora_A", "lora_A"))
                elif "ff.net.0.proj" in key:
                    key = key.replace("ff.net.0.proj", rename_dict["ff.net.0.proj"])
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                elif "ff.net.2" in key:
                    key = key.replace("ff.net.2", rename_dict["ff.net.2"])
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                elif "ff_context.net.0.proj" in key:
                    key = key.replace("ff_context.net.0.proj", rename_dict["ff_context.net.0.proj"])
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                elif "ff_context.net.2" in key:
                    key = key.replace("ff_context.net.2", rename_dict["ff_context.net.2"])
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ]
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                elif "norm1.linear" in key:
                    key = key.replace("norm1.linear", "norm_msa_a.linear")
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ][: 3 * dim]

                    key = key.replace("norm_msa_a.linear", "norm_mlp_a.linear")
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ][3 * dim :]

                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))
                elif "norm1_context.linear" in key:
                    key = key.replace("norm1_context.linear", "norm_msa_b.linear")
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ][: 3 * dim]

                    key = key.replace("norm_msa_b.linear", "norm_mlp_b.linear")
                    lora_state_dict_[key.replace("lora_A", "lora_A")] = param
                    lora_state_dict_[key.replace("lora_A", "lora_B")] = lora_state_dict[
                        origin_key.replace("lora_A", "lora_B")
                    ][3 * dim :]

                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_B"))
                    lora_state_dict.pop(origin_key.replace("lora_A", "lora_A"))

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

        for key, param in lora_state_dict_.items():
            origin_key = key
            if "lora_A.weight" not in key:
                continue
            lora_args = {}
            lora_args["down"] = param
            lora_args["up"] = lora_state_dict_[origin_key.replace("lora_A.weight", "lora_B.weight")]
            lora_args["rank"] = lora_args["up"].shape[1]
            alpha_key = origin_key.replace("lora_A.weight", "alpha").replace("lora_up.weight", "alpha")
            if alpha_key in lora_state_dict_:
                alpha = lora_state_dict_[alpha_key]
            else:
                alpha = lora_args["rank"]  # 如果alpha不存在，则取alpha/rank = 1
            lora_args["alpha"] = alpha
            key = key.replace(".lora_A.weight", "")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if "lora_te" in key or "lora_unet" in key:
            return self._from_kohya(lora_state_dict)
        elif key.startswith("transformer"):
            return self._from_diffusers(lora_state_dict)
        raise ValueError(f"Unsupported key: {key}")


class FluxImagePipeline(BasePipeline):
    lora_converter = FluxLoRAConverter()

    def __init__(
        self,
        config: FluxPipelineConfig,
        tokenizer: CLIPTokenizer,
        tokenizer_2: T5TokenizerFast,
        text_encoder_1: FluxTextEncoder1,
        text_encoder_2: FluxTextEncoder2,
        dit: Union[FluxDiT, FluxDiTFBCache],
        vae_decoder: FluxVAEDecoder,
        vae_encoder: FluxVAEEncoder,
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
        self.ip_adapter = None
        self.redux = None
        self.model_names = [
            "text_encoder_1",
            "text_encoder_2",
            "dit",
            "vae_decoder",
            "vae_encoder",
        ]

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | FluxPipelineConfig) -> "FluxImagePipeline":
        if isinstance(model_path_or_config, str):
            config = FluxPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        logger.info(f"loading state dict from {config.model_path} ...")
        model_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        if config.vae_path is None:
            config.vae_path = fetch_model("muse/FLUX.1-dev-fp8", path="ae-bf16.safetensors")
        logger.info(f"loading state dict from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        clip_state_dict = None
        if config.clip_path is None and config.load_text_encoder:
            config.clip_path = fetch_model("muse/FLUX.1-dev-fp8", path="clip-bf16.safetensors")
        if config.load_text_encoder:
            logger.info(f"loading state dict from {config.clip_path} ...")
            clip_state_dict = cls.load_model_checkpoint(config.clip_path, device="cpu", dtype=config.clip_dtype)

        t5_state_dict = None
        if config.t5_path is None and config.load_text_encoder:
            config.t5_path = fetch_model(
                "muse/FLUX.1-dev-fp8",
                path=["t5-fp8-00001-of-00002.safetensors", "t5-fp8-00002-of-00002.safetensors"],
            )
        if config.load_text_encoder:
            logger.info(f"loading state dict from {config.t5_path} ...")
            t5_state_dict = cls.load_model_checkpoint(config.t5_path, device="cpu", dtype=config.t5_dtype)

        state_dicts = FluxStateDicts(
            model=model_state_dict,
            vae=vae_state_dict,
            clip=clip_state_dict,
            t5=t5_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: FluxStateDicts, config: FluxPipelineConfig) -> "FluxImagePipeline":
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
    def _from_state_dict(cls, state_dicts: FluxStateDicts, config: FluxPipelineConfig) -> "FluxImagePipeline":
        init_device = "cpu" if config.offload_mode is not None else config.device
        if config.load_text_encoder:
            tokenizer = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)
            tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)
            with LoRAContext():
                text_encoder_1 = FluxTextEncoder1.from_state_dict(
                    state_dicts.clip, device=init_device, dtype=config.clip_dtype
                )
            text_encoder_2 = FluxTextEncoder2.from_state_dict(state_dicts.t5, device=init_device, dtype=config.t5_dtype)

        vae_decoder = FluxVAEDecoder.from_state_dict(state_dicts.vae, device=init_device, dtype=config.vae_dtype)
        vae_encoder = FluxVAEEncoder.from_state_dict(state_dicts.vae, device=init_device, dtype=config.vae_dtype)

        with LoRAContext():
            if config.use_fbcache:
                dit = FluxDiTFBCache.from_state_dict(
                    state_dicts.model,
                    device=("cpu" if config.use_fsdp else init_device),
                    dtype=config.model_dtype,
                    in_channel=config.control_type.get_in_channel(),
                    relative_l1_threshold=config.fbcache_relative_l1_threshold,
                )
            else:
                dit = FluxDiT.from_state_dict(
                    state_dicts.model,
                    device=("cpu" if config.use_fsdp else init_device),
                    dtype=config.model_dtype,
                    in_channel=config.control_type.get_in_channel(),
                )
            if config.use_fp8_linear:
                enable_fp8_linear(dit)

        pipe = cls(
            config=config,
            tokenizer=tokenizer if config.load_text_encoder else None,
            tokenizer_2=tokenizer_2 if config.load_text_encoder else None,
            text_encoder_1=text_encoder_1 if config.load_text_encoder else None,
            text_encoder_2=text_encoder_2 if config.load_text_encoder else None,
            dit=dit,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
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
                model_names=["text_encoder_2"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.use_torch_compile:
            pipe.compile()
        return pipe

    def update_weights(self, state_dicts: FluxStateDicts) -> None:
        self.update_component(self.dit, state_dicts.model, self.config.device, self.config.model_dtype)
        self.update_component(self.text_encoder_1, state_dicts.clip, self.config.device, self.config.clip_dtype)
        self.update_component(self.text_encoder_2, state_dicts.t5, self.config.device, self.config.t5_dtype)
        self.update_component(self.vae_decoder, state_dicts.vae, self.config.device, self.config.vae_dtype)
        self.update_component(self.vae_encoder, state_dicts.vae, self.config.device, self.config.vae_dtype)

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
        self.text_encoder_1.unload_loras()
        self.text_encoder_2.unload_loras()

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
        image_ids = self.dit.prepare_image_ids(latents)
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
        batch_cfg: bool = False,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                latents=latents,
                timestep=timestep,
                prompt_emb=positive_prompt_emb,
                add_text_embeds=positive_add_text_embeds,
                image_emb=image_emb,
                image_ids=image_ids,
                text_ids=text_ids,
                guidance=guidance,
                controlnet_params=controlnet_params,
                current_step=current_step,
                total_step=total_step,
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents=latents,
                timestep=timestep,
                prompt_emb=positive_prompt_emb,
                add_text_embeds=positive_add_text_embeds,
                image_emb=image_emb,
                image_ids=image_ids,
                text_ids=text_ids,
                guidance=guidance,
                controlnet_params=controlnet_params,
                current_step=current_step,
                total_step=total_step,
            )
            negative_noise_pred = self.predict_noise(
                latents=latents,
                timestep=timestep,
                prompt_emb=negative_prompt_emb,
                add_text_embeds=negative_add_text_embeds,
                image_emb=image_emb,
                image_ids=image_ids,
                text_ids=text_ids,
                guidance=guidance,
                controlnet_params=controlnet_params,
                current_step=current_step,
                total_step=total_step,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            add_text_embeds = torch.cat([positive_add_text_embeds, negative_add_text_embeds], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            image_emb = torch.cat([image_emb, image_emb], dim=0) if image_emb is not None else None
            image_ids = torch.cat([image_ids, image_ids], dim=0)
            text_ids = torch.cat([text_ids, text_ids], dim=0)
            guidance = torch.cat([guidance, guidance], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_emb,
                add_text_embeds=add_text_embeds,
                image_emb=image_emb,
                image_ids=image_ids,
                text_ids=text_ids,
                guidance=guidance,
                controlnet_params=controlnet_params,
                current_step=current_step,
                total_step=total_step,
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
        height, width = latents.shape[2:]
        latents = self.dit.patchify(latents)
        image_seq_len = latents.shape[1]

        double_block_output, single_block_output = None, None
        if self.config.control_type == ControlType.normal:
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
        elif self.config.control_type == ControlType.bfl_kontext:
            for idx, controlnet_param in enumerate(controlnet_params):
                control_latents = controlnet_param.image * controlnet_param.scale
                control_image_ids = self.dit.prepare_image_ids(control_latents)
                control_image_ids[..., 0] = idx + 1
                control_latents = self.dit.patchify(control_latents)
                latents = torch.cat((latents, control_latents), dim=1)
                image_ids = torch.cat((image_ids, control_image_ids), dim=1)
        else:
            controlnet_param = controlnet_params[0]
            control_latents = controlnet_param.image * controlnet_param.scale
            control_latents = self.dit.patchify(control_latents)
            latents = torch.cat((latents, control_latents), dim=2)

        latents = latents.to(self.dtype)
        self.load_models_to_device(["dit"])

        attn_kwargs = self.get_attn_kwargs(latents)
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
            attn_kwargs=attn_kwargs,
        )
        noise_pred = noise_pred[:, :image_seq_len]
        noise_pred = self.dit.unpatchify(noise_pred, height, width)
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
            image = self.preprocess_image(input_image).to(device=self.device)
            latents = self.encode_image(image)
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
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

    def prepare_masked_latent(self, image: Image.Image, mask: Image.Image | None, height: int, width: int):
        self.load_models_to_device(["vae_encoder"])
        if mask is None:
            if self.config.control_type == ControlType.bfl_kontext:
                width, height = image.size
                aspect_ratio = width / height
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
                width, height = 16 * (width // 16), 16 * (height // 16)
            image = image.resize((width, height))
            image = self.preprocess_image(image).to(device=self.device)
            latent = self.encode_image(image)
        else:
            if self.config.control_type == ControlType.normal:
                image = image.resize((width, height))
                mask = mask.resize((width, height))
                image = self.preprocess_image(image).to(device=self.device)
                mask = self.preprocess_mask(mask).to(device=self.device)
                masked_image = image.clone()
                masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1
                latent = self.encode_image(masked_image)
                mask = torch.nn.functional.interpolate(mask, size=(latent.shape[2], latent.shape[3])).to(latent.dtype)
                mask = 1 - mask
                latent = torch.cat([latent, mask], dim=1)
            elif self.config.control_type == ControlType.bfl_fill:
                image = image.resize((width, height))
                mask = mask.resize((width, height))
                image = self.preprocess_image(image).to(device=self.device)
                mask = self.preprocess_mask(mask).to(device=self.device)
                image = image * (1 - mask)
                image = self.encode_image(image)
                mask = rearrange(mask, "b 1 (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
                latent = torch.cat((image, mask), dim=1)
            else:
                raise ValueError(f"Unsupported mask latent prepare for controlnet type: {self.config.control_type}")
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
            control_condition = param.model.patchify(param.image)
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

            attn_kwargs = self.get_attn_kwargs(latents)
            double_block_output, single_block_output = param.model(
                hidden_states=latents,
                control_condition=control_condition,
                control_scale=current_scale,
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=add_text_embeds,
                image_ids=image_ids,
                text_ids=text_ids,
                guidance=guidance,
                attn_kwargs=attn_kwargs,
            )
            if self.offload_mode is not None:
                param.model.to("cpu")
                empty_cache()
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
        if isinstance(self.dit, FluxDiTFBCache):
            self.dit.refresh_cache_status(num_inference_steps)
        if not isinstance(controlnet_params, list):
            controlnet_params = [controlnet_params]
        if self.config.control_type in [ControlType.bfl_control, ControlType.bfl_fill]:
            assert controlnet_params and len(controlnet_params) == 1, (
                "bfl_controlnet or bfl_fill must have one controlnet"
            )

        if input_image is not None:
            width, height = input_image.size
        self.validate_image_size(height, width, minimum=64, multiple_of=16)

        if dist.is_initialized() and seed is None:
            raise ValueError("must provide a seed when parallelism is enabled")
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
        self.sampler.initialize(sigmas=sigmas)

        # Encode prompts
        self.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        positive_prompt_emb, positive_add_text_embeds = self.encode_prompt(prompt, clip_skip=clip_skip)
        if cfg_scale > 1.0:
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
            image_emb = self.ip_adapter.encode_image(ref_image)
        elif self.redux is not None:
            ref_prompt_embeds = self.redux(ref_image)
            flattened_ref_emb = ref_prompt_embeds.view(1, -1, ref_prompt_embeds.size(-1))
            positive_prompt_emb = torch.cat([positive_prompt_emb, flattened_ref_emb], dim=1)

        # Extra input
        image_ids, text_ids, guidance = self.prepare_extra_input(
            latents, positive_prompt_emb, guidance=flux_guidance_scale
        )

        # Denoise
        self.load_models_to_device([])
        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
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
                batch_cfg=self.config.batch_cfg,
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
