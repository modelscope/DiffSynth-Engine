from typing import Type

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from tqdm import tqdm

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.distributed.parallel_state import (
    get_global_rank,
    get_tensor_model_parallel_world_size,
    get_ulysses_parallel_world_size,
    is_sp_group_initialized,
    is_tp_group_initialized,
    is_world_group_initialized,
)
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.load_utils import fix_state_dict_key, load_model_weights, prepare_model_weights

logger = logging.get_logger(__name__)


class Pipeline:
    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config
        self.device = pipeline_config.device

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | PipelineConfig):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def compile_transformer_blocks(model: nn.Module) -> nn.Module:
        repeated_blocks = getattr(model, "_repeated_blocks", None)
        if not repeated_blocks:
            raise ValueError(f"`_repeated_blocks` is not defined for {type(model).__name__}")

        has_compiled_region = False
        for submodule in model.modules():
            if submodule.__class__.__name__ in repeated_blocks:
                submodule.compile()
                has_compiled_region = True

        if not has_compiled_region:
            raise ValueError(
                f"None of the repeated block classes {repeated_blocks} were found in {type(model).__name__}"
            )
        return model

    @classmethod
    def init_transformer(
        cls,
        model_cls: Type[nn.Module],
        pipeline_config: PipelineConfig,
        empty_weights: bool = False,
        subfolder: str = "transformer",
    ):
        use_fsdp = pipeline_config.use_fsdp and is_world_group_initialized()

        with set_forward_context(attn_type=pipeline_config.attn_type):
            with init_empty_weights():
                config = model_cls.load_config(
                    pipeline_config.model_path,
                    subfolder=subfolder,
                    local_files_only=True,
                )
                model = model_cls.from_config(config)

        if empty_weights:
            return model

        if use_fsdp:
            repeated_blocks = getattr(model, "_repeated_blocks", None)
            if not repeated_blocks:
                raise ValueError(f"`_repeated_blocks` is not defined for {type(model).__name__}")

            for submodule in model.modules():
                if submodule.__class__.__name__ in repeated_blocks:
                    fully_shard(submodule)
            fully_shard(model)

        tp_size = get_tensor_model_parallel_world_size() if is_tp_group_initialized() else 1
        if tp_size > 1:
            num_heads = getattr(model.config, "num_attention_heads", None)
            if num_heads is not None and num_heads % tp_size != 0:
                raise ValueError(f"num_attention_heads ({num_heads}) must be divisible by tp_degree ({tp_size})")
            if num_heads is not None and is_sp_group_initialized():
                ulysses_size = get_ulysses_parallel_world_size()
                heads_per_tp_partition = num_heads // tp_size
                if ulysses_size > 1 and heads_per_tp_partition % ulysses_size != 0:
                    raise ValueError(
                        f"With tp_degree={tp_size} and sp_ulysses_degree={ulysses_size}, "
                        f"heads_per_tp_partition ({heads_per_tp_partition}) must be divisible by "
                        "sp_ulysses_degree."
                    )

        load_device = "cpu" if use_fsdp else pipeline_config.device
        model_dtype = pipeline_config.model_dtype
        keep_in_fp32_modules = getattr(model_cls, "_keep_in_fp32_modules", None)
        keep_in_fp32 = bool(model_dtype is not None and model_dtype != torch.float32 and keep_in_fp32_modules)
        load_dtype = None if keep_in_fp32 else model_dtype
        state_dict = prepare_model_weights(
            model,
            pipeline_config.model_path,
            subfolder=subfolder,
            device=load_device,
            dtype=load_dtype,
            broadcast_from_rank0=not use_fsdp,
        )

        if keep_in_fp32:
            # avoid precision loss: keep modules (time embedder, modulation, norms) in fp32
            for k, v in state_dict.items():
                if any(m in k.split(".") for m in keep_in_fp32_modules):
                    state_dict[k] = v.to(dtype=torch.float32)
                else:
                    state_dict[k] = v.to(dtype=model_dtype)

        if use_fsdp:
            set_model_state_dict(
                model,
                state_dict,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            )
        else:
            model.load_state_dict(state_dict, strict=True, assign=True)
            model.to(device=pipeline_config.device)

        del state_dict
        if pipeline_config.use_torch_compile:
            model = cls.compile_transformer_blocks(model)
        return model

    @staticmethod
    def init_text_encoder(
        model_cls: Type[nn.Module],
        pipeline_config: PipelineConfig,
        key_mapping: dict = None,
        empty_weights: bool = False,
        strict: bool = True,
    ):
        use_fsdp = pipeline_config.use_fsdp and is_world_group_initialized()

        with init_empty_weights():
            config = model_cls.config_class.from_pretrained(
                pipeline_config.model_path,
                subfolder="text_encoder",
                local_files_only=True,
            )
            model = model_cls(config)

        if empty_weights:
            return model

        if use_fsdp:
            if hasattr(model, "tie_weights"):
                model.tie_weights()

            no_split_modules = getattr(model, "_no_split_modules", None)
            if not no_split_modules:
                raise ValueError(f"`_no_split_modules` is not defined for {type(model).__name__}")

            for submodule in model.modules():
                if submodule.__class__.__name__ in no_split_modules:
                    fully_shard(submodule)
            fully_shard(model)

        load_device = "cpu" if use_fsdp else pipeline_config.device
        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="text_encoder",
            device=load_device,
            dtype=pipeline_config.text_encoder_dtype,
            broadcast_from_rank0=not use_fsdp,
        )

        if key_mapping:
            state_dict = fix_state_dict_key(state_dict, key_mapping)

        if use_fsdp:
            set_model_state_dict(
                model,
                state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=strict,
                ),
            )
        else:
            model.load_state_dict(state_dict, strict=strict, assign=True)
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            model.to(device=pipeline_config.device)

        del state_dict
        return model

    @staticmethod
    def init_vae(
        model_cls: Type[nn.Module],
        pipeline_config: PipelineConfig,
        empty_weights: bool = False,
        subfolder: str = "vae",
    ):
        with init_empty_weights():
            config = model_cls.load_config(
                pipeline_config.model_path,
                subfolder=subfolder,
                local_files_only=True,
            )
            model = model_cls.from_config(config)

        if empty_weights:
            return model

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder=subfolder,
            device=pipeline_config.device,
            dtype=pipeline_config.vae_dtype,
        )
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device=pipeline_config.device)

        del state_dict
        return model

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(self._progress_bar_config)
        if "disable" not in progress_bar_config:
            is_rank_zero = not is_world_group_initialized() or get_global_rank() == 0
            progress_bar_config["disable"] = not is_rank_zero

        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
