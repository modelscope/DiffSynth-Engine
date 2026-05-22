from typing import Type

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from tqdm import tqdm

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.distributed.parallel_state import get_global_rank, is_world_group_initialized
from diffsynth_engine.forward_context import set_forward_context
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.load_utils import fix_state_dict_key, load_model_weights

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
    def init_transformer(model_cls: Type[nn.Module], pipeline_config: PipelineConfig, empty_weights: bool = False):
        use_fsdp = pipeline_config.use_fsdp and is_world_group_initialized()

        with set_forward_context(attn_type=pipeline_config.attn_type):
            with init_empty_weights():
                config = model_cls.load_config(
                    pipeline_config.model_path,
                    subfolder="transformer",
                    local_files_only=True,
                )
                model = model_cls.from_config(config)

        if empty_weights:
            return model

        if use_fsdp:
            for block in model.transformer_blocks:
                fully_shard(block)
            fully_shard(model)

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="transformer",
            device="cpu" if use_fsdp else pipeline_config.device,
            dtype=pipeline_config.model_dtype,
            broadcast_from_rank0=not use_fsdp,
        )

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
        return model

    @staticmethod
    def init_text_encoder(
        model_cls: Type[nn.Module],
        pipeline_config: PipelineConfig,
        key_mapping: dict = None,
        empty_weights: bool = False,
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
            for layer in model.model.language_model.layers:
                fully_shard(layer)
            fully_shard(model)

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="text_encoder",
            device="cpu" if use_fsdp else pipeline_config.device,
            dtype=pipeline_config.text_encoder_dtype,
            broadcast_from_rank0=not use_fsdp,
        )

        if key_mapping:
            state_dict = fix_state_dict_key(state_dict, key_mapping)

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
        return model

    @staticmethod
    def init_vae(model_cls: Type[nn.Module], pipeline_config: PipelineConfig, empty_weights: bool = False):
        with init_empty_weights():
            config = model_cls.load_config(
                pipeline_config.model_path,
                subfolder="vae",
                local_files_only=True,
            )
            model = model_cls.from_config(config)

        if empty_weights:
            return model

        state_dict = load_model_weights(
            pipeline_config.model_path,
            subfolder="vae",
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

    # TODO: preprocess & postprocess & LoRA
