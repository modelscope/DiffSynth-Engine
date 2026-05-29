from typing import Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from diffusers.configuration_utils import ConfigMixin

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import CONFIG_NAME
from diffsynth_engine.utils.load_utils import load_weights_into_module

logger = logging.get_logger(__name__)


class DiffusionModel(nn.Module, ConfigMixin):
    config_name = CONFIG_NAME

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        subfolder: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # load config
        config_dict = cls.load_config(model_path, subfolder=subfolder, local_files_only=True)

        # initialize model
        with init_empty_weights():
            model = cls.from_config(config_dict)

        load_weights_into_module(model, model_path, subfolder, device, dtype)
        model.to(device=device)
        return model


class AutoregressiveModel(nn.Module):
    config_name = CONFIG_NAME

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        subfolder: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # load config
        config = cls.config_class.from_pretrained(model_path, subfolder=subfolder, local_files_only=True)

        # initialize model
        with init_empty_weights():
            model = cls(config)

        load_weights_into_module(model, model_path, subfolder, device, dtype)
        model.to(device=device)
        return model
