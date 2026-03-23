from typing import Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from diffusers.configuration_utils import ConfigMixin

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import CONFIG_NAME
from diffsynth_engine.utils.load_utils import load_model_weights

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

        # load model weights
        state_dict = load_model_weights(model_path, subfolder, device, dtype)
        model.load_state_dict(state_dict, strict=True, assign=True)
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

        # load model weights
        state_dict = load_model_weights(model_path, subfolder, device, dtype)
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device=device)
        return model
