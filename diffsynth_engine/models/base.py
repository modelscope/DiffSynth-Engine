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

    # This is identical to diffusers' ModelMixin._keep_in_fp32_modules.
    _keep_in_fp32_modules: list[str] | None = None

    # ModelMixin._keys_to_ignore_on_load_unexpected.
    _keys_to_ignore_on_load_unexpected: list[str] | None = None

    @property
    def dtype(self) -> torch.dtype:
        param = next(self.parameters(), None)
        if param is None:
            raise RuntimeError(f"{type(self).__name__} has no parameters, cannot determine dtype")
        return param.dtype

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

        # avoids precision loss
        if dtype is not None and dtype != torch.float32 and cls._keep_in_fp32_modules:
            state_dict = load_model_weights(model_path, subfolder, device, dtype=None)
            for key in state_dict:
                if any(m in key.split(".") for m in cls._keep_in_fp32_modules):
                    state_dict[key] = state_dict[key].to(device=device, dtype=torch.float32)
                else:
                    state_dict[key] = state_dict[key].to(device=device, dtype=dtype)
        else:
            state_dict = load_model_weights(model_path, subfolder, device, dtype)

        # Filter out unexpected keys that the model explicitly ignores
        if cls._keys_to_ignore_on_load_unexpected:
            keys_to_remove = [
                key for key in state_dict if any(pattern in key for pattern in cls._keys_to_ignore_on_load_unexpected)
            ]
            for key in keys_to_remove:
                del state_dict[key]
            if keys_to_remove:
                logger.info(
                    f"Dropped {len(keys_to_remove)} unexpected key(s) matching "
                    f"{cls._keys_to_ignore_on_load_unexpected} from state_dict."
                )

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
