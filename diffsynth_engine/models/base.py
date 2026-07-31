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

    _keep_in_fp32_modules: list[str] | None = None
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

        # avoid precision loss
        if dtype is not None and dtype != torch.float32 and cls._keep_in_fp32_modules:
            state_dict = load_model_weights(
                model_path,
                subfolder,
                device,
                dtype=None,
            )
            for k, v in state_dict.items():
                if any(m in k.split(".") for m in cls._keep_in_fp32_modules):
                    state_dict[k] = v.to(dtype=torch.float32)
                else:
                    state_dict[k] = v.to(dtype=dtype)
        else:
            state_dict = load_model_weights(
                model_path,
                subfolder,
                device,
                dtype,
            )

        # drop unexpected keys
        if cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in state_dict if any(pat in k for pat in cls._keys_to_ignore_on_load_unexpected)]
            for k in unexpected_keys:
                del state_dict[k]
            if unexpected_keys:
                logger.info(
                    f"Dropped {len(unexpected_keys)} unexpected key(s) matching "
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
