import os
import torch
import torch.nn as nn
from typing import Dict, Union, List, Any
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.models.basic.lora import LoRALinear, LoRAConv2d


class StateDictConverter:
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return state_dict


class PreTrainedModel(nn.Module):
    converter = StateDictConverter()
    _supports_parallelization = False

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        state_dict = self.converter.convert(state_dict)
        super().load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, os.PathLike],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        state_dict = load_file(pretrained_model_path)
        return cls.from_state_dict(state_dict, device=device, dtype=dtype, **kwargs)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, **kwargs):
        model = cls(device="meta", dtype=dtype, **kwargs)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def load_loras(self, lora_args: List[Dict[str, Any]], fused: bool = True):
        for args in lora_args:
            key = args["key"]
            module = self.get_submodule(key)
            if not isinstance(module, (LoRALinear, LoRAConv2d)):
                raise ValueError(f"Unsupported lora key: {key}")
            if fused:
                module.add_frozen_lora(**args)
            else:
                module.add_lora(**args)

    def unload_loras(self):
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()

    def get_tp_plan(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support TP")

    def get_fsdp_module_cls(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support FSDP")


def split_suffix(name: str):
    suffix_list = [
        ".lora_up.weight",
        ".lora_down.weight",
        ".weight",
        ".bias",
        ".alpha",
    ]
    for suffix in suffix_list:
        if name.endswith(suffix):
            return name.replace(suffix, ""), suffix
    return name, ""
