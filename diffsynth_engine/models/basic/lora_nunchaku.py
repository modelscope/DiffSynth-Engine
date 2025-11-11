import torch
import torch.nn as nn
from collections import OrderedDict

from .lora import LoRA
from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    unpack_lowrank_weight,
)


class LoRASVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        origin_linear: SVDQW4A4Linear,
    ):
        super().__init__()

        self.origin_linear = origin_linear
        self.base_rank = self.origin_linear.rank
        self._lora_dict = OrderedDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.origin_linear(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.origin_linear, name)

    def _apply_lora_weights(self, name: str, down: torch.Tensor, up: torch.Tensor, alpha: int, scale: float, rank: int):
        final_scale = scale * (alpha / rank)

        up_scaled = (up * final_scale).to(
            dtype=self.origin_linear.proj_up.dtype, device=self.origin_linear.proj_up.device
        )
        down_final = down.to(dtype=self.origin_linear.proj_down.dtype, device=self.origin_linear.proj_down.device)

        with torch.no_grad():
            pd_packed = self.origin_linear.proj_down.data
            pu_packed = self.origin_linear.proj_up.data
            pd = unpack_lowrank_weight(pd_packed, down=True)
            pu = unpack_lowrank_weight(pu_packed, down=False)

            new_proj_down = torch.cat([pd, down_final], dim=0)
            new_proj_up = torch.cat([pu, up_scaled], dim=1)

            self.origin_linear.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
            self.origin_linear.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)

        current_total_rank = self.origin_linear.rank
        self.origin_linear.rank += rank
        self._lora_dict[name] = {"rank": rank, "alpha": alpha, "scale": scale, "start_idx": current_total_rank}

    def add_frozen_lora(
        self,
        name: str,
        scale: float,
        rank: int,
        alpha: int,
        up: torch.Tensor,
        down: torch.Tensor,
        device: str,
        dtype: torch.dtype,
        **kwargs,
    ):
        if name in self._lora_dict:
            raise ValueError(f"LoRA with name '{name}' already exists.")

        self._apply_lora_weights(name, down, up, alpha, scale, rank)

    def add_qkv_lora(
        self,
        name: str,
        scale: float,
        rank: int,
        alpha: int,
        q_up: torch.Tensor,
        q_down: torch.Tensor,
        k_up: torch.Tensor,
        k_down: torch.Tensor,
        v_up: torch.Tensor,
        v_down: torch.Tensor,
        device: str,
        dtype: torch.dtype,
        **kwargs,
    ):
        if name in self._lora_dict:
            raise ValueError(f"LoRA with name '{name}' already exists.")

        fused_down = torch.cat([q_down, k_down, v_down], dim=0)

        fused_rank = 3 * rank
        out_q, out_k = q_up.shape[0], k_up.shape[0]
        fused_up = torch.zeros((self.out_features, fused_rank), device=q_up.device, dtype=q_up.dtype)
        fused_up[:out_q, :rank] = q_up
        fused_up[out_q : out_q + out_k, rank : 2 * rank] = k_up
        fused_up[out_q + out_k :, 2 * rank :] = v_up

        self._apply_lora_weights(name, fused_down, fused_up, alpha, scale, rank)

    def modify_scale(self, name: str, scale: float):
        if name not in self._lora_dict:
            raise ValueError(f"LoRA name {name} not found in {self.__class__.__name__}")

        info = self._lora_dict[name]
        old_scale = info["scale"]

        if old_scale == scale:
            return

        if old_scale == 0:
            scale_factor = 0.0
        else:
            scale_factor = scale / old_scale

        with torch.no_grad():
            lora_rank = info["rank"]
            start_idx = info["start_idx"]
            end_idx = start_idx + lora_rank

            pu_packed = self.origin_linear.proj_up.data
            pu = unpack_lowrank_weight(pu_packed, down=False)
            pu[:, start_idx:end_idx] *= scale_factor

            self.origin_linear.proj_up.data = pack_lowrank_weight(pu, down=False)

        self._lora_dict[name]["scale"] = scale

    def clear(self, release_all_cpu_memory: bool = False):
        if not self._lora_dict:
            return

        with torch.no_grad():
            pd_packed = self.origin_linear.proj_down.data
            pu_packed = self.origin_linear.proj_up.data

            pd = unpack_lowrank_weight(pd_packed, down=True)
            pu = unpack_lowrank_weight(pu_packed, down=False)

            pd_reset = pd[: self.base_rank, :].clone()
            pu_reset = pu[:, : self.base_rank].clone()

            self.origin_linear.proj_down.data = pack_lowrank_weight(pd_reset, down=True)
            self.origin_linear.proj_up.data = pack_lowrank_weight(pu_reset, down=False)

            self.origin_linear.rank = self.base_rank

        self._lora_dict.clear()


class LoRAAWQW4A16Linear(nn.Module):
    def __init__(self, origin_linear: AWQW4A16Linear):
        super().__init__()
        self.origin_linear = origin_linear
        self._lora_dict = OrderedDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_output = self.origin_linear(x)

        for name, lora in self._lora_dict.items():
            quantized_output += lora(x.to(lora.dtype)).to(quantized_output.dtype)

        return quantized_output

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.origin_linear, name)

    def add_lora(
        self,
        name: str,
        scale: float,
        rank: int,
        alpha: int,
        up: torch.Tensor,
        down: torch.Tensor,
        device: str,
        dtype: torch.dtype,
        **kwargs,
    ):
        up_linear = nn.Linear(rank, self.out_features, bias=False, device="meta", dtype=dtype).to_empty(device=device)
        down_linear = nn.Linear(self.in_features, rank, bias=False, device="meta", dtype=dtype).to_empty(device=device)

        up_linear.weight.data = up.reshape(self.out_features, rank)
        down_linear.weight.data = down.reshape(rank, self.in_features)

        lora = LoRA(scale, rank, alpha, up_linear, down_linear, device, dtype)
        self._lora_dict[name] = lora

    def modify_scale(self, name: str, scale: float):
        if name not in self._lora_dict:
            raise ValueError(f"LoRA name {name} not found in {self.__class__.__name__}")
        self._lora_dict[name].scale = scale

    def add_frozen_lora(self, *args, **kwargs):
        raise NotImplementedError("Frozen LoRA (merging weights) is not supported for AWQW4A16Linear.")

    def clear(self, *args, **kwargs):
        self._lora_dict.clear()


def patch_nunchaku_model_for_lora(model: nn.Module):
    def _recursive_patch(module: nn.Module):
        for name, child_module in module.named_children():
            replacement = None
            if isinstance(child_module, AWQW4A16Linear):
                replacement = LoRAAWQW4A16Linear(child_module)
            elif isinstance(child_module, SVDQW4A4Linear):
                replacement = LoRASVDQW4A4Linear(child_module)

            if replacement:
                setattr(module, name, replacement)
            else:
                _recursive_patch(child_module)

    _recursive_patch(model)
