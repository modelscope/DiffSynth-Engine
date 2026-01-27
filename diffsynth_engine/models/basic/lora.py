import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from typing import Union
from collections import OrderedDict
from contextlib import contextmanager


class LoRA(nn.Module):
    def __init__(
        self,
        scale: float,
        rank: int,
        alpha: int,
        up: Union[nn.Linear, nn.Conv2d, torch.Tensor],
        down: Union[nn.Linear, nn.Conv2d, torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.scale = scale
        self.rank = rank
        self.alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        self.up = up.to(device=device, dtype=dtype)
        self.down = down.to(device=device, dtype=dtype)

    def forward(self, x):
        if isinstance(self.up, torch.Tensor) and isinstance(self.down, torch.Tensor):
            return self.scale * (self.alpha / self.rank) * (x @ self.down.T @ self.up.T)
        return self.scale * (self.alpha / self.rank) * (self.up(self.down(x)))

    def apply_to(self, w: Union[nn.Linear, nn.Conv2d, nn.Parameter, torch.Tensor]):
        if isinstance(self.up, torch.Tensor) and isinstance(self.down, torch.Tensor):
            delta_w = self.scale * (self.alpha / self.rank) * (self.up @ self.down)
        else:
            delta_w = self.scale * (self.alpha / self.rank) * (self.up.weight @ self.down.weight)
        if isinstance(w, (nn.Linear, nn.Conv2d)):
            delta_w = delta_w.to(device=w.weight.data.device, dtype=self.dtype)
            w_dtype = w.weight.data.dtype
            w.weight.data = w.weight.data.to(self.dtype)
            w.weight.data.add_(delta_w)
            w.weight.data = w.weight.data.to(w_dtype)
        elif isinstance(w, nn.Parameter):
            delta_w = delta_w.to(device=w.data.device, dtype=self.dtype)
            w_dtype = w.data.dtype
            w.data = w.data.to(self.dtype)
            w.data.add_(delta_w)
            w.data = w.data.to(w_dtype)
        elif isinstance(w, torch.Tensor):
            delta_w = delta_w.to(device=w.device, dtype=self.dtype)
            w_dtype = w.dtype
            w = w.to(self.dtype)
            w.add_(delta_w)
            w = w.to(w_dtype)


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # LoRA
        self._lora_dict = OrderedDict()
        # Frozen LoRA
        self.patched_frozen_lora = False
        self._original_weight = None

    @staticmethod
    def from_linear(linear: nn.Linear):
        lora_linear = LoRALinear(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device="meta",
            dtype=linear.weight.dtype,
        ).to_empty(device=linear.weight.device)
        lora_linear.weight = linear.weight
        lora_linear.bias = linear.bias
        return lora_linear

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
        up_linear = nn.Linear(
            up.shape[1],
            up.shape[0],
            bias=False,
            device="meta",
            dtype=dtype,
        ).to_empty(device=device)
        down_linear = nn.Linear(
            down.shape[0],
            down.shape[1],
            bias=False,
            device="meta",
            dtype=dtype,
        ).to_empty(device=device)
        up_linear.weight.data = up
        down_linear.weight.data = down
        lora = LoRA(scale, rank, alpha, up_linear, down_linear, device, dtype)
        self._lora_dict[name] = lora

    def modify_scale(self, name: str, scale: float):
        if name not in self._lora_dict:
            raise ValueError(f"LoRA name {name} not found in LoRALinear {self.__class__.__name__}")
        self._lora_dict[name].scale = scale

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
        save_original_weight: bool = True,
        **kwargs,
    ):
        if save_original_weight and self._original_weight is None:
            if self.weight.dtype == torch.float8_e4m3fn:
                self._original_weight = self.weight.to(dtype=torch.bfloat16, device="cpu", copy=True).pin_memory()
            else:
                self._original_weight = self.weight.to(device="cpu", copy=True).pin_memory()
        lora = LoRA(scale, rank, alpha, up, down, device, dtype)
        lora.apply_to(self)
        self.patched_frozen_lora = True

    def clear(self, release_all_cpu_memory: bool = False):
        if self.patched_frozen_lora and self._original_weight is None:
            raise RuntimeError(
                "Current LoRALinear has patched by frozen LoRA, but original weight is not saved, so you cannot clear LoRA."
            )
        self._lora_dict.clear()
        if self._original_weight is not None:
            self.weight.data.copy_(
                self._original_weight.to(device=self.weight.data.device, dtype=self.weight.data.dtype)
            )
            if release_all_cpu_memory:
                del self._original_weight
            self.patched_frozen_lora = False

    def forward(self, x):
        w_x = super().forward(x)
        for name, lora in self._lora_dict.items():
            w_x += lora(x)
        return w_x


class LoRAConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        # LoRA
        self._lora_dict = OrderedDict()
        # Frozen LoRA
        self._original_weight = None
        self.patched_frozen_lora = False

    @staticmethod
    def from_conv2d(conv2d: nn.Conv2d):
        lora_conv2d = LoRAConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            conv2d.bias is not None,
            conv2d.padding_mode,
            device="meta",
            dtype=conv2d.weight.dtype,
        ).to_empty(device=conv2d.weight.device)
        lora_conv2d.weight = conv2d.weight
        lora_conv2d.bias = conv2d.bias
        return lora_conv2d

    def _construct_lora(
        self,
        name: str,
        scale: float,
        rank: int,
        alpha: int,
        up: torch.Tensor,
        down: torch.Tensor,
        device: str,
        dtype: torch.dtype,
    ):
        down_conv = nn.Conv2d(
            self.in_channels,
            rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            device="meta",
            dtype=dtype,
        ).to_empty(device=device)
        down_conv.weight.data = down
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        # refer from diffusers
        up_conv = nn.Conv2d(
            rank,
            self.out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
            device="meta",
            dtype=dtype,
        ).to_empty(device=device)
        up_conv.weight.data = up

        lora = LoRA(scale, rank, alpha, up_conv, down_conv, device, dtype)
        return lora

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
        self._lora_dict[name] = self._construct_lora(name, scale, rank, alpha, up, down, device, dtype)

    def modify_scale(self, name: str, scale: float):
        if name not in self._lora_dict:
            raise ValueError(f"LoRA name {name} not found in LoRAConv2d {self.__class__.__name__}")
        self._lora_dict[name].scale = scale

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
        save_original_weight: bool = True,
    ):
        if save_original_weight and self._original_weight is None:
            if self.weight.dtype == torch.float8_e4m3fn:
                self._original_weight = self.weight.to(dtype=torch.bfloat16, device="cpu", copy=True).pin_memory()
            else:
                self._original_weight = self.weight.to(device="cpu", copy=True).pin_memory()
        lora = self._construct_lora(name, scale, rank, alpha, up, down, device, dtype)
        lora.apply_to(self)
        self.patched_frozen_lora = True

    def clear(self, release_all_cpu_memory: bool = False):
        if self.patched_frozen_lora and self._original_weight is None:
            raise RuntimeError(
                "Current LoRALinear has patched by frozen LoRA, but original weight is not saved, so you cannot clear LoRA."
            )
        self._lora_dict.clear()
        if self._original_weight is not None:
            self.weight.copy_(self._original_weight.to(device=self.weight.device, dtype=self.weight.dtype))
            if release_all_cpu_memory:
                del self._original_weight
            self.patched_frozen_lora = False

    def forward(self, x):
        w_x = super().forward(x)
        for name, lora in self._lora_dict.items():
            w_x += lora(x)
        return w_x


@contextmanager
def LoRAContext():
    origin_linear = torch.nn.Linear
    origin_conv2d = torch.nn.Conv2d

    torch.nn.Linear = LoRALinear
    torch.nn.Conv2d = LoRAConv2d
    yield
    torch.nn.Linear = origin_linear
    torch.nn.Conv2d = origin_conv2d
