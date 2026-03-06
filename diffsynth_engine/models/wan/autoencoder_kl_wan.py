# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_wan.py

# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from diffsynth_engine.models.base import DiffusionModel
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

CACHE_T = 2


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        batch_size, channels, time, height, width = x.shape
        x = x.view(
            batch_size,
            channels,
            time // self.factor_t,
            self.factor_t,
            height // self.factor_s,
            self.factor_s,
            width // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            batch_size,
            channels * self.factor,
            time // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        x = x.view(
            batch_size,
            self.out_channels,
            self.group_size,
            time // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class WanCausalConv3d(nn.Conv3d):
    """
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class WanRMS_norm(nn.Module):
    """
    A custom RMS normalization layer.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class WanUpsample(nn.Upsample):
    """Perform upsampling while ensuring the output tensor has the same data type as the input."""

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module):
    """
    A custom resampling module for 2D and 3D data.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        batch_size, channels, time, height, width = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(batch_size, 2, channels, time, height, width)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(batch_size, channels, time * 2, height, width)
        time = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.resample(x)
        x = x.view(batch_size, time, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    """
    A custom residual block module.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class WanAttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        query, key, value = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(query, key, value)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        x = self.proj(x)

        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x


class WanResidualDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False):
        super().__init__()

        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    """
    A 3D encoder module for WanVAE.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling with residual shortcut for the WanVAE decoder (Wan 2.2).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder (Wan 2.1).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=None):
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class WanDecoder3d(nn.Module):
    """
    A 3D decoder module for WanVAE.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and not is_residual:
                in_dim = in_dim // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


def patchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")

    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})")

    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)
    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")

    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)

    x = x.view(batch_size, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(batch_size, channels, frames, height * patch_size, width * patch_size)
    return x


class AutoencoderKLWan(DiffusionModel):
    """
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in Wan 2.1.

    Adapted from diffusers AutoencoderKLWan with the following changes:
    - Inherits DiffusionModel instead of ModelMixin + AutoencoderMixin + ConfigMixin + FromOriginalModelMixin
    - Removes @apply_forward_hook decorators (accelerate integration hooks)
    - Removes _supports_gradient_checkpointing, _group_offload_block_modules, _skip_keys class attributes
    - Adds mask-based spatial tiled encode/decode from DiffSynth-Engine-main
    - Keeps all init params for backward compatibility via ConfigMixin

    Deleted methods/attributes from original Mixins:
    - FromOriginalModelMixin: provided from_single_file() for loading original checkpoints
    - AutoencoderMixin: provided enable_tiling/slicing convenience methods (re-implemented directly)
    - ModelMixin: provided from_pretrained/save_pretrained (DiffusionModel already has from_pretrained)
    - @apply_forward_hook: accelerate forward hooks for automatic device movement
    - _supports_gradient_checkpointing: diffusers gradient checkpointing flag
    - _group_offload_block_modules: diffusers module group offload config
    - _skip_keys: AlignDeviceHook skip keys for mutable state
    """

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: Optional[int] = None,
        z_dim: int = 16,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: Optional[int] = None,
        scale_factor_temporal: Optional[int] = 4,
        scale_factor_spatial: Optional[int] = 8,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.encoder = WanEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
        )

        self.spatial_compression_ratio = scale_factor_spatial

        self.use_slicing = False
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[int] = None,
        tile_sample_stride_width: Optional[int] = None,
    ) -> None:
        """
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def disable_tiling(self) -> None:
        """Disable tiled VAE decoding."""
        self.use_tiling = False

    def enable_slicing(self) -> None:
        """Enable sliced VAE decoding (process one batch element at a time)."""
        self.use_slicing = True

    def disable_slicing(self) -> None:
        """Disable sliced VAE decoding."""
        self.use_slicing = False

    def clear_cache(self):
        """Clear the feature cache for causal convolutions."""
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            The latent representations of the encoded videos.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        self.clear_cache()
        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True
                )
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)

        out = torch.clamp(out, min=-1.0, max=1.0)

        self.clear_cache()
        if not return_dict:
            return (out,)

        return DecoderOutput(sample=out)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images using a tiled encoder (diffusers blend-based approach)."""
        _, _, num_frames, height, width = x.shape
        encode_spatial_compression_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            assert encode_spatial_compression_ratio % self.config.patch_size == 0
            encode_spatial_compression_ratio = self.spatial_compression_ratio // self.config.patch_size

        latent_height = height // encode_spatial_compression_ratio
        latent_width = width // encode_spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // encode_spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // encode_spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // encode_spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // encode_spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1) : 1 + 4 * k,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    tile = self.quant_conv(tile)
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """Decode a batch of images using a tiled decoder (diffusers blend-based approach)."""
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            blend_width = self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(
                        tile, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=(k == 0)
                    )
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_sample_stride_height, :tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if self.config.patch_size is not None:
            dec = unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    # ---- Mask-based spatial tiled encode/decode (from DiffSynth-Engine-main) ----

    def _build_1d_mask(self, length: int, left_bound: bool, right_bound: bool, border_width: int) -> torch.Tensor:
        """Build a 1D linear ramp mask for tile blending."""
        mask = torch.ones((length,))
        if not left_bound:
            mask[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            mask[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return mask

    def _build_spatial_mask(
        self,
        data: torch.Tensor,
        is_bound: Tuple[bool, bool, bool, bool],
        border_width: Tuple[int, int],
    ) -> torch.Tensor:
        """Build a 2D spatial mask for tile blending using linear ramps at borders."""
        _, _, _, spatial_height, spatial_width = data.shape
        height_mask = self._build_1d_mask(spatial_height, is_bound[0], is_bound[1], border_width[0])
        width_mask = self._build_1d_mask(spatial_width, is_bound[2], is_bound[3], border_width[1])

        height_mask = height_mask.unsqueeze(1).expand(spatial_height, spatial_width)
        width_mask = width_mask.unsqueeze(0).expand(spatial_height, spatial_width)

        mask = torch.stack([height_mask, width_mask]).min(dim=0).values
        mask = mask.reshape(1, 1, 1, spatial_height, spatial_width)
        return mask

    def tiled_encode_with_mask(
        self,
        x: torch.Tensor,
        tile_size: Tuple[int, int] = (256, 256),
        tile_stride: Tuple[int, int] = (192, 192),
    ) -> torch.Tensor:
        """
        Encode using mask-weighted spatial tiling (from DiffSynth-Engine-main).

        This approach uses smooth gradient masks at tile boundaries for blending,
        which can produce better results than the simple blend approach.

        Args:
            x: Input tensor [B, C, T, H, W].
            tile_size: (height, width) of each tile in pixel space.
            tile_stride: (height, width) stride between tiles in pixel space.

        Returns:
            Encoded latent tensor.
        """
        _, _, num_frames, height, width = x.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
            _, _, _, height, width = x.shape

        encode_spatial_compression_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            encode_spatial_compression_ratio = self.spatial_compression_ratio // self.config.patch_size
            size_h = size_h // self.config.patch_size
            size_w = size_w // self.config.patch_size
            stride_h = stride_h // self.config.patch_size
            stride_w = stride_w // self.config.patch_size

        # Build tile tasks
        tasks = []
        for h in range(0, height, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= height:
                continue
            for w in range(0, width, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= width:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))

        out_t = 1 + (num_frames - 1) // 4
        latent_h = height // encode_spatial_compression_ratio
        latent_w = width // encode_spatial_compression_ratio

        weight = torch.zeros((1, 1, out_t, latent_h, latent_w), dtype=x.dtype, device="cpu")
        values = torch.zeros((1, self.z_dim, out_t, latent_h, latent_w), dtype=x.dtype, device="cpu")

        for h, h_end, w, w_end in tasks:
            tile = x[:, :, :, h:h_end, w:w_end]

            self.clear_cache()
            iter_ = 1 + (num_frames - 1) // 4
            for k in range(iter_):
                self._enc_conv_idx = [0]
                if k == 0:
                    enc_out = self.encoder(
                        tile[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                    )
                else:
                    enc_chunk = self.encoder(
                        tile[:, :, 1 + 4 * (k - 1) : 1 + 4 * k, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                    enc_out = torch.cat([enc_out, enc_chunk], 2)
            enc_out = self.quant_conv(enc_out)

            # Build posterior and sample
            enc_out_cpu = enc_out.to("cpu")

            mask = self._build_spatial_mask(
                enc_out_cpu,
                is_bound=(h == 0, h_end >= height, w == 0, w_end >= width),
                border_width=(
                    (size_h - stride_h) // encode_spatial_compression_ratio,
                    (size_w - stride_w) // encode_spatial_compression_ratio,
                ),
            ).to(dtype=x.dtype, device="cpu")

            target_h = h // encode_spatial_compression_ratio
            target_w = w // encode_spatial_compression_ratio
            values[
                :,
                :,
                :,
                target_h : target_h + enc_out_cpu.shape[3],
                target_w : target_w + enc_out_cpu.shape[4],
            ] += enc_out_cpu * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + enc_out_cpu.shape[3],
                target_w : target_w + enc_out_cpu.shape[4],
            ] += mask

        self.clear_cache()
        result = values / weight
        return result.to(x.device)

    def tiled_decode_with_mask(
        self,
        z: torch.Tensor,
        tile_size: Tuple[int, int] = (32, 32),
        tile_stride: Tuple[int, int] = (24, 24),
    ) -> torch.Tensor:
        """
        Decode using mask-weighted spatial tiling (from DiffSynth-Engine-main).

        This approach uses smooth gradient masks at tile boundaries for blending,
        which can produce better results than the simple blend approach.

        Args:
            z: Input latent tensor [B, C, T, H, W].
            tile_size: (height, width) of each tile in latent space.
            tile_stride: (height, width) stride between tiles in latent space.

        Returns:
            Decoded video tensor, clamped to [-1, 1].
        """
        _, _, num_frames, height, width = z.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        upsampling_factor = self.spatial_compression_ratio

        # Build tile tasks
        tasks = []
        for h in range(0, height, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= height:
                continue
            for w in range(0, width, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= width:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))

        out_t = num_frames * 4 - 3
        out_channels = self.config.out_channels
        out_h = height * upsampling_factor
        out_w = width * upsampling_factor
        if self.config.patch_size is not None:
            out_h = out_h // self.config.patch_size
            out_w = out_w // self.config.patch_size

        weight = torch.zeros((1, 1, out_t, out_h, out_w), dtype=z.dtype, device="cpu")
        values = torch.zeros((1, out_channels, out_t, out_h, out_w), dtype=z.dtype, device="cpu")

        for h, h_end, w, w_end in tasks:
            tile_z = z[:, :, :, h:h_end, w:w_end]

            self.clear_cache()
            tile_x = self.post_quant_conv(tile_z)
            for k in range(num_frames):
                self._conv_idx = [0]
                if k == 0:
                    dec_out = self.decoder(
                        tile_x[:, :, k : k + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                else:
                    dec_chunk = self.decoder(
                        tile_x[:, :, k : k + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                    )
                    dec_out = torch.cat([dec_out, dec_chunk], 2)

            if self.config.patch_size is not None:
                dec_out = unpatchify(dec_out, patch_size=self.config.patch_size)

            dec_out_cpu = dec_out.to("cpu")

            effective_upsampling = upsampling_factor
            if self.config.patch_size is not None:
                effective_upsampling = upsampling_factor // self.config.patch_size

            mask = self._build_spatial_mask(
                dec_out_cpu,
                is_bound=(h == 0, h_end >= height, w == 0, w_end >= width),
                border_width=(
                    (size_h - stride_h) * effective_upsampling,
                    (size_w - stride_w) * effective_upsampling,
                ),
            ).to(dtype=z.dtype, device="cpu")

            target_h = h * effective_upsampling
            target_w = w * effective_upsampling
            values[
                :,
                :,
                :,
                target_h : target_h + dec_out_cpu.shape[3],
                target_w : target_w + dec_out_cpu.shape[4],
            ] += dec_out_cpu * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + dec_out_cpu.shape[3],
                target_w : target_w + dec_out_cpu.shape[4],
            ] += mask

        self.clear_cache()
        result = values / weight
        result = result.float().clamp(-1, 1)
        return result.to(z.device)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior or use the mode.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                Random number generator for sampling.
        """
        posterior = self.encode(sample).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec
