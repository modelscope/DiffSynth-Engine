from typing import Callable, List, Tuple, Dict, Any
import math
import json
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import MelScale

from diffsynth_engine.models.base import PreTrainedModel
from diffsynth_engine.utils.constants import ACE_VAE_VOCODER_CONFIG_FILE
# TODO: this is still shitty. need modification

class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.register_buffer("window", torch.hann_window(win_length, device=device, dtype=dtype))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)

        y = F.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        dtype = y.dtype
        spec = torch.stft(
            y.float(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = spec.to(dtype)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center, device=device, dtype=dtype)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            "slaney",
            "slaney",
        )

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def forward(self, x: torch.Tensor, return_linear: bool = False) -> torch.Tensor:
        linear = self.spectrogram(x)
        x = self.mel_scale(linear)
        x = self.compress(x)
        # print(x.shape)
        if return_linear:
            return x, self.compress(linear)

        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
            device=device,
            dtype=dtype,
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim), device=device, dtype=dtype)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim, device=device, dtype=dtype)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim), device=device, dtype=dtype))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)

        if apply_residual:
            x = input + x

        return x


class ParallelConvNeXtBlock(nn.Module):
    def __init__(self, kernel_sizes: List[int], *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(kernel_size=kernel_size, *args, **kwargs) for kernel_size in kernel_sizes]
        )

    def forward(self, x: torch.torch.Tensor) -> torch.torch.Tensor:
        return torch.stack(
            [block(x, apply_residual=False) for block in self.blocks] + [x],
            dim=1,
        ).sum(dim=1)


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        layer_scale_init_value=1e-6,
        kernel_sizes: Tuple[int] = (7,),
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.channel_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                dims[0],
                kernel_size=7,
                padding=3,
                padding_mode="replicate",
                device=device,
                dtype=dtype,
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first", device=device, dtype=dtype),
        )
        self.channel_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first", device=device, dtype=dtype),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1, device=device, dtype=dtype),
            )
            self.channel_layers.append(mid_layer)

        block_fn = (
            partial(ConvNeXtBlock, kernel_size=kernel_sizes[0], device=device, dtype=dtype)
            if len(kernel_sizes) == 1
            else partial(ParallelConvNeXtBlock, kernel_sizes=kernel_sizes, device=device, dtype=dtype)
        )

        self.stages = nn.ModuleList()

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    block_fn(
                        dim=dims[i],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for _ in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first", device=device, dtype=dtype)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.torch.Tensor,
    ) -> torch.torch.Tensor:
        for channel_layer, stage in zip(self.channel_layers, self.stages):
            x = channel_layer(x)
            x = stage(x)

        return self.norm(x)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                        device=device,
                        dtype=dtype,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                        device=device,
                        dtype=dtype,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                        device=device,
                        dtype=dtype,
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        device=device,
                        dtype=dtype,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        device=device,
                        dtype=dtype,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        device=device,
                        dtype=dtype,
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: Tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: Tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: Tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        use_template: bool = True,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        post_activation: Callable = partial(nn.SiLU, inplace=True),
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        assert math.prod(upsample_rates) == hop_length, f"hop_length must be {math.prod(upsample_rates)}"

        self.conv_pre = weight_norm(
            nn.Conv1d(
                num_mels,
                upsample_initial_channel,
                pre_conv_kernel_size,
                1,
                padding=get_padding(pre_conv_kernel_size),
                device=device,
                dtype=dtype,
            )
        )

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.use_template = use_template
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                        device=device,
                        dtype=dtype,
                    )
                )
            )

            if not use_template:
                continue

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1, device=device, dtype=dtype))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d, device=device, dtype=dtype))

        self.activation_post = post_activation()
        self.conv_post = weight_norm(
            nn.Conv1d(
                ch,
                1,
                post_conv_kernel_size,
                1,
                padding=get_padding(post_conv_kernel_size),
                device=device,
                dtype=dtype,
            )
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, template=None):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            if self.use_template:
                x = x + self.noise_convs[i](template)

            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ADaMoSHiFiGANV1(PreTrainedModel):
    def __init__(
        self,
        input_channels: int = 128,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [128, 256, 384, 512],
        kernel_sizes: Tuple[int] = (7,),
        upsample_rates: Tuple[int] = (4, 4, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: Tuple[int] = (8, 8, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: Tuple[int] = (3, 7, 11, 13),
        resblock_dilation_sizes: Tuple[Tuple[int]] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        num_mels: int = 512,
        upsample_initial_channel: int = 1024,
        use_template: bool = False,
        pre_conv_kernel_size: int = 13,
        post_conv_kernel_size: int = 13,
        sampling_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: int = 40,
        f_max: int = 16000,
        n_mels: int = 128,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.backbone = ConvNeXtEncoder(
            input_channels=input_channels,
            depths=depths,
            dims=dims,
            kernel_sizes=kernel_sizes,
            device=device,
            dtype=dtype,
        )

        self.head = HiFiGANGenerator(
            hop_length=hop_length,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            num_mels=num_mels,
            upsample_initial_channel=upsample_initial_channel,
            use_template=use_template,
            pre_conv_kernel_size=pre_conv_kernel_size,
            post_conv_kernel_size=post_conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.mel_transform = LogMelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def decode(self, mel):
        y = self.backbone(mel)
        y = self.head(y)
        return y

    @torch.no_grad()
    def encode(self, x):
        return self.mel_transform(x)

    def forward(self, mel):
        y = self.backbone(mel)
        y = self.head(y)
        return y

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ) -> "ADaMoSHiFiGANV1":
        model = cls(**config, device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    @staticmethod
    def get_model_config() -> dict:
        config_file = ACE_VAE_VOCODER_CONFIG_FILE
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
