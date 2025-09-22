import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm
from typing import Any, Dict

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.utils.constants import QWEN_IMAGE_VAE_KEYMAP_FILE

CACHE_T = 2

with open(QWEN_IMAGE_VAE_KEYMAP_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


def block_causal_mask(x, block_size):
    # params
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size

    # build mask
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :, i * block_size : (i + 1) * block_size, : (i + 1) * block_size] = 1
    return mask


class QwenCausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class QwenRMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode, keep_channels=False):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim if keep_channels else dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim if keep_channels else dim // 2, 3, padding=1),
            )
            self.time_conv = QwenCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                key = id(self.resample)
                if key not in feat_cache:
                    feat_cache[key] = "Rep"
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[key] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat([feat_cache[key][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[key] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[key] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[key])
                    feat_cache[key] = cache_x

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                key = id(self.time_conv)
                if key not in feat_cache:
                    feat_cache[key] = x.clone()
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[key][:, :, -1:, :, :], x], 2))
                    feat_cache[key] = cache_x
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            QwenRMS_norm(in_dim, images=False),
            nn.SiLU(),
            QwenCausalConv3d(in_dim, out_dim, 3, padding=1),
            QwenRMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            QwenCausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = QwenCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None):
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, QwenCausalConv3d) and feat_cache is not None:
                key = id(layer)
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and key in feat_cache:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[key][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[key] if key in feat_cache else None)
                feat_cache[key] = cache_x
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = QwenRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            # attn_mask=block_causal_mask(q, block_size=h * w)
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


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
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
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


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode, keep_channels=True))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]

        # init block
        self.conv1 = QwenCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        use_down_residual_block = in_channels == 12
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if use_down_residual_block:
                t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
                downsamples.append(
                    Down_ResidualBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        dropout=dropout,
                        mult=num_res_blocks,
                        temperal_downsample=t_down_flag,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                # residual (+attention) blocks
                for _ in range(num_res_blocks):
                    downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
        )

        # output blocks
        self.head = nn.Sequential(
            QwenRMS_norm(out_dim, images=False), nn.SiLU(), QwenCausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None):
        if feat_cache is not None:
            key = id(self.conv1)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and key in feat_cache:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[key][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[key] if key in feat_cache else None)
            feat_cache[key] = cache_x
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, QwenCausalConv3d) and feat_cache is not None:
                key = id(layer)
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and key in feat_cache:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[key][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[key] if key in feat_cache else None)
                feat_cache[key] = cache_x
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        out_channels=3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_upsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv1 = QwenCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        use_up_residual_block = out_channels == 12
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if use_up_residual_block:
                t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
                upsamples.append(
                    Up_ResidualBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        dropout=dropout,
                        mult=num_res_blocks + 1,
                        temperal_upsample=t_up_flag,
                        up_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                # residual (+attention) blocks
                if i == 1 or i == 2 or i == 3:
                    in_dim = in_dim // 2
                for _ in range(num_res_blocks + 1):
                    upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                    in_dim = out_dim

                # upsample block
                if i != len(dim_mult) - 1:
                    mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                    upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            QwenRMS_norm(out_dim, images=False), nn.SiLU(), QwenCausalConv3d(out_dim, out_channels, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, first_chunk=False):
        ## conv1
        if feat_cache is not None:
            key = id(self.conv1)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and key in feat_cache:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[key][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[key] if key in feat_cache else None)
            feat_cache[key] = cache_x
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if check_is_instance(layer, Up_ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, first_chunk)
            elif feat_cache is not None:
                x = layer(x, feat_cache)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, QwenCausalConv3d) and feat_cache is not None:
                key = id(layer)
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and key in feat_cache:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[key][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[key] if key in feat_cache else None)
                feat_cache[key] = cache_x
            else:
                x = layer(x)
        return x


class VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_dim=96,
        decoder_dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            in_channels, encoder_dim, z_dim * 2, dim_mult, num_res_blocks, self.temperal_downsample, dropout
        )
        self.conv1 = QwenCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = QwenCausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            out_channels, decoder_dim, z_dim, dim_mult, num_res_blocks, self.temperal_upsample, dropout
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        feat_cache = {}
        x = patchify(x, patch_size=2 if self.in_channels == 12 else 1)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_cache)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=feat_cache,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        feat_cache = {}
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache, first_chunk=True)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache)
                out = torch.cat([out, out_], 2)  # may add tensor offload
        out = unpatchify(out, patch_size=2 if self.out_channels == 12 else 1)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)


class QwenImageVAEStateDictConverter(StateDictConverter):
    def from_diffusers(self, state_dict):
        rename_dict = config["diffusers"]["rename_dict"]
        state_dict_ = {}
        for name, param in state_dict.items():
            name_ = f"model.{name}"
            if name_ in rename_dict:
                name_ = rename_dict[name_]
            state_dict_[name_] = param
        return state_dict_

    def from_civitai(self, state_dict):
        state_dict_ = {}
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        for name in state_dict:
            state_dict_["model." + name] = state_dict[name]
        return state_dict_

    def convert(self, state_dict):
        if "post_quant_conv.bias" in state_dict:
            return self.from_diffusers(state_dict)
        else:
            return self.from_civitai(state_dict)


class QwenImageVAE(PreTrainedModel):
    converter = QwenImageVAEStateDictConverter()

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_dim=96,
        decoder_dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[False, True, True],
        dropout=0.0,
        patch_size=1,
        mean=None,
        std=None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        mean = mean or [
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
        ]
        std = std or [
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
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = (
            VideoVAE(
                in_channels=in_channels,
                out_channels=out_channels,
                encoder_dim=encoder_dim,
                decoder_dim=decoder_dim,
                z_dim=z_dim,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                temperal_downsample=temperal_downsample,
                dropout=dropout,
            )
            .eval()
            .requires_grad_(False)
        )
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.upsampling_factor = 8 * patch_size

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ) -> "QwenImageVAE":
        model = cls(**config, device="meta", dtype=dtype)
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride, progress_callback=None):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = device if dist.is_initialized() else "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )

        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, (h, h_, w, w_) in enumerate(tqdm(tasks, desc="VAE DECODING", disable=hide_progress)):
            if dist.is_initialized() and (i % dist.get_world_size() != dist.get_rank()):
                continue
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
            if progress_callback is not None and not hide_progress:
                progress_callback(i + 1, len(tasks), "VAE DECODING")
        if progress_callback is not None and not hide_progress:
            progress_callback(len(tasks), len(tasks), "VAE DECODING")
        if dist.is_initialized():
            dist.all_reduce(values)
            dist.all_reduce(weight)
        values = values / weight
        values = values.float().clamp_(-1, 1)
        return values

    def tiled_encode(self, video, device, tile_size, tile_stride, progress_callback=None):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = device if dist.is_initialized() else "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )

        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, (h, h_, w, w_) in enumerate(tqdm(tasks, desc="VAE ENCODING", disable=hide_progress)):
            if dist.is_initialized() and (i % dist.get_world_size() != dist.get_rank()):
                continue
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
            if progress_callback is not None and not hide_progress:
                progress_callback(i + 1, len(tasks), "VAE ENCODING")
        if progress_callback is not None and not hide_progress:
            progress_callback(len(tasks), len(tasks), "VAE ENCODING")
        if dist.is_initialized():
            dist.all_reduce(values)
            dist.all_reduce(weight)
        values = values / weight
        values = values.float()
        return values

    def single_encode(self, video, device, progress_callback=None):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        if progress_callback is not None:
            progress_callback(1, 1, "VAE ENCODING")
        return x.float()

    def single_decode(self, hidden_state, device, progress_callback=None):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        if progress_callback is not None:
            progress_callback(1, 1, "VAE DECODING")
        return video.float().clamp_(-1, 1)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), progress_callback=None):
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(
                    video, device, tile_size, tile_stride, progress_callback=progress_callback
                )
            else:
                hidden_state = self.single_encode(video, device, progress_callback=progress_callback)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(
        self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), progress_callback=None
    ):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for i, hidden_state in enumerate(hidden_states):
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                assert tile_size[0] % self.patch_size == 0 and tile_size[1] % self.patch_size == 0
                assert tile_stride[0] % self.patch_size == 0 and tile_stride[1] % self.patch_size == 0
                tile_size = (tile_size[0] // self.patch_size, tile_size[1] // self.patch_size)
                tile_stride = (tile_stride[0] // self.patch_size, tile_stride[1] // self.patch_size)
                video = self.tiled_decode(
                    hidden_state, device, tile_size, tile_stride, progress_callback=progress_callback
                )
            else:
                video = self.single_decode(hidden_state, device, progress_callback=progress_callback)
            video = video.squeeze(0)
            videos.append(video)
        return videos
