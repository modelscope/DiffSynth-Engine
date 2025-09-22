import os
import json
import torch
import torch.nn as nn
from typing import Dict

from diffsynth_engine.models.basic.attention import Attention
from diffsynth_engine.models.basic.unet_helper import ResnetBlock, UpSampler, DownSampler
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.utils.constants import VAE_CONFIG_FILE
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

with open(VAE_CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


class VAEStateDictConverter(StateDictConverter):
    def __init__(self, has_encoder: bool = False, has_decoder: bool = False):
        self.has_encoder = has_encoder
        self.has_decoder = has_decoder

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = config["civitai"]["rename_dict"]
        new_state_dict = {}
        for key, param in state_dict.items():
            if key not in rename_dict:
                continue
            new_key = rename_dict[key]
            if "transformer_blocks" in new_key:
                param = param.squeeze()
            new_state_dict[new_key] = param
        return new_state_dict

    def _filter(self, state_dict: Dict[str, torch.Tensor]):
        new_state_dict = {}
        for key, param in state_dict.items():
            if self.has_encoder and self.has_decoder:
                new_state_dict[key] = param
            elif self.has_encoder and key.startswith("encoder."):
                new_state_dict[key[len("encoder.") :]] = param
            elif self.has_decoder and key.startswith("decoder."):
                new_state_dict[key[len("decoder.") :]] = param
        return new_state_dict

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.has_decoder or self.has_encoder, "Either decoder or encoder must be present"
        if (
            "first_stage_model.decoder.conv_in.weight" in state_dict
            or "first_stage_model.encoder.conv_in.weight" in state_dict
        ):
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return self._filter(state_dict)


class VAEAttentionBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        in_channels,
        num_layers=1,
        norm_num_groups=32,
        eps=1e-5,
        attn_impl: str = "auto",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True, device=device, dtype=dtype
        )

        self.transformer_blocks = nn.ModuleList(
            [
                Attention(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    bias_q=True,
                    bias_kv=True,
                    bias_out=True,
                    attn_impl=attn_impl,
                    device=device,
                    dtype=dtype,
                )
                for d in range(num_layers)
            ]
        )

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class VAEDecoder(PreTrainedModel):
    converter = VAEStateDictConverter(has_decoder=True)

    def __init__(
        self,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0,
        use_post_quant_conv: bool = True,
        attn_impl: str = "auto",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.use_post_quant_conv = use_post_quant_conv
        if use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(
                latent_channels, latent_channels, kernel_size=1, device=device, dtype=dtype
            )
        self.conv_in = nn.Conv2d(latent_channels, 512, kernel_size=3, padding=1, device=device, dtype=dtype)

        self.blocks = nn.ModuleList(
            [
                # UNetMidBlock2D
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                VAEAttentionBlock(1, 512, 512, 1, eps=1e-6, device=device, dtype=dtype, attn_impl=attn_impl),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                # UpDecoderBlock2D
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                UpSampler(512, device=device, dtype=dtype),
                # UpDecoderBlock2D
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                UpSampler(512, device=device, dtype=dtype),
                # UpDecoderBlock2D
                ResnetBlock(512, 256, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(256, 256, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(256, 256, eps=1e-6, device=device, dtype=dtype),
                UpSampler(256, device=device, dtype=dtype),
                # UpDecoderBlock2D
                ResnetBlock(256, 128, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
            ]
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6, device=device, dtype=dtype)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        original_dtype = sample.dtype
        sample = sample.to(dtype=next(iter(self.parameters())).dtype)
        if tiled:
            raise NotImplementedError()

        # 1. pre-process
        sample = sample / self.scaling_factor + self.shift_factor
        if self.use_post_quant_conv:
            sample = self.post_quant_conv(sample)
        hidden_states = self.conv_in(sample)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.to(original_dtype)

        return hidden_states

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0,
        use_post_quant_conv: bool = True,
        attn_impl: str = "auto",
    ):
        model = cls(
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            use_post_quant_conv=use_post_quant_conv,
            attn_impl=attn_impl,
            device="meta",
            dtype=dtype,
        )
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str | os.PathLike, **kwargs):
        raise NotImplementedError()


class VAEEncoder(PreTrainedModel):
    converter = VAEStateDictConverter(has_encoder=True)

    def __init__(
        self,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0,
        use_quant_conv: bool = True,
        attn_impl: str = "auto",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.use_quant_conv = use_quant_conv
        if use_quant_conv:
            self.quant_conv = nn.Conv2d(
                2 * latent_channels, 2 * latent_channels, kernel_size=1, device=device, dtype=dtype
            )
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1, device=device, dtype=dtype)

        self.blocks = nn.ModuleList(
            [
                # DownEncoderBlock2D
                ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
                DownSampler(128, padding=0, extra_padding=True, device=device, dtype=dtype),
                # DownEncoderBlock2D
                ResnetBlock(128, 256, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(256, 256, eps=1e-6, device=device, dtype=dtype),
                DownSampler(256, padding=0, extra_padding=True, device=device, dtype=dtype),
                # DownEncoderBlock2D
                ResnetBlock(256, 512, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                DownSampler(512, padding=0, extra_padding=True, device=device, dtype=dtype),
                # DownEncoderBlock2D
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                # UNetMidBlock2D
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
                VAEAttentionBlock(1, 512, 512, 1, eps=1e-6, device=device, dtype=dtype, attn_impl=attn_impl),
                ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            ]
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, device=device, dtype=dtype)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 2 * latent_channels, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        original_dtype = sample.dtype
        sample = sample.to(dtype=next(iter(self.parameters())).dtype)
        if tiled:
            raise NotImplementedError()

        # 1. pre-process
        hidden_states = self.conv_in(sample)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if self.use_quant_conv:
            hidden_states = self.quant_conv(hidden_states)
        hidden_states = hidden_states[:, : self.latent_channels]
        hidden_states = (hidden_states - self.shift_factor) * self.scaling_factor
        hidden_states = hidden_states.to(original_dtype)
        return hidden_states

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0,
        use_quant_conv: bool = True,
        attn_impl: str = "auto",
    ):
        model = cls(
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            use_quant_conv=use_quant_conv,
            attn_impl=attn_impl,
            device="meta",
            dtype=dtype,
        )
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str | os.PathLike, **kwargs):
        raise NotImplementedError()


class VAE(PreTrainedModel):
    converter = VAEStateDictConverter(has_encoder=True, has_decoder=True)

    def __init__(
        self,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        attn_impl: str = "auto",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.encoder = VAEEncoder(
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            use_quant_conv=use_quant_conv,
            attn_impl=attn_impl,
            device=device,
            dtype=dtype,
        )
        self.decoder = VAEDecoder(
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            use_post_quant_conv=use_post_quant_conv,
            attn_impl=attn_impl,
            device=device,
            dtype=dtype,
        )

    def encode(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        return self.encoder(sample, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, **kwargs)

    def decode(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        return self.decoder(sample, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, **kwargs)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        attn_impl: str = "auto",
    ):
        model = cls(
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            attn_impl=attn_impl,
            device="meta",
            dtype=dtype,
        )
        model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model
