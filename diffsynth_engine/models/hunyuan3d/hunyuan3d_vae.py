from typing import Optional, Dict

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

from diffsynth_engine.models.basic.attention import attention
from diffsynth_engine.models.hunyuan3d.volume_decoder import VanillaVolumeDecoder
from diffsynth_engine.models.hunyuan3d.surface_extractor import MCSurfaceExtractor
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs: int = 8, input_dim: int = 3) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.frequencies = frequencies
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        out_dim = input_dim * (self.num_freqs * 2 + 1)
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.frequencies = self.frequencies.to(x.device)
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


class MLP(nn.Module):
    def __init__(self, *, width: int, expand_ratio: int = 4, output_width: int = None):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        out = attention(q, k, v)
        return rearrange(out, "b s n d -> b s (n d)")


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads, n_ctx=n_ctx, width=width, norm_layer=norm_layer, qk_norm=qk_norm
        )

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, n_data: Optional[int] = None, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        out = attention(q, k, v)
        return rearrange(out, "b s n d -> b s (n d)")


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, width=width, norm_layer=norm_layer, qk_norm=qk_norm
        )
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        return self.c_proj(x)


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        out_channels: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=8)
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents, width=width, mlp_expand_ratio=4, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        x = self.cross_attn_decoder(query_embeddings, latents)
        x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class ShapeVAEDecoderStateDictConverter(StateDictConverter):
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                continue
            if key.startswith("pre_kl."):
                continue
            new_state_dict[key] = value
        return new_state_dict


class ShapeVAEDecoder(PreTrainedModel):
    converter = ShapeVAEDecoderStateDictConverter()

    def __init__(
        self,
        num_latents: int = 4096,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.volume_decoder = VanillaVolumeDecoder()
        self.surface_extractor = MCSurfaceExtractor()

        self.geo_decoder = CrossAttentionDecoder(
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        self.scale_factor = 1.0039506158752403

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs[0]

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents
