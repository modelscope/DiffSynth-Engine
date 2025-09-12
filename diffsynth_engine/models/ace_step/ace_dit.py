from typing import Any, Dict, List, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.attention import attention
from diffsynth_engine.models.basic.transformer_helper import RMSNorm
from diffsynth_engine.models.wan.wan_dit import rope_apply, modulate
from diffsynth_engine.models.ace_step.ace_lyric_encoder import ConformerEncoder
from diffsynth_engine.utils.constants import ACE_DIT_CONFIG_FILE


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device="cuda:0"):
        super().__init__() # TODO: how to deal with meta device issue?
        device = "cuda:2"
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.int64).float() / dim))
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.int64).float()
        freqs = torch.outer(t, self.inv_freq)
        self.freqs_cis_cached = torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        return self.freqs_cis_cached[:seq_len][None, :, None, :].to(x.device)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.k = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.v = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.o = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.attn_kwargs = attn_kwargs if attn_kwargs is not None else {}

    def forward(self, x, freqs, attn_mask):  # x: (b, s, d), attn_mask: (b, s)
        q, k, v = self.q(x), self.k(x), self.v(x)
        num_heads = q.shape[2] // self.head_dim
        attn_mask = attn_mask[:, :, None, None]
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        q = F.relu(rope_apply(q, freqs) * attn_mask)
        k = F.relu(rope_apply(k, freqs) * attn_mask)
        v = v * attn_mask
        q = rearrange(q, "b s n d -> b n d s")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n d s")
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)  # b n (d+1) s
        x = torch.matmul(torch.matmul(v, k), q)  # inner: b n (d+1) d
        x = x[:, :, :-1] / (x[:, :, -1:] + 1e-15)  # b n d s
        x = rearrange(x, "b n d s -> b s (n d)")
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.k = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.v = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.o = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.attn_kwargs = attn_kwargs if attn_kwargs is not None else {}

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        freqs: torch.Tensor,
        freqs_ctx: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_mask_ctx: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.q(x), self.k(ctx), self.v(ctx)
        num_heads = q.shape[2] // self.head_dim
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        q = rope_apply(q, freqs)
        k = rope_apply(k, freqs_ctx)
        mask = attn_mask[:, :, None] * attn_mask_ctx[:, None, :]
        mask = torch.where(mask == 1, 0.0, -torch.inf)[:, None]
        mask = mask.expand(-1, num_heads, -1, -1).to(q.dtype)
        x = attention(q, k, v, attn_mask=mask, **self.attn_kwargs)
        x = rearrange(x, "b s n d -> b s (n d)")
        return self.o(x)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        groups: int = 1,
        use_bias: bool = False,
        act: str | None = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=use_bias,
            device=device,
            dtype=dtype,
        )
        self.act = nn.SiLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            kernel_size=1,
            use_bias=True,
            act="silu",
            device=device,
            dtype=dtype,
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            groups=hidden_features * 2,
            use_bias=True,
            act="silu",
            device=device,
            dtype=dtype,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            in_features,
            kernel_size=1,
            use_bias=False,
            act=None,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x *= self.glu_act(gate)
        x = self.point_conv(x)
        x = x.transpose(1, 2)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = RMSNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.attn = SelfAttention(dim, num_heads, attn_kwargs=attn_kwargs, device=device, dtype=dtype)
        self.cross_attn = CrossAttention(dim, num_heads, attn_kwargs=attn_kwargs, device=device, dtype=dtype)
        self.ff = GLUMBConv(in_features=dim, hidden_features=int(dim * mlp_ratio), device=device, dtype=dtype)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim, device=device, dtype=dtype) / dim**0.5)

    def forward(self, x, context, t_mod, freqs, freqs_ctx, attn_mask, attn_mask_ctx):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.squeeze(1) for t in (self.scale_shift_table[None] + rearrange(t_mod, "b (c d) -> b c d", c=6)).chunk(6, dim=1)
        ]
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x += gate_msa * self.attn(input_x, freqs, attn_mask)
        x += self.cross_attn(x, context, freqs, freqs_ctx, attn_mask, attn_mask_ctx)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x += gate_mlp * self.ff(input_x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=(16, 1),
        in_channels=8,
        embed_dim=1152,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.early_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 256,
                kernel_size=patch_size,
                stride=patch_size,
                device=device,
                dtype=dtype,
            ),
            nn.GroupNorm(num_groups=32, num_channels=in_channels * 256, eps=1e-6, device=device, dtype=dtype),
            nn.Conv2d(
                in_channels * 256,
                embed_dim,
                kernel_size=1,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, latent):
        # early convolutions, N x C x H x W -> N x 256 * sqrt(patch_size) x H/patch_size x W/patch_size
        latent = self.early_conv_layers(latent)
        return rearrange(latent, "b c h w -> b (h w) c")


class FinalLayer(nn.Module):
    """Similar to `Head` in Wan2.1."""

    def __init__(self, hidden_size, patch_size=[16, 1], out_channels=256):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ACEStepDiTStateDictConverter(StateDictConverter):
    def convert(self, state_dict):
        for key in list(state_dict.keys()):
            # change all linear_q / linear_k / linear_v / linear_p to q / k / v / p
            if "linear_q" in key:
                new_key = key.replace("linear_q", "q")
                state_dict[new_key] = state_dict.pop(key)
            elif "linear_k" in key:
                new_key = key.replace("linear_k", "k")
                state_dict[new_key] = state_dict.pop(key)
            elif "linear_v" in key:
                new_key = key.replace("linear_v", "v")
                state_dict[new_key] = state_dict.pop(key)
            elif "linear_p" in key:
                new_key = key.replace("linear_pos", "p")
                state_dict[new_key] = state_dict.pop(key)
            elif "linear_out" in key:
                new_key = key.replace("linear_out", "o")
                state_dict[new_key] = state_dict.pop(key)
            # change all to_q / to_k / to_v / to_out to q / k / v / o
            elif "to_q" in key:
                new_key = key.replace("to_q", "q")
                state_dict[new_key] = state_dict.pop(key)
            elif "to_k" in key:
                new_key = key.replace("to_k", "k")
                state_dict[new_key] = state_dict.pop(key)
            elif "to_v" in key:
                new_key = key.replace("to_v", "v")
                state_dict[new_key] = state_dict.pop(key)
            elif "to_out.0" in key:
                new_key = key.replace("to_out.0", "o")
                state_dict[new_key] = state_dict.pop(key)
            # remove all add_{q/k/v}_proj
            elif "add_q_proj" in key or "add_k_proj" in key or "add_v_proj" in key or "to_add_out" in key:
                state_dict.pop(key)
            # remove all projectors.
            elif "projectors" in key:
                state_dict.pop(key)
            # rename timestep_embedder.linear_1 into time_embedder.timestep_embedder.0
            elif "timestep_embedder.linear_1" in key:
                new_key = key.replace("timestep_embedder.linear_1", "time_embedder.timestep_embedder.0")
                state_dict[new_key] = state_dict.pop(key)
            # rename timestep_embedder.linear_2 into time_embedder.timestep_embedder.2
            elif "timestep_embedder.linear_2" in key:
                new_key = key.replace("timestep_embedder.linear_2", "time_embedder.timestep_embedder.2")
                state_dict[new_key] = state_dict.pop(key)
        return state_dict


class ACEStepDiT(PreTrainedModel):
    converter = ACEStepDiTStateDictConverter()
    _supports_parallelization = True

    def __init__(
        self,
        num_layers: int = 24,
        head_dim: int = 64,
        num_heads: int = 20,
        mlp_ratio: float = 2.5,
        out_channels: int = 8,
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        speaker_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        lyric_encoder_vocab_size: int = 6693,
        lyric_hidden_size: int = 1024,
        patch_size: List[int] = [16, 1],
        attn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
            device=device,
        )

        inner_dim = num_heads * head_dim
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=inner_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_kwargs=attn_kwargs,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.time_embedder = TimestepEmbeddings(dim_in=256, dim_out=inner_dim, device=device, dtype=dtype)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, device=device, dtype=dtype))

        self.speaker_embedder = nn.Linear(speaker_embedding_dim, inner_dim, device=device, dtype=dtype)
        self.genre_embedder = nn.Linear(text_embedding_dim, inner_dim, device=device, dtype=dtype)
        self.lyric_embs = nn.Embedding(lyric_encoder_vocab_size, lyric_hidden_size, device=device, dtype=dtype)
        self.lyric_encoder = ConformerEncoder(input_size=lyric_hidden_size)
        self.lyric_proj = nn.Linear(lyric_hidden_size, inner_dim, device=device, dtype=dtype)
        self.proj_in = PatchEmbed(patch_size=patch_size, embed_dim=inner_dim, device=device, dtype=dtype)
        self.final_layer = FinalLayer(inner_dim, patch_size=patch_size, out_channels=out_channels)

    def forward_lyric_encoder(self, context_lyric: torch.LongTensor, attn_mask_lyric: torch.LongTensor):
        lyric_embs = self.lyric_embs(context_lyric)  # N x T x D
        prompt_prenet_out = self.lyric_encoder(lyric_embs, attn_mask_lyric)
        prompt_prenet_out = self.lyric_proj(prompt_prenet_out)
        return prompt_prenet_out

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (x y c) -> b c (h x) (w y)",
            h=1,
            x=self.patch_size[0],
            y=self.patch_size[1],
        )

    def encode(
        self,
        context_prompt: torch.Tensor,
        context_lyric: torch.LongTensor,
        attn_mask_prompt: torch.LongTensor,
        attn_mask_lyric: torch.LongTensor,
    ):
        b = context_prompt.shape[0]
        device, dtype = context_prompt.device, context_prompt.dtype
        context_speaker = torch.zeros(
            b,
            1,
            self.speaker_embedder.in_features,
            device=device,
            dtype=dtype,
        )
        context_speaker = self.speaker_embedder(context_speaker)
        context_prompt = self.genre_embedder(context_prompt)
        context_lyric = self.forward_lyric_encoder(context_lyric=context_lyric, attn_mask_lyric=attn_mask_lyric)
        context = torch.cat([context_speaker, context_prompt, context_lyric], dim=1)

        attn_mask_speaker = torch.ones(b, 1, device=device)
        context_mask = torch.cat([attn_mask_speaker, attn_mask_prompt, attn_mask_lyric], dim=1)
        return context, context_mask

    def decode(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor],
        context: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_mask_ctx: torch.Tensor,
    ):
        t = self.time_embedder(timestep, x.dtype)
        t_mod = self.t_block(t)

        x = self.proj_in(x)

        freqs = self.rotary_emb(x)
        freqs_ctx = self.rotary_emb(context)

        for block in self.transformer_blocks:
            x = block(
                x=x,
                context=context,
                t_mod=t_mod,
                freqs=freqs,
                freqs_ctx=freqs_ctx,
                attn_mask=attn_mask,
                attn_mask_ctx=attn_mask_ctx,
            )

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context_prompt: torch.Tensor,
        context_lyric: torch.LongTensor,
        attn_mask: torch.Tensor,
        attn_mask_prompt: torch.LongTensor,
        attn_mask_lyric: torch.LongTensor,
    ):
        context, attn_mask_ctx = self.encode(
            context_prompt=context_prompt,
            context_lyric=context_lyric,
            attn_mask_prompt=attn_mask_prompt,
            attn_mask_lyric=attn_mask_lyric,
        )
        output_length = x.shape[-1]

        output = self.decode(
            x=x,
            timestep=timestep,
            context=context,
            attn_mask=attn_mask,
            attn_mask_ctx=attn_mask_ctx,
        )

        if output_length > output.shape[-1]:
            output = F.pad(output, (0, output_length - output.shape[-1], 0, 0), "constant", 0)
        elif output_length < output.shape[-1]:
            output = output[:, :, :, :output_length]
        return output
    
    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        assign: bool = True,
    ):
        model = cls(**config, device="meta", dtype=dtype, attn_kwargs=attn_kwargs)
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=assign)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    @staticmethod
    def get_model_config() -> dict:
        config_file = ACE_DIT_CONFIG_FILE
        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def compile_repeated_blocks(self, *args, **kwargs):
        for block in self.transformer_blocks:
            block.compile(*args, **kwargs)
