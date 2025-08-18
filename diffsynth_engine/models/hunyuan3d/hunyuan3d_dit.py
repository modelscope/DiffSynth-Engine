import math
import torch
import torch.nn as nn
from einops import rearrange
from diffsynth_engine.models.base import PreTrainedModel
from diffsynth_engine.models.basic.attention import attention
from .moe import MoEBlock


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, downscale_freq_shift: float = 0.0, scale: int = 1, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t):
        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)
        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        k_dim=None,
        qkv_bias=False,
        qk_norm=True,
    ):
        super().__init__()
        if k_dim is None:
            k_dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(k_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(k_dim, dim, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, y=None):
        self_attn = False
        if y is None:
            self_attn = True
            y = x
        b, s1, _ = x.shape
        _, s2, _ = y.shape
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)
        # 下面这段reshape完全是错误的，但是不能改，因为hunyuan3d 2.1这个模型训练时就是这样写的
        if self_attn:
            qkv = torch.cat((q, k, v), dim=-1)
            split_size = qkv.shape[-1] // self.num_heads // 3
            qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
            q, k, v = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((k, v), dim=-1)
            split_size = kv.shape[-1] // self.num_heads // 2
            kv = kv.view(1, -1, self.num_heads, split_size * 2)
            k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.view(b, s2, self.num_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        out = attention(q, k, v)
        out = rearrange(out, "b s n d -> b s (n d)")
        return self.out_proj(out)


class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        skip_connection: bool = False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn2 = Attention(hidden_size, num_heads=num_heads, k_dim=context_dim)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        if use_moe:
            self.moe = MoEBlock(hidden_size, num_experts, moe_top_k)
        else:
            self.mlp = MLP(width=hidden_size)
        if self.skip_connection:
            self.skip_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, cond=None, residual=None):
        if self.skip_connection:
            x = self.skip_norm(self.skip_linear(torch.cat([residual, x], dim=-1)))
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), cond)
        if self.use_moe:
            x = x + self.moe(self.norm3(x))
        else:
            x = x + self.mlp(self.norm3(x))
        return x


class HunYuan3DDiT(PreTrainedModel):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    context_dim=context_dim,
                    skip_connection=layer > depth // 2,
                    use_moe=True if depth - layer <= num_moe_layers else False,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                )
                for layer in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, out_channels)

    def forward(self, x, t, cond):
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        x = torch.cat([t, x], dim=1)
        residuals = []
        for idx, block in enumerate(self.blocks):
            residual = None if idx <= self.depth // 2 else residuals.pop()
            x = block(x, cond, residual)
            if idx < self.depth // 2:
                residuals.append(x)
        x = self.final_layer(x)
        return x
