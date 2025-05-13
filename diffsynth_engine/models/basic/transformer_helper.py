import torch
import torch.nn as nn
import math


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, single=False, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.single = single
        self.linear = nn.Linear(dim, dim * (2 if single else 6), device=device, dtype=dtype)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)

    def forward(self, x, emb):
        emb = self.linear(nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale) + shift
            return x
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
            x = self.norm(x) * (1 + scale_msa) + shift_msa
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    def __init__(self, dim, device: str, dtype: torch.dtype):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 3 * dim, bias=True, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class RoPEEmbedding(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0, "The dimension must be even."

        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)

        batch_size, seq_length = pos.shape
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)

        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
        return out.float()

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([self.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-5,
        elementwise_affine=True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        norm_result = self.norm(x.float()).to(x.dtype)
        if self.elementwise_affine:
            return norm_result * self.weight
        return norm_result


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
