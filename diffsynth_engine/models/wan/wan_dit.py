import math
import json
import torch
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.models.basic.transformer_helper import RMSNorm
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils.constants import (
    WAN_DIT_1_3B_T2V_CONFIG_FILE,
    WAN_DIT_14B_I2V_CONFIG_FILE,
    WAN_DIT_14B_T2V_CONFIG_FILE,
    WAN_DIT_14B_FLF2V_CONFIG_FILE,
)
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.parallel import sequence_parallel, sequence_parallel_unshard

T5_TOKEN_NUM = 512
FLF_TOKEN_NUM = 257 * 2


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs):
    b, s, n, d = x.shape
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(b, s, n, d // 2, 2))
    x_out = torch.view_as_real(x_out * freqs)
    return x_out.to(x.dtype).flatten(3)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        attn_impl: Optional[str] = None,
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
        self.norm_q = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.norm_k = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.attn_impl = attn_impl

    def forward(self, x, freqs):
        q, k, v = self.norm_q(self.q(x)), self.norm_k(self.k(x)), self.v(x)
        num_heads = q.shape[2] // self.head_dim
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = attention_ops.attention(
            q=rope_apply(q, freqs),
            k=rope_apply(k, freqs),
            v=v,
            attn_impl=self.attn_impl,
        )
        x = x.flatten(2)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
        attn_impl: Optional[str] = None,
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
        self.norm_q = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.norm_k = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim, device=device, dtype=dtype)
            self.v_img = nn.Linear(dim, dim, device=device, dtype=dtype)
            self.norm_k_img = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.attn_impl = attn_impl

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :-T5_TOKEN_NUM]
            ctx = y[:, -T5_TOKEN_NUM:]
        else:
            ctx = y
        q, k, v = self.norm_q(self.q(x)), self.norm_k(self.k(ctx)), self.v(ctx)
        num_heads = q.shape[2] // self.head_dim
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)

        x = attention_ops.attention(q, k, v, attn_impl=self.attn_impl).flatten(2)
        if self.has_image_input:
            k_img, v_img = self.norm_k_img(self.k_img(img)), self.v_img(img)
            k_img = rearrange(k_img, "b s (n d) -> b s n d", n=num_heads)
            v_img = rearrange(v_img, "b s (n d) -> b s n d", n=num_heads)
            y = attention_ops.attention(q, k_img, v_img, attn_impl=self.attn_impl).flatten(2)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(dim, num_heads, eps, attn_impl=attn_impl, device=device, dtype=dtype)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input, attn_impl=attn_impl, device=device, dtype=dtype
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(dim, eps=eps, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim, device=device, dtype=dtype),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim, device=device, dtype=dtype) / dim**0.5)

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        flf_pos_emb: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim, device=device, dtype=dtype),
            nn.Linear(in_dim, in_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, device=device, dtype=dtype),
            nn.LayerNorm(out_dim, device=device, dtype=dtype),
        )
        if flf_pos_emb:
            self.emb_pos = nn.Parameter((torch.zeros(1, FLF_TOKEN_NUM, in_dim)))

    def forward(self, x: torch.Tensor):
        if hasattr(self, "emb_pos"):
            b, s, d = x.shape
            x = x.view(-1, 2 * s, d)
            x = x + self.emb_pos
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size), device=device, dtype=dtype)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim, device=device, dtype=dtype) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation + t_mod).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanDiTStateDictConverter(StateDictConverter):
    def convert(self, state_dict):
        return state_dict


class WanDiT(PreTrainedModel):
    converter = WanDiTStateDictConverter()

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        flf_pos_emb: bool = False,
        attn_impl: Optional[str] = None,
        use_usp: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, device=device, dtype=dtype),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, attn_impl, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps, device=device, dtype=dtype)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, flf_pos_emb, device=device, dtype=dtype)  # clip_feature_dim = 1280

        self.use_usp = use_usp

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)  # b c f h w -> b 4c f h/2 w/2
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,  # clip_vision_encoder(img)
        y: Optional[torch.Tensor] = None,  # vae_encoder(img)
    ):
        with gguf_inference():
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            context = self.text_embedding(context)
            if self.has_image_input:
                x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
                clip_embdding = self.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)  # (b, s1 + s2, d)
            x, (f, h, w) = self.patchify(x)
            freqs = (
                torch.cat(
                    [
                        self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                        self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                )
                .reshape(f * h * w, 1, -1)
                .to(x.device)
            )

            with sequence_parallel([x, freqs], seq_dims=(1, 0), enabled=self.use_usp):
                for block in self.blocks:
                    x = block(x, context, t_mod, freqs)
                x = self.head(x, t)
                (x,) = sequence_parallel_unshard((x,), seq_dims=(1,), seq_lens=(f * h * w,))
            x = self.unpatchify(x, (f, h, w))
            return x

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        device,
        dtype,
        model_type="1.3b-t2v",
        attn_impl: Optional[str] = None,
        use_usp=False,
        assign=True,
    ):
        if model_type == "1.3b-t2v":
            config = json.load(open(WAN_DIT_1_3B_T2V_CONFIG_FILE, "r"))
        elif model_type == "14b-t2v":
            config = json.load(open(WAN_DIT_14B_T2V_CONFIG_FILE, "r"))
        elif model_type == "14b-i2v":
            config = json.load(open(WAN_DIT_14B_I2V_CONFIG_FILE, "r"))
        elif model_type == "14b-flf2v":
            config = json.load(open(WAN_DIT_14B_FLF2V_CONFIG_FILE, "r"))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        with no_init_weights():
            model = torch.nn.utils.skip_init(
                cls, **config, device=device, dtype=dtype, attn_impl=attn_impl, use_usp=use_usp
            )
            model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=assign)
        model.to(device=device, dtype=dtype)
        return model

    def get_tp_plan(self):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
        )
        from torch.distributed.tensor import Replicate, Shard

        tp_plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_embedding.0": ColwiseParallel(),
            "time_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Shard(1), None, None, None),  # sequence parallel
                use_local_output=True,
            ),
            "head": PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            ),
        }
        for idx in range(len(self.blocks)):
            tp_plan.update(
                {
                    f"blocks.{idx}.self_attn": PrepareModuleInput(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    f"blocks.{idx}.self_attn.q": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.self_attn.k": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.self_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.self_attn.o": RowwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.self_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.self_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn": PrepareModuleInput(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    f"blocks.{idx}.cross_attn.q": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.k": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.o": RowwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.k_img": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.v_img": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.norm_k_img": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.ffn": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    f"blocks.{idx}.ffn.0": ColwiseParallel(),
                    f"blocks.{idx}.ffn.2": RowwiseParallel(output_layouts=Shard(1)),
                }
            )
        return tp_plan
