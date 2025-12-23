import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
from torch.nn.utils.rnn import pad_sequence
import math

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.models.basic.transformer_helper import RMSNorm
from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)


class ZImageStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict_ = {}
        for name, param in state_dict.items():
            name_ = name
            if "attention.to_out.0" in name:
                name_ = name.replace("attention.to_out.0", "attention.to_out")
            if "adaLN_modulation.0" in name:
                name_ = name.replace("adaLN_modulation.0", "adaLN_modulation")
            state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = self._from_diffusers(state_dict)
        return state_dict


class ZImageTimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True, device=device, dtype=dtype),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        t_freq = t_freq.to(dtype=weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZImageFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, adaln_embed_dim=256, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True, device=device, dtype=dtype)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, adaln_embed_dim), hidden_size, bias=True, device=device, dtype=dtype),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class ZImageRopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
        device: str = "cuda:0",
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None
        self.device = device

    def precompute_freqs_cis(self, dim: List[int], end: List[int], theta: float = 256.0):
        freqs_cis = []
        for i, (d, e) in enumerate(zip(dim, end)):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
            freqs_cis.append(freqs_cis_i)
        return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != ids.device:
            self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


def apply_rotary_emb_zimage(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.to_k = nn.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.to_v = nn.Linear(dim, dim, bias=False, device=device, dtype=dtype)

        self.norm_q = RMSNorm(head_dim, eps=eps, device=device, dtype=dtype) if qk_norm else None
        self.norm_k = RMSNorm(head_dim, eps=eps, device=device, dtype=dtype) if qk_norm else None

        self.to_out = nn.Linear(dim, dim, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(*q.shape[:2], self.heads, self.head_dim)
        k = k.view(*k.shape[:2], self.heads, self.head_dim)
        v = v.view(*v.shape[:2], self.heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = apply_rotary_emb_zimage(q, freqs_cis)
            k = apply_rotary_emb_zimage(k, freqs_cis)

        out = attention_ops.attention(q, k, v, attn_mask=attn_mask, **kwargs)

        out = out.flatten(2)
        out = self.to_out(out)
        return out


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        adaln_embed_dim: int = 256,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.modulation = modulation

        self.attention = ZImageAttention(
            dim=dim,
            num_heads=n_heads,
            head_dim=dim // n_heads,
            qk_norm=qk_norm,
            eps=1e-5,
            device=device,
            dtype=dtype,
        )

        self.feed_forward = ZImageFeedForward(dim=dim, hidden_dim=int(dim / 3 * 8), device=device, dtype=dtype)

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)

        if modulation:
            self.adaLN_modulation = nn.Linear(min(dim, adaln_embed_dim), 4 * dim, bias=True, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            mod_output = self.adaLN_modulation(adaln_input)
            mod_output = mod_output.unsqueeze(1)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod_output.chunk(4, dim=2)

            gate_msa = gate_msa.tanh()
            gate_mlp = gate_mlp.tanh()
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis=freqs_cis, attn_mask=attn_mask)
            x = x + gate_msa * self.attention_norm2(attn_out)

            ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            x = x + gate_mlp * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(self.attention_norm1(x), freqs_cis=freqs_cis, attn_mask=attn_mask)
            x = x + self.attention_norm2(attn_out)

            ffn_out = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn_out)

        return x


class ZImageDiT(PreTrainedModel):
    converter = ZImageStateDictConverter()
    _supports_parallelization = True

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.dtype = dtype
        self.device = device
        self.ADALN_EMBED_DIM = 256
        self.SEQ_MULTI_OF = 32

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            key = f"{patch_size}-{f_patch_size}"
            all_x_embedder[key] = nn.Linear(
                f_patch_size * patch_size * patch_size * in_channels, dim, bias=True, device=device, dtype=dtype
            )
            all_final_layer[key] = ZImageFinalLayer(
                dim, patch_size * patch_size * f_patch_size * in_channels, device=device, dtype=dtype
            )

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True, device=device, dtype=dtype
                )
                for _ in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False, device=device, dtype=dtype
                )
                for _ in range(n_refiner_layers)
            ]
        )

        self.t_embedder = ZImageTimestepEmbedder(
            min(dim, self.ADALN_EMBED_DIM), mid_size=1024, device=device, dtype=dtype
        )

        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps, device=device, dtype=dtype),
            nn.Linear(cap_feat_dim, dim, bias=True, device=device, dtype=dtype),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(dim, n_heads, n_kv_heads, norm_eps, qk_norm, device=device, dtype=dtype)
                for _ in range(n_layers)
            ]
        )

        self.rope_embedder = ZImageRopeEmbedder(
            theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens, device=device
        )

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(self, all_image, all_cap_feats, patch_size, f_patch_size):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out, all_image_size, all_image_pos_ids, all_image_pad_mask = [], [], [], []
        all_cap_feats_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % self.SEQ_MULTI_OF
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1), start=(1, 0, 0), device=device
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)

            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_cap_pad_mask.append(cap_pad_mask)

            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
            all_cap_feats_out.append(cap_padded_feat)

            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % self.SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens), start=(cap_ori_len + cap_padding_len + 1, 0, 0), device=device
            ).flatten(0, 2)

            if image_padding_len > 0:
                pad_grid = self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device).flatten(0, 2)
                image_padded_pos_ids = torch.cat([image_ori_pos_ids, pad_grid.repeat(image_padding_len, 1)], dim=0)
                image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
                image_pad_mask = torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            else:
                image_padded_pos_ids = image_ori_pos_ids
                image_padded_feat = image
                image_pad_mask = torch.zeros((image_ori_len,), dtype=torch.bool, device=device)

            all_image_pos_ids.append(image_padded_pos_ids)
            all_image_pad_mask.append(image_pad_mask)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH, pW, pF = patch_size, patch_size, f_patch_size
        bsz = len(x)
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    def forward(
        self,
        image: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        cap_feats: Union[torch.Tensor, List[torch.Tensor]],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        if isinstance(image, torch.Tensor):
            image = list(image.unbind(0))
        if isinstance(cap_feats, torch.Tensor):
            cap_feats = list(cap_feats.unbind(0))

        use_cfg = len(image) > 1
        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        with (
            fp8_inference(fp8_linear_enabled),
            gguf_inference(),
            cfg_parallel((image, timestep, cap_feats), use_cfg=use_cfg),
        ):
            bsz = len(image)
            device = image[0].device

            t = timestep * self.t_scale
            t = self.t_embedder(t)

            (x, cap_feats_processed, x_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask) = (
                self.patchify_and_embed(image, cap_feats, patch_size, f_patch_size)
            )

            x_item_seqlens = [len(_) for _ in x]
            x_max_item_seqlen = max(x_item_seqlens)
            x = torch.cat(x, dim=0)
            x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

            adaln_input = t.type_as(x)

            x[torch.cat(x_inner_pad_mask)] = self.x_pad_token

            x = list(x.split(x_item_seqlens, dim=0))
            x_freqs_cis = list(
                self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0)
            )

            x = pad_sequence(x, batch_first=True, padding_value=0.0)
            x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
            x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

            x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(x_item_seqlens):
                x_attn_mask[i, :seq_len] = 1

            x_attn_mask_4d = x_attn_mask.unsqueeze(1).unsqueeze(1)

            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask_4d, x_freqs_cis, adaln_input)

            cap_item_seqlens = [len(_) for _ in cap_feats_processed]
            cap_max_item_seqlen = max(cap_item_seqlens)
            cap_feats_tensor = torch.cat(cap_feats_processed, dim=0)
            cap_feats_tensor = self.cap_embedder(cap_feats_tensor)
            mask_tmp = torch.cat(cap_inner_pad_mask)
            target_len = mask_tmp.sum()
            if target_len > 0:
                cap_feats_tensor[mask_tmp] = self.cap_pad_token.to(dtype=cap_feats_tensor.dtype).expand(target_len, -1)

            cap_feats_list = list(cap_feats_tensor.split(cap_item_seqlens, dim=0))
            cap_freqs_cis = list(
                self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
            )

            cap_feats_padded = pad_sequence(cap_feats_list, batch_first=True, padding_value=0.0)
            cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
            cap_freqs_cis = cap_freqs_cis[:, : cap_feats_padded.shape[1]]

            cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(cap_item_seqlens):
                cap_attn_mask[i, :seq_len] = 1
            cap_attn_mask_4d = cap_attn_mask.unsqueeze(1).unsqueeze(1)

            for layer in self.context_refiner:
                cap_feats_padded = layer(cap_feats_padded, cap_attn_mask_4d, cap_freqs_cis, adaln_input=None)

            unified = []
            unified_freqs_cis = []
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                cap_len = cap_item_seqlens[i]
                unified.append(torch.cat([x[i][:x_len], cap_feats_padded[i][:cap_len]]))
                unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))

            unified_item_seqlens = [len(_) for _ in unified]
            unified_max_item_seqlen = max(unified_item_seqlens)

            unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
            unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)

            unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_item_seqlens):
                unified_attn_mask[i, :seq_len] = 1
            unified_attn_mask_4d = unified_attn_mask.unsqueeze(1).unsqueeze(1)

            with sequence_parallel((unified, unified_freqs_cis), seq_dims=(1, 1)):
                for layer in self.layers:
                    unified = layer(unified, unified_attn_mask_4d, unified_freqs_cis, adaln_input)
                (unified,) = sequence_parallel_unshard((unified,), seq_dims=(1,), seq_lens=(unified.shape[1],))
            unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
            unified_list = list(unified.unbind(dim=0))

            output = self.unpatchify(unified_list, x_size, patch_size, f_patch_size)

        (output,) = cfg_parallel_unshard((output,), use_cfg=use_cfg)

        return output

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        **kwargs,
    ):
        model = cls(device="meta", dtype=dtype, **kwargs)
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def compile_repeated_blocks(self, *args, **kwargs):
        for block in self.noise_refiner:
            block.compile(*args, **kwargs)
        for block in self.context_refiner:
            block.compile(*args, **kwargs)
        for block in self.layers:
            block.compile(*args, **kwargs)

    def get_fsdp_module_cls(self):
        return {ZImageTransformerBlock}
