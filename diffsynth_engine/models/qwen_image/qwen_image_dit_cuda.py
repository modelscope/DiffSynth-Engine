import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional
from einops import rearrange
from math import prod

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.models.basic.timestep import TimestepEmbeddings
from diffsynth_engine.models.basic.transformer_helper import AdaLayerNorm, GELU, RMSNorm
from diffsynth_engine.models.qwen_image.qwen_image_cuda_ext import (
    modulate_forward as modulate_forward_cuda,
    modulate_indexed_forward as modulate_indexed_forward_cuda,
    rotary_emb_forward as rotary_emb_forward_cuda,
)

from diffsynth_engine.utils.gguf import gguf_inference
from diffsynth_engine.utils.fp8_linear import fp8_inference
from diffsynth_engine.utils.parallel import (
    cfg_parallel,
    cfg_parallel_unshard,
    sequence_parallel,
    sequence_parallel_unshard,
)


class QwenImageDiTStateDictConverter(StateDictConverter):
    def __init__(self):
        pass

    def _from_diffusers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict_ = {}
        dim = 3072
        for name, param in state_dict.items():
            name_ = name
            if name.startswith("transformer") and "attn.to_out.0" in name:
                name_ = name.replace("attn.to_out.0", "attn.to_out")
            if "timestep_embedder.linear_1" in name:
                name_ = name.replace("timestep_embedder.linear_1", "timestep_embedder.0")
            if "timestep_embedder.linear_2" in name:
                name_ = name.replace("timestep_embedder.linear_2", "timestep_embedder.2")
            if "norm_out.linear" in name:
                param = torch.concat([param[dim:], param[:dim]], dim=0)
            state_dict_[name_] = param
        return state_dict_

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = self._from_diffusers(state_dict)
        return state_dict


class QwenEmbedRope(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        scale_rope=False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        with torch.device("cpu" if device == "meta" else device):
            pos_index = torch.arange(10000)
            neg_index = torch.arange(10000).flip(0) * -1 - 1
            self.pos_freqs = torch.cat(
                [
                    self.rope_params(pos_index, self.axes_dim[0], self.theta),
                    self.rope_params(pos_index, self.axes_dim[1], self.theta),
                    self.rope_params(pos_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            )
            self.neg_freqs = torch.cat(
                [
                    self.rope_params(neg_index, self.axes_dim[0], self.theta),
                    self.rope_params(neg_index, self.axes_dim[1], self.theta),
                    self.rope_params(neg_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            )
        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_length, device):
        """
        Args:
            video_fhw (List[Tuple[int, int, int]]): A list of (frame, height, width) tuples for each video/image
            txt_length (int): The maximum length of the text sequences
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + txt_length, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class QwenFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(GELU(dim, inner_dim, approximate="tanh", device=device, dtype=dtype))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]):
    if (
        isinstance(freqs_cis, torch.Tensor)
        and x.is_cuda
        and freqs_cis.is_cuda
        and x.is_contiguous()
        and freqs_cis.is_contiguous()
        and x.dim() == 4
        and freqs_cis.dim() == 2
        and x.shape[1] == freqs_cis.shape[0]
        and x.shape[-1] % 2 == 0
        and freqs_cis.dtype == torch.complex64
    ):
        x_out = rotary_emb_forward_cuda(x, freqs_cis)
        if x_out is not None:
            return x_out

    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (b, s, h, d) -> (b, s, h, d/2, 2)
    x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3)  # (b, s, h, d/2, 2) -> (b, s, h, d)
    return x_out.type_as(x)


@dataclass
class ImageTokenCache:
    static_indices: torch.Tensor
    img_k_static: torch.Tensor
    img_v_static: torch.Tensor


class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.to_k = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.to_v = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.norm_q = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.add_q_proj = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.add_k_proj = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.add_v_proj = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.to_out = nn.Linear(dim_a, dim_a, device=device, dtype=dtype)
        self.to_add_out = nn.Linear(dim_b, dim_b, device=device, dtype=dtype)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b s (h d) -> b s h d", h=self.num_heads)

    def project_image_qkv(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_q = self._reshape_heads(self.to_q(image))
        img_k = self._reshape_heads(self.to_k(image))
        img_v = self._reshape_heads(self.to_v(image))
        return img_q, img_k, img_v

    def project_image_kv(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_k = self._reshape_heads(self.to_k(image))
        img_v = self._reshape_heads(self.to_v(image))
        img_k = self.norm_k(img_k)
        return img_k, img_v

    def project_text_qkv(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        txt_q = self._reshape_heads(self.add_q_proj(text))
        txt_k = self._reshape_heads(self.add_k_proj(text))
        txt_v = self._reshape_heads(self.add_v_proj(text))
        return txt_q, txt_k, txt_v

    def normalize_qk(
        self, img_q: torch.Tensor, img_k: torch.Tensor, txt_q: torch.Tensor, txt_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        return img_q, img_k, txt_q, txt_k

    def apply_rotary(
        self,
        img_q: torch.Tensor,
        img_k: torch.Tensor,
        txt_q: torch.Tensor,
        txt_k: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if rotary_emb is None:
            return img_q, img_k, txt_q, txt_k
        img_freqs, txt_freqs = rotary_emb
        img_q = apply_rotary_emb_qwen(img_q, img_freqs)
        img_k = apply_rotary_emb_qwen(img_k, img_freqs)
        txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
        txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)
        return img_q, img_k, txt_q, txt_k

    def apply_image_rotary(
        self,
        img_q: Optional[torch.Tensor],
        img_k: torch.Tensor,
        img_freqs: torch.Tensor,
        token_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if token_indices is not None:
            img_freqs = img_freqs.index_select(0, token_indices)
        if img_q is not None:
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
        img_k = apply_rotary_emb_qwen(img_k, img_freqs)
        return img_q, img_k

    def apply_text_rotary(
        self, txt_q: torch.Tensor, txt_k: torch.Tensor, txt_freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
        txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)
        return txt_q, txt_k

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.project_image_qkv(image)
        txt_q, txt_k, txt_v = self.project_text_qkv(text)
        img_q, img_k, txt_q, txt_k = self.normalize_qk(img_q, img_k, txt_q, txt_k)
        img_q, img_k, txt_q, txt_k = self.apply_rotary(img_q, img_k, txt_q, txt_k, rotary_emb)

        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)

        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        joint_attn_out = attention_ops.attention(joint_q, joint_k, joint_v, attn_mask=attn_mask, **attn_kwargs)

        joint_attn_out = rearrange(joint_attn_out, "b s h d -> b s (h d)").to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, : text.shape[1], :]
        img_attn_output = joint_attn_out[:, text.shape[1] :, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def forward_with_cached_image_kv(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        cached_img_k: torch.FloatTensor,
        cached_img_v: torch.FloatTensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        image_token_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if image_token_indices is None:
            image_token_indices = torch.arange(image.shape[1], device=image.device, dtype=torch.long)

        img_q, img_k, img_v = self.project_image_qkv(image)
        txt_q, txt_k, txt_v = self.project_text_qkv(text)
        img_q, img_k, txt_q, txt_k = self.normalize_qk(img_q, img_k, txt_q, txt_k)

        if rotary_emb is not None:
            img_freqs, txt_freqs = rotary_emb
            img_q, img_k = self.apply_image_rotary(
                img_q=img_q,
                img_k=img_k,
                img_freqs=img_freqs,
                token_indices=image_token_indices,
            )
            txt_q, txt_k = self.apply_text_rotary(txt_q, txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k, cached_img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v, cached_img_v], dim=1)

        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        attn_mask_dyn = attn_mask
        if attn_mask is not None:
            txt_len = text.shape[1]
            query_indices = torch.cat(
                [
                    torch.arange(txt_len, device=image.device, dtype=torch.long),
                    txt_len + image_token_indices,
                ],
                dim=0,
            )
            attn_mask_dyn = attn_mask.index_select(2, query_indices)
        joint_attn_out = attention_ops.attention(joint_q, joint_k, joint_v, attn_mask=attn_mask_dyn, **attn_kwargs)
        joint_attn_out = rearrange(joint_attn_out, "b s h d -> b s (h d)").to(joint_q.dtype)

        txt_attn_output = self.to_add_out(joint_attn_out[:, : text.shape[1], :])
        img_attn_output = self.to_out(joint_attn_out[:, text.shape[1] :, :])
        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            device=device,
            dtype=dtype,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim, device=device, dtype=dtype)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim, device=device, dtype=dtype)
        self.zero_cond_t = zero_cond_t
        self._image_token_cache: Optional[ImageTokenCache] = None

    def clear_image_token_cache(self):
        self._image_token_cache = None

    def _resolve_cache_state(
        self, modulate_index: Optional[torch.Tensor], use_image_token_cache: bool
    ) -> Tuple[Optional[torch.Tensor], bool, bool, Optional[int]]:
        static_indices = self._get_static_token_indices(modulate_index) if use_image_token_cache else None
        use_static_cache = use_image_token_cache and static_indices is not None
        cache_ready = (
            use_static_cache
            and self._image_token_cache is not None
            and self._image_token_cache.static_indices.shape == static_indices.shape
            and torch.equal(self._image_token_cache.static_indices, static_indices)
        )
        dynamic_seq_len = None
        if use_static_cache:
            dynamic_seq_len = int((modulate_index[0].squeeze(-1) == 0).sum().item())
        return static_indices, use_static_cache, cache_ready, dynamic_seq_len

    def _prepare_cached_image(self, image: torch.Tensor, cache_ready: bool, dynamic_seq_len: Optional[int]) -> torch.Tensor:
        if cache_ready and dynamic_seq_len is not None and image.shape[1] != dynamic_seq_len:
            return image[:, :dynamic_seq_len, :]
        return image

    def _align_modulate_index_to_image(
        self, modulate_index: Optional[torch.Tensor], image: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if modulate_index is not None and modulate_index.shape[1] != image.shape[1]:
            # Cached path runs on dynamic-only tokens, which are the prefix (all zero-valued index).
            return modulate_index[:, : image.shape[1], :]
        return modulate_index

    def _apply_img_mlp_residual(
        self, image: torch.Tensor, img_mod_mlp: torch.Tensor, img_modulate_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, img_modulate_index)
        img_mlp_out = self.img_mlp(img_modulated_2)
        return image + img_gate_2 * img_mlp_out

    def _apply_txt_mlp_residual(self, text: torch.Tensor, txt_mod_mlp: torch.Tensor) -> torch.Tensor:
        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)
        return text + txt_gate_2 * txt_mlp_out

    def _get_static_token_indices(self, modulate_index: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if modulate_index is None:
            return None
        index = modulate_index
        if index.dim() == 3:
            index = index.squeeze(-1)
        # `modulate_index` is shared across the batch; use the first sample mask.
        static_mask = index[0] == 1
        if not torch.any(static_mask):
            return None
        return torch.nonzero(static_mask, as_tuple=False).squeeze(-1)

    def _build_static_kv_cache(
        self,
        img_modulated: torch.Tensor,
        static_indices: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_static = img_modulated.index_select(1, static_indices)
        img_k_static, img_v_static = self.attn.project_image_kv(img_static)
        if rotary_emb is not None:
            img_freqs, _ = rotary_emb
            _, img_k_static = self.attn.apply_image_rotary(
                img_q=None,
                img_k=img_k_static,
                img_freqs=img_freqs,
                token_indices=static_indices,
            )
        return img_k_static, img_v_static

    def _modulate(self, x, mod_params, index=None):
        if (
            x.is_cuda
            and mod_params.is_cuda
            and x.is_contiguous()
            and mod_params.is_contiguous()
            and x.dim() == 3
            and mod_params.dim() == 2
            and mod_params.shape[1] == x.shape[2] * 3
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and mod_params.dtype == x.dtype
        ):
            if index is None and mod_params.shape[0] == x.shape[0]:
                out = modulate_forward_cuda(x, mod_params)
                if out is not None:
                    return out

            if (
                index is not None
                and index.is_cuda
                and index.is_contiguous()
                and index.dim() == 3
                and index.shape[1] == x.shape[1]
                and index.shape[2] == 1
                and mod_params.shape[0] == x.shape[0] * 2
                and index.dtype in (torch.int32, torch.int64)
            ):
                out = modulate_indexed_forward_cuda(x, mod_params, index)
                if out is not None:
                    return out

        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
            shift_0_exp = shift_0.unsqueeze(1)
            shift_1_exp = shift_1.unsqueeze(1)
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)
            shift_result = torch.where(index == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)
        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[List[int]] = None,
        use_image_token_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        static_indices, use_static_cache, cache_ready, dynamic_seq_len = self._resolve_cache_state(
            modulate_index, use_image_token_cache
        )
        image = self._prepare_cached_image(image, cache_ready, dynamic_seq_len)
        img_modulate_index = self._align_modulate_index_to_image(modulate_index, image)

        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, img_modulate_index)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        if not cache_ready:
            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated,
                text=txt_modulated,
                rotary_emb=rotary_emb,
                attn_mask=attn_mask,
                attn_kwargs=attn_kwargs,
            )

            image = image + img_gate * img_attn_out
            text = text + txt_gate * txt_attn_out
            image = self._apply_img_mlp_residual(image, img_mod_mlp, img_modulate_index)
            text = self._apply_txt_mlp_residual(text, txt_mod_mlp)

            if use_static_cache:
                img_k_static, img_v_static = self._build_static_kv_cache(img_modulated, static_indices, rotary_emb)
                self._image_token_cache = ImageTokenCache(
                    static_indices=static_indices.detach().clone(),
                    img_k_static=img_k_static.detach().clone(),
                    img_v_static=img_v_static.detach().clone(),
                )
        else:
            cached_k_static = self._image_token_cache.img_k_static
            cached_v_static = self._image_token_cache.img_v_static

            dynamic_indices = torch.arange(image.shape[1], device=image.device, dtype=torch.long)
            img_attn_dyn_out, txt_attn_out = self.attn.forward_with_cached_image_kv(
                image=img_modulated,
                text=txt_modulated,
                cached_img_k=cached_k_static,
                cached_img_v=cached_v_static,
                rotary_emb=rotary_emb,
                attn_mask=attn_mask,
                attn_kwargs=attn_kwargs,
                image_token_indices=dynamic_indices,
            )
            text = text + txt_gate * txt_attn_out

            image = image + img_gate * img_attn_dyn_out
            image = self._apply_img_mlp_residual(image, img_mod_mlp, img_modulate_index)
            text = self._apply_txt_mlp_residual(text, txt_mod_mlp)

        return text, image


class QwenImageDiT(PreTrainedModel):
    converter = QwenImageDiTStateDictConverter()
    _supports_parallelization = True

    def __init__(
        self,
        num_layers: int = 60,
        zero_cond_t: bool = False,
        use_image_token_cache: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True, device=device)

        self.time_text_embed = TimestepEmbeddings(256, 3072, device=device, dtype=dtype)

        self.txt_norm = RMSNorm(3584, eps=1e-6, device=device, dtype=dtype)

        self.img_in = nn.Linear(64, 3072, device=device, dtype=dtype)
        self.txt_in = nn.Linear(3584, 3072, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                    zero_cond_t=zero_cond_t,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, device=device, dtype=dtype)
        self.proj_out = nn.Linear(3072, 64, device=device, dtype=dtype)
        self.zero_cond_t = zero_cond_t
        self.use_image_token_cache = use_image_token_cache

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(
            hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    def process_entity_masks(
        self,
        text: torch.Tensor,
        text_seq_lens: torch.LongTensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        video_fhw: List[Tuple[int, int, int]],
        entity_text: List[torch.Tensor],
        entity_seq_lens: List[torch.LongTensor],
        entity_masks: List[torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ):
        entity_seq_lens = [seq_lens.max().item() for seq_lens in entity_seq_lens]
        text_seq_lens = entity_seq_lens + [text_seq_lens.max().item()]
        entity_text = [
            self.txt_in(self.txt_norm(text[:, :seq_len])) for text, seq_len in zip(entity_text, entity_seq_lens)
        ]
        text = torch.cat(entity_text + [text], dim=1)

        entity_txt_freqs = [self.pos_embed(video_fhw, seq_len, device)[1] for seq_len in entity_seq_lens]
        img_freqs, txt_freqs = rotary_emb
        txt_freqs = torch.cat(entity_txt_freqs + [txt_freqs], dim=0)
        rotary_emb = (img_freqs, txt_freqs)

        global_mask = torch.ones_like(entity_masks[0], device=device, dtype=dtype)
        patched_masks = [self.patchify(mask) for mask in entity_masks + [global_mask]]
        batch_size, image_seq_len = patched_masks[0].shape[:2]
        total_seq_len = sum(text_seq_lens) + image_seq_len
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), device=device, dtype=torch.bool)

        # text-image attention mask
        img_start, img_end = sum(text_seq_lens), total_seq_len
        cumsum = [0]
        for seq_len in text_seq_lens:
            cumsum.append(cumsum[-1] + seq_len)
        for i, patched_mask in enumerate(patched_masks):
            txt_start, txt_end = cumsum[i], cumsum[i + 1]
            mask = torch.sum(patched_mask, dim=-1) > 0
            mask = mask.unsqueeze(1).repeat(1, text_seq_lens[i], 1)
            # text-to-image attention
            attention_mask[:, txt_start:txt_end, img_start:img_end] = mask
            # image-to-text attention
            attention_mask[:, img_start:img_end, txt_start:txt_end] = mask.transpose(1, 2)
        # entity text tokens should not attend to each other
        for i in range(len(text_seq_lens)):
            for j in range(len(text_seq_lens)):
                if i == j:
                    continue
                i_start, i_end = cumsum[i], cumsum[i + 1]
                j_start, j_end = cumsum[j], cumsum[j + 1]
                attention_mask[:, i_start:i_end, j_start:j_end] = False

        attn_mask = torch.zeros_like(attention_mask, device=device, dtype=dtype)
        attn_mask[~attention_mask] = -torch.inf
        attn_mask = attn_mask.unsqueeze(1)
        return text, rotary_emb, attn_mask

    def forward(
        self,
        image: torch.Tensor,
        edit: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        text: torch.Tensor = None,
        text_seq_lens: torch.LongTensor = None,
        context_latents: Optional[torch.Tensor] = None,
        entity_text: Optional[List[torch.Tensor]] = None,
        entity_seq_lens: Optional[List[torch.LongTensor]] = None,
        entity_masks: Optional[List[torch.Tensor]] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        h, w = image.shape[-2:]
        fp8_linear_enabled = getattr(self, "fp8_linear_enabled", False)
        use_cfg = image.shape[0] > 1
        with (
            fp8_inference(fp8_linear_enabled),
            gguf_inference(),
            cfg_parallel(
                (
                    image,
                    *(edit if edit is not None else ()),
                    timestep,
                    text,
                    text_seq_lens,
                    *(entity_text if entity_text is not None else ()),
                    *(entity_seq_lens if entity_seq_lens is not None else ()),
                    *(entity_masks if entity_masks is not None else ()),
                    context_latents,
                ),
                use_cfg=use_cfg,
            ),
        ):
            if self.zero_cond_t:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = None
            conditioning = self.time_text_embed(timestep, image.dtype)
            video_fhw = [(1, h // 2, w // 2)]  # frame, height, width
            text_seq_len = text_seq_lens.max().item()
            image = self.patchify(image)
            image_seq_len = image.shape[1]
            if context_latents is not None:
                context_latents = context_latents.to(dtype=image.dtype)
                context_latents = self.patchify(context_latents)
                image = torch.cat([image, context_latents], dim=1)
                video_fhw += [(1, h // 2, w // 2)]
            if edit is not None:
                for img in edit:
                    img = img.to(dtype=image.dtype)
                    edit_h, edit_w = img.shape[-2:]
                    img = self.patchify(img)
                    image = torch.cat([image, img], dim=1)
                    video_fhw += [(1, edit_h // 2, edit_w // 2)]
            if self.zero_cond_t:
                modulate_index = torch.tensor(
                    [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in [video_fhw]],
                    device=timestep.device,
                    dtype=torch.int,
                )
                modulate_index = modulate_index.unsqueeze(-1)
            rotary_emb = self.pos_embed(video_fhw, text_seq_len, image.device)

            image = self.img_in(image)
            text = self.txt_in(self.txt_norm(text[:, :text_seq_len]))

            attn_mask = None
            if entity_text is not None:
                text, rotary_emb, attn_mask = self.process_entity_masks(
                    text,
                    text_seq_lens,
                    rotary_emb,
                    video_fhw,
                    entity_text,
                    entity_seq_lens,
                    entity_masks,
                    image.device,
                    image.dtype,
                )

            # warning: Eligen does not work with sequence parallel because long context attention does not support attention masks
            img_freqs, txt_freqs = rotary_emb
            with sequence_parallel((image, text, img_freqs, txt_freqs, modulate_index), seq_dims=(1, 1, 0, 0, 1)):
                rotary_emb = (img_freqs, txt_freqs)
                # Cache decision is per denoising step, but the KV cache itself lives per block.
                # We pass the flag into every block so each block can decide whether to reuse
                # the static-image KV part (non-dynamic tokens) or run full attention.
                for block in self.transformer_blocks:
                    text, image = block(
                        image=image,
                        text=text,
                        temb=conditioning,
                        rotary_emb=rotary_emb,
                        attn_mask=attn_mask,
                        attn_kwargs=attn_kwargs,
                        modulate_index=modulate_index,
                        # Controls block-level static image KV reuse path.
                        use_image_token_cache=self.use_image_token_cache,
                    )
                if self.zero_cond_t:
                    conditioning = conditioning.chunk(2, dim=0)[0]
                image = self.norm_out(image, conditioning)
                image = self.proj_out(image)
                (image,) = sequence_parallel_unshard((image,), seq_dims=(1,), seq_lens=(image_seq_len,))
            image = image[:, :image_seq_len]
            image = self.unpatchify(image, h, w)

        (image,) = cfg_parallel_unshard((image,), use_cfg=use_cfg)
        return image

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        num_layers: int = 60,
        use_zero_cond_t: bool = False,
    ):
        model = cls(device="meta", dtype=dtype, num_layers=num_layers, zero_cond_t=use_zero_cond_t)
        model = model.requires_grad_(False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model

    def compile_repeated_blocks(self, *args, **kwargs):
        for block in self.transformer_blocks:
            block.compile(*args, **kwargs)

    def clear_image_token_caches(self):
        for block in self.transformer_blocks:
            block.clear_image_token_cache()

    def set_image_token_cache_enabled(self, enabled: bool, clear_existing_cache: bool = False):
        self.use_image_token_cache = enabled
        if clear_existing_cache:
            self.clear_image_token_caches()

    def get_fsdp_module_cls(self):
        return {QwenImageTransformerBlock}
