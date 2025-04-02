import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.flag import (
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
    XFORMERS_AVAILABLE,
    SDPA_AVAILABLE,
)

logger = logging.get_logger(__name__)


def _eager_attn(query, key, value, attn_mask=None, scale=None):
    scale = 1 / query.shape[-1] ** 0.5 if scale is None else scale
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(-1)
    return attn @ value

def attention(q, k, v, attn_mask=None, attn_impl:Optional[str]=None):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    assert attn_impl in [None, 'auto', 'eager', 'flash_attn_2', 'flash_attn_3', 'xformers', 'sdpa', 'sage_attn', 'sparge_attn']
    if attn_impl is None or attn_impl == "auto":    
        if FLASH_ATTN_3_AVAILABLE:
            from flash_attn_interface import flash_attn_varlen_func
            return flash_attn_varlen_func(q, k, v, attn_mask=attn_mask)
        elif FLASH_ATTN_2_AVAILABLE:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(q, k, v, attn_mask=attn_mask)
        elif XFORMERS_AVAILABLE:
            import xformers.ops as xops
            return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        elif SDPA_AVAILABLE:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)
        else:
            return _eager_attn(q, k, v, attn_mask=attn_mask)
    else:
        # (b, s, n, d)
        if attn_impl == "eager":
            return _eager_attn(q, k, v, attn_mask=attn_mask)
        elif attn_impl == "flash_attn_3":
            if not FLASH_ATTN_3_AVAILABLE:
                raise ValueError("attn_impl is 'flash_attn_3', but Flash attention 3 is not available")
            from flash_attn_interface import flash_attn_varlen_func
            return flash_attn_varlen_func(q, k, v, attn_mask=attn_mask)
        elif attn_impl == "flash_attn_2":
            if not FLASH_ATTN_2_AVAILABLE:
                raise ValueError("attn_impl is 'flash_attn_2', but Flash attention 2 is not available")
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(q, k, v, attn_mask=attn_mask)
        elif attn_impl == "xformers":
            if not XFORMERS_AVAILABLE:
                raise ValueError("attn_impl is 'xformers', but XFormers is not available")
            import xformers.ops as xops
            return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        else:
            # (b, n, s, d)        
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if attn_impl == "sdpa":
                if not SDPA_AVAILABLE:
                    raise ValueError("attn_impl is 'sdpa', but Torch SDPA is not available")
                output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            elif attn_impl == "sage_attn":
                if not SAGE_ATTN_AVAILABLE:
                    raise ValueError("attn_impl is 'sage_attn', but Sage attention is not available")
                from sageattention import sageattn
                output = sageattn(q, k, v, attn_mask=attn_mask)
            elif attn_impl == "sparge_attn":
                if not SPARGE_ATTN_AVAILABLE:
                    raise ValueError("attn_impl is 'sparge_attn', but Sparge attention is not available")
                from spas_sage_attn import spas_sage2_attn_meansim_cuda
                output = spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=attn_mask)
            else:
                raise ValueError(f"Invalid attention implementation: {attn_impl}")
            return output.transpose(1,2)

class Attention(nn.Module):
    def __init__(
        self,
        q_dim,
        num_heads,
        head_dim,
        kv_dim=None,
        bias_q=False,
        bias_kv=False,
        bias_out=False,
        scale=None,
        attn_impl: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(q_dim, dim_inner, bias=bias_q, device=device, dtype=dtype)
        self.to_k = nn.Linear(kv_dim, dim_inner, bias=bias_kv, device=device, dtype=dtype)
        self.to_v = nn.Linear(kv_dim, dim_inner, bias=bias_kv, device=device, dtype=dtype)
        self.to_out = nn.Linear(dim_inner, q_dim, bias=bias_out, device=device, dtype=dtype)
        self.attn_impl = attn_impl

    def forward(
        self,
        x:torch.Tensor,
        y:Optional[torch.Tensor]=None,
        attn_mask:Optional[torch.Tensor]=None,
    ):
        if y is None:
            y = x
        q = rearrange(self.to_q(x), "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(self.to_k(y), "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(self.to_v(y), "b s (n d) -> b s n d", n=self.num_heads)
        out = attention(q, k, v, attn_mask=attn_mask, attn_impl=self.attn_impl)
        out = rearrange(out, "b s n d -> b s (n d)", n=self.num_heads)
        return self.to_out(out)
