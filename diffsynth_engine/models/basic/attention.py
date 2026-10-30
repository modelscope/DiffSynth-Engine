import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.flag import (
    FLASH_ATTN_4_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
    XFORMERS_AVAILABLE,
    SDPA_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    SPARGE_ATTN_AVAILABLE,
    VIDEO_SPARSE_ATTN_AVAILABLE,
    AITER_AVAILABLE,
)
from diffsynth_engine.utils.platform import DTYPE_FP8, IS_NPU

FA3_MAX_HEADDIM = 256

logger = logging.get_logger(__name__)

if FLASH_ATTN_4_AVAILABLE:
    from flash_attn.cute.interface import flash_attn_func as flash_attn4
if FLASH_ATTN_3_AVAILABLE:
    from flash_attn_interface import flash_attn_func as flash_attn3
if FLASH_ATTN_2_AVAILABLE:
    from flash_attn import flash_attn_func as flash_attn2
if XFORMERS_AVAILABLE:
    from xformers.ops import memory_efficient_attention

    def memory_align(x: torch.Tensor, dim=-1, alignment: int = 8):
        padding_size = (alignment - x.shape[dim] % alignment) % alignment
        padded_x = F.pad(x, (0, padding_size), "constant", 0)
        return padded_x[..., : x.shape[dim]]

    def xformers_attn(q, k, v, attn_mask=None, scale=None):
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = repeat(attn_mask, "S L -> B H S L", B=q.shape[0], H=q.shape[2])
            attn_mask = memory_align(attn_mask.contiguous())
        return memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=scale)


if SDPA_AVAILABLE:

    def sdpa_attn(q, k, v, attn_mask=None, is_causal=False, scale=None):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
        return out.transpose(1, 2)


if SAGE_ATTN_AVAILABLE:
    from sageattention import sageattn

    def sage_attn(q, k, v, attn_mask=None, scale=None):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = sageattn(q, k, v, attn_mask=attn_mask, sm_scale=scale)
        return out.transpose(1, 2)


if SPARGE_ATTN_AVAILABLE:
    from spas_sage_attn import spas_sage2_attn_meansim_cuda
    from spas_sage_attn.autotune import SparseAttentionMeansim

    def sparge_attn(
        q,
        k,
        v,
        attn_mask=None,
        scale=None,
        smooth_k=True,
        simthreshd1=0.6,
        cdfthreshd=0.98,
        pvthreshd=50,
    ):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = spas_sage2_attn_meansim_cuda(
            q,
            k,
            v,
            attn_mask=attn_mask,
            scale=scale,
            smooth_k=smooth_k,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            pvthreshd=pvthreshd,
        )
        return out.transpose(1, 2)


if AITER_AVAILABLE:
    from aiter import flash_attn_func as aiter_flash_attn
    from aiter import flash_attn_fp8_pertensor_func as aiter_flash_attn_fp8

if VIDEO_SPARSE_ATTN_AVAILABLE:
    from diffsynth_engine.models.basic.video_sparse_attention import (
        video_sparse_attn,
        distributed_video_sparse_attn,
    )


def eager_attn(q, k, v, attn_mask=None, scale=None):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = 1 / q.shape[-1] ** 0.5 if scale is None else scale
    q = q * scale
    attn = torch.matmul(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(-1)
    out = attn @ v
    return out.transpose(1, 2)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    attn_impl: Optional[str] = "auto",
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    **kwargs,
):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    assert attn_impl in [
        None,
        "auto",
        "eager",
        "fa2",
        "fa3",
        "fa3_fp8",
        "fa4",
        "aiter",
        "aiter_fp8",
        "xformers",
        "sdpa",
        "sage",
        "sparge",
        "vsa",
    ]
    flash_attn3_compatible = q.shape[-1] <= FA3_MAX_HEADDIM
    if attn_impl is None or attn_impl == "auto":
        if FLASH_ATTN_4_AVAILABLE:
            # FA4 also has the same max-head-256 limitation as FA3
            if flash_attn3_compatible and attn_mask is None:
                attn_out = flash_attn4(q, k, v, softmax_scale=scale)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                return attn_out
            else:
                if not flash_attn3_compatible:
                    logger.warning(
                        f"head_dim={q.shape[-1]}, but flash_attn_4 only supports head dimension at most {FA3_MAX_HEADDIM}, will use fallback attention implementation"
                    )
                else:
                    logger.debug(
                        "flash_attn_4 does not support attention mask, will use fallback attention implementation"
                    )
        if FLASH_ATTN_3_AVAILABLE:
            if flash_attn3_compatible and attn_mask is None:
                return flash_attn3(q, k, v, softmax_scale=scale)
            else:
                if not flash_attn3_compatible:
                    logger.warning(
                        f"head_dim={q.shape[-1]}, but flash_attn_3 only supports head dimension at most {FA3_MAX_HEADDIM}, will use fallback attention implementation"
                    )
                else:
                    logger.debug(
                        "flash_attn_3 does not support attention mask, will use fallback attention implementation"
                    )
        if AITER_AVAILABLE:
            if flash_attn3_compatible:
                return aiter_flash_attn(q, k, v, softmax_scale=scale)
            else:
                logger.warning(
                    f"head_dim={q.shape[-1]}, but aiter_flash_attn only supports head dimension at most {FA3_MAX_HEADDIM}, will use fallback attention implementation"
                )
        if XFORMERS_AVAILABLE:
            return xformers_attn(q, k, v, attn_mask=attn_mask, scale=scale)
        if SDPA_AVAILABLE:
            return sdpa_attn(q, k, v, attn_mask=attn_mask, scale=scale)
        if FLASH_ATTN_2_AVAILABLE:
            return flash_attn2(q, k, v, softmax_scale=scale)
        return eager_attn(q, k, v, attn_mask=attn_mask, scale=scale)
    else:
        if attn_impl == "eager":
            return eager_attn(q, k, v, attn_mask=attn_mask, scale=scale)
        if attn_impl == "fa3" or attn_impl == "fa3_fp8":
            if not flash_attn3_compatible:
                raise RuntimeError(
                    f"head_dim={q.shape[-1]}, but flash_attn_3 only supports head dimension at most {FA3_MAX_HEADDIM}"
                )
            if attn_mask is not None:
                raise RuntimeError("flash_attn_3 does not support attention mask")
            if attn_impl == "fa3":
                return flash_attn3(q, k, v, softmax_scale=scale)
            else:
                origin_dtype = q.dtype
                q = q.to(dtype=DTYPE_FP8)
                k = k.to(dtype=DTYPE_FP8)
                v = v.to(dtype=DTYPE_FP8)
                out = flash_attn3(q, k, v, softmax_scale=scale)
                return out.to(dtype=origin_dtype)
        if attn_impl == "aiter" or attn_impl == "aiter_fp8":
            if not flash_attn3_compatible:
                raise RuntimeError(
                    f"head_dim={q.shape[-1]}, but aiter_flash_attn only supports head dimension at most {FA3_MAX_HEADDIM}"
                )
            if attn_mask is not None:
                raise RuntimeError("aiter_flash_attn does not support attention mask")
            if attn_impl == "aiter":
                return aiter_flash_attn(q, k, v, softmax_scale=scale)
            else:
                origin_dtype = q.dtype
                q = q.to(dtype=DTYPE_FP8)
                k = k.to(dtype=DTYPE_FP8)
                v = v.to(dtype=DTYPE_FP8)
                out = aiter_flash_attn_fp8(q, k, v, softmax_scale=scale)
                return out.to(dtype=origin_dtype)
        if attn_impl == "fa4":
            if not flash_attn3_compatible:
                raise RuntimeError(
                    f"head_dim={q.shape[-1]}, but flash_attn_4 only supports head dimension at most {FA3_MAX_HEADDIM}"
                )
            if attn_mask is not None:
                raise RuntimeError("flash_attn_4 does not support attention mask")
            attn_out = flash_attn4(q, k, v, softmax_scale=scale)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            return attn_out
        if attn_impl == "fa2":
            return flash_attn2(q, k, v, softmax_scale=scale)
        if attn_impl == "xformers":
            return xformers_attn(q, k, v, attn_mask=attn_mask, scale=scale)
        if attn_impl == "sdpa":
            return sdpa_attn(q, k, v, attn_mask=attn_mask, scale=scale)
        if attn_impl == "sage":
            return sage_attn(q, k, v, attn_mask=attn_mask, scale=scale)
        if attn_impl == "sparge":
            return sparge_attn(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=scale,
                smooth_k=kwargs.get("smooth_k", True),
                simthreshd1=kwargs.get("simthreshd1", 0.6),
                cdfthreshd=kwargs.get("cdfthreshd", 0.98),
                pvthreshd=kwargs.get("pvthreshd", 50),
            )
        if attn_impl == "vsa":
            return video_sparse_attn(
                q,
                k,
                v,
                g,
                sparsity=kwargs.get("sparsity"),
                num_tiles=kwargs.get("num_tiles"),
                total_seq_length=kwargs.get("total_seq_length"),
                tile_partition_indices=kwargs.get("tile_partition_indices"),
                reverse_tile_partition_indices=kwargs.get("reverse_tile_partition_indices"),
                variable_block_sizes=kwargs.get("variable_block_sizes"),
                non_pad_index=kwargs.get("non_pad_index"),
            )
        raise ValueError(f"Invalid attention implementation: {attn_impl}")


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
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if y is None:
            y = x
        q = rearrange(self.to_q(x), "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(self.to_k(y), "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(self.to_v(y), "b s (n d) -> b s n d", n=self.num_heads)
        out = attention(q, k, v, attn_mask=attn_mask, attn_impl=self.attn_impl, scale=self.scale)
        out = rearrange(out, "b s n d -> b s (n d)", n=self.num_heads)
        return self.to_out(out)


def npu_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Ulysses sequence-parallel attention on Ascend NPU.

    all_to_all (gather sequence, scatter heads) -> fused attention -> all_to_all (inverse).
    q/k/v: [B, L_local, N, D], returns [B, L_local, N, D].

    Uses torch_npu.npu_fusion_attention, which is the same kernel that SDPA
    dispatches to on NPU, so results match single-device inference exactly.
    """
    import torch_npu
    from yunchang.comm.all_to_all import all_to_all_4D

    from diffsynth_engine.utils.process_group import (
        get_sp_ulysses_group,
        get_sp_ulysses_world_size,
    )

    if scale is None:
        scale = q.shape[-1] ** -0.5

    sp = get_sp_ulysses_world_size()
    if sp <= 1:
        return torch_npu.npu_fusion_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            head_num=q.shape[2],
            input_layout="BSND",
            scale=scale,
            keep_prob=1.0,
        )[0]

    group = get_sp_ulysses_group()
    q = all_to_all_4D(q.contiguous(), scatter_idx=2, gather_idx=1, group=group)
    k = all_to_all_4D(k.contiguous(), scatter_idx=2, gather_idx=1, group=group)
    v = all_to_all_4D(v.contiguous(), scatter_idx=2, gather_idx=1, group=group)

    # all_to_all_4D returns a transposed view; the fused kernel needs contiguous memory
    out = torch_npu.npu_fusion_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        head_num=q.shape[2],
        input_layout="BSND",
        scale=scale,
        keep_prob=1.0,
    )[0]
    return all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=group)


def npu_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Ring attention on Ascend NPU: peer-to-peer KV rotation + online softmax.

    Each rank holds one contiguous chunk of the sequence. K/V are rotated around the
    ring one step at a time, every block is scored with the fused kernel and merged
    through ``update_out_and_lse``, so each query still attends to the full context.
    Communication per rank is independent of world size, and the next block transfer
    overlaps with the current block's compute.

    yunchang ships an NPU-specific ring implementation, but it gates the compute on
    ``if step <= comm.rank`` (that is the causal branch, which must be guarded by
    ``not is_causal``) and never merges the per-block softmax statistics, so it
    effectively returns the local chunk only. This function reuses yunchang's
    RingComm and update_out_and_lse, and re-implements the loop.
    """
    import torch_npu
    from yunchang.ring.ring_npu_flash_attn import npu_fused_attn_forward
    from yunchang.ring.ring_pytorch_attn import update_out_and_lse
    from yunchang.ring.utils import RingComm

    from diffsynth_engine.utils.process_group import get_sp_group

    if scale is None:
        scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    head_num = q.shape[2]

    comm = RingComm(get_sp_group())
    if comm.world_size == 1:
        return npu_fused_attn_forward(q, k, v, head_num, "BSND", scale)[0]

    out, lse = None, None
    next_k = next_v = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        block_out, block_max, block_sum, _ = npu_fused_attn_forward(q, k, v, head_num, "BSND", scale)
        # the trailing dim of the softmax statistics is alignment padding repeating the
        # same value, so take index 0 instead of summing over it
        block_lse = torch.log(block_sum[..., 0].float().clamp(min=1e-30)) + block_max[..., 0].float()
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    return out.to(q.dtype)


# Ascend NPU: two sequence-parallel attention schemes for ring_degree > 1.
#
#   "gather" (default): all_gather the full K/V to every rank and score it in one
#       fused-attention call. Matches single-device numerics most closely because the
#       full context is scored in one shot, at the cost of O(world_size * seq) traffic.
#   "ring": rotate K/V around the ring and merge per-block softmax statistics. Traffic
#       per rank is independent of world size and transfers overlap with compute, but
#       every block is rounded to bf16 before merging, which measurably lowers output
#       fidelity on a 40-step diffusion pipeline (0.94189 vs 0.95069 SSIM against the
#       Qwen-Image reference, measured on 2 x Ascend950PR).
NPU_RING_IMPL = os.environ.get("NPU_RING_IMPL", "gather").lower()


def npu_sp_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Sequence-parallel attention for ring_degree > 1: gather full K/V, score once.

    Q stays local (length L_local), K/V are gathered to the full length, so the
    result is mathematically identical to running without sequence parallelism.
    """
    import torch.distributed as dist

    import torch_npu

    from diffsynth_engine.utils.process_group import get_sp_group, get_sp_world_size

    if scale is None:
        scale = q.shape[-1] ** -0.5

    world_size = get_sp_world_size()
    if world_size <= 1:
        return npu_ulysses_attention(q, k, v, scale=scale)

    group = get_sp_group()

    def _gather_seq(t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        parts = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(parts, t, group=group)
        return torch.cat(parts, dim=1)

    k_full = _gather_seq(k)
    v_full = _gather_seq(v)
    return torch_npu.npu_fusion_attention(
        q.contiguous(),
        k_full,
        v_full,
        head_num=q.shape[2],
        input_layout="BSND",
        scale=scale,
        keep_prob=1.0,
    )[0]


def long_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    attn_impl: Optional[str] = None,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    **kwargs,
):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    from yunchang import LongContextAttention
    from yunchang.kernels import AttnType

    assert attn_impl in [
        None,
        "auto",
        "fa2",
        "fa3",
        "fa3_fp8",
        "aiter",
        "aiter_fp8",
        "sdpa",
        "sage",
        "sparge",
        "vsa",
    ]
    assert attn_mask is None, "long context attention does not support attention mask"

    if IS_NPU and attn_impl in (None, "auto"):
        # torch_npu does not implement SDPA's mem-efficient backend, which
        # AttnType.TORCH_EFFICIENT requests; it would silently fall back to CPU and
        # leave the output there, breaking the HCCL collectives that follow.
        # Use the fused Ascend kernel instead. An explicitly requested attn_impl is
        # left alone so callers keep control.
        from diffsynth_engine.utils.process_group import get_sp_ring_world_size

        if get_sp_ring_world_size() > 1:
            if NPU_RING_IMPL == "ring":
                return npu_ring_attention(q, k, v, scale=scale)
            return npu_sp_gather_attention(q, k, v, scale=scale)
        return npu_ulysses_attention(q, k, v, scale=scale)

    flash_attn3_compatible = q.shape[-1] <= FA3_MAX_HEADDIM
    if attn_impl is None or attn_impl == "auto":
        if FLASH_ATTN_3_AVAILABLE:
            if flash_attn3_compatible:
                return LongContextAttention(attn_type=AttnType.FA3)(q, k, v, softmax_scale=scale)
            else:
                logger.warning(
                    f"head_dim={q.shape[-1]}, but flash_attn_3 only supports head dimension at most {FA3_MAX_HEADDIM}, will use fallback attention implementation"
                )
        if AITER_AVAILABLE:
            if flash_attn3_compatible:
                return LongContextAttention(attn_type=AttnType.AITER)(q, k, v, softmax_scale=scale)
            else:
                logger.warning(
                    f"head_dim={q.shape[-1]}, but aiter_flash_attn only supports head dimension at most {FA3_MAX_HEADDIM}, will use fallback attention implementation"
                )
        if SDPA_AVAILABLE:
            return LongContextAttention(attn_type=AttnType.TORCH_EFFICIENT)(q, k, v, softmax_scale=scale)
        if FLASH_ATTN_2_AVAILABLE:
            return LongContextAttention(attn_type=AttnType.FA)(q, k, v, softmax_scale=scale)
        raise ValueError("No available long context attention implementation")
    else:
        if attn_impl == "fa3" or attn_impl == "fa3_fp8":
            if not flash_attn3_compatible:
                raise RuntimeError(
                    f"head_dim={q.shape[-1]}, but flash_attn_3 only supports head dimension at most {FA3_MAX_HEADDIM}"
                )
            if attn_impl == "fa3":
                return LongContextAttention(attn_type=AttnType.FA3)(q, k, v, softmax_scale=scale)

            origin_dtype = q.dtype
            q = q.to(dtype=DTYPE_FP8)
            k = k.to(dtype=DTYPE_FP8)
            v = v.to(dtype=DTYPE_FP8)
            out = LongContextAttention(attn_type=AttnType.FA3)(q, k, v, softmax_scale=scale)
            return out.to(dtype=origin_dtype)
        if attn_impl == "aiter" or attn_impl == "aiter_fp8":
            if not flash_attn3_compatible:
                raise RuntimeError(
                    f"head_dim={q.shape[-1]}, but aiter_flash_attn only supports head dimension at most {FA3_MAX_HEADDIM}"
                )
            if attn_impl == "aiter":
                return LongContextAttention(attn_type=AttnType.AITER)(q, k, v, softmax_scale=scale)

            origin_dtype = q.dtype
            q = q.to(dtype=DTYPE_FP8)
            k = k.to(dtype=DTYPE_FP8)
            v = v.to(dtype=DTYPE_FP8)
            out = LongContextAttention(attn_type=AttnType.AITER)(q, k, v, softmax_scale=scale)
            return out.to(dtype=origin_dtype)
        if attn_impl == "fa2":
            return LongContextAttention(attn_type=AttnType.FA)(q, k, v, softmax_scale=scale)
        if attn_impl == "sdpa":
            return LongContextAttention(attn_type=AttnType.TORCH_EFFICIENT)(q, k, v, softmax_scale=scale)
        if attn_impl == "sage":
            return LongContextAttention(attn_type=AttnType.SAGE_AUTO)(q, k, v, softmax_scale=scale)
        if attn_impl == "sparge":
            attn_processor = SparseAttentionMeansim()
            # default args from spas_sage2_attn_meansim_cuda
            attn_processor.smooth_k = torch.tensor(kwargs.get("smooth_k", True))
            attn_processor.simthreshd1 = torch.tensor(kwargs.get("simthreshd1", 0.6))
            attn_processor.cdfthreshd = torch.tensor(kwargs.get("cdfthreshd", 0.98))
            attn_processor.pvthreshd = torch.tensor(kwargs.get("pvthreshd", 50))
            return LongContextAttention(attn_type=AttnType.SPARSE_SAGE, attn_processor=attn_processor)(
                q, k, v, softmax_scale=scale
            )
        if attn_impl == "vsa":
            return distributed_video_sparse_attn(
                q,
                k,
                v,
                g,
                sparsity=kwargs.get("sparsity"),
                num_tiles=kwargs.get("num_tiles"),
                total_seq_length=kwargs.get("total_seq_length"),
                tile_partition_indices=kwargs.get("tile_partition_indices"),
                reverse_tile_partition_indices=kwargs.get("reverse_tile_partition_indices"),
                variable_block_sizes=kwargs.get("variable_block_sizes"),
                non_pad_index=kwargs.get("non_pad_index"),
            )
        raise ValueError(f"Invalid long context attention implementation: {attn_impl}")
