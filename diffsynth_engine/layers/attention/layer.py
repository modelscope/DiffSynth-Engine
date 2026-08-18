# Adapted from https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from torch import distributed as dist
import torch
import torch.nn as nn

from diffsynth_engine.distributed.comm import SeqAllToAll4D
from diffsynth_engine.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_group,
    get_ulysses_parallel_world_size,
    is_sp_group_initialized,
)
from diffsynth_engine.forward_context import ForwardContext, get_forward_context
from diffsynth_engine.layers.attention.ring import ring_flash_attention_forward
from diffsynth_engine.registry import get_attn_backend


class LocalAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        attn_type: str | None = None,
        **extra_impl_args,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads

        attn_backend = get_attn_backend(attn_type)
        if not attn_backend.supports_head_size(head_size):
            raise ValueError(f"Attention backend {attn_type!r} does not support head size {head_size}.")

        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            **extra_impl_args,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]

        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        attn_kwargs = {"attn_metadata": attn_metadata}
        attn_kwargs.update(kwargs)

        output = self.attn_impl.forward(q, k, v, **attn_kwargs)
        return output


class USPAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        attn_type: str | None = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        **extra_impl_args,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.attn_type = str(attn_type) if attn_type is not None else None

        attn_backend = get_attn_backend(attn_type)
        if not attn_backend.supports_head_size(head_size):
            raise ValueError(f"Attention backend {attn_type!r} does not support head size {head_size}.")

        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            **extra_impl_args,
        )

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]

        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        attn_kwargs = {"attn_metadata": attn_metadata}
        attn_kwargs.update(kwargs)

        ulysses_parallel_world_size = get_ulysses_parallel_world_size() if is_sp_group_initialized() else 1
        ring_parallel_world_size = get_ring_parallel_world_size() if is_sp_group_initialized() else 1

        if ring_parallel_world_size > 1 and self.attn_type == "mindie":
            raise RuntimeError(
                "NPU MindIE attention currently supports Ulysses only "
                f"(sp_ring_degree must be 1, got {ring_parallel_world_size})"
            )

        if ulysses_parallel_world_size > 1:
            q = SeqAllToAll4D.apply(get_sp_group().ulysses_group, q, self.scatter_idx, self.gather_idx)
            k = SeqAllToAll4D.apply(get_sp_group().ulysses_group, k, self.scatter_idx, self.gather_idx)
            v = SeqAllToAll4D.apply(get_sp_group().ulysses_group, v, self.scatter_idx, self.gather_idx)

        if ring_parallel_world_size > 1:
            # warning: attn_kwargs is not supported for ring flash attention
            output = ring_flash_attention_forward(q, k, v, self.attn_impl, **attn_kwargs)
        else:
            output = self.attn_impl.forward(q, k, v, **attn_kwargs)

        if ulysses_parallel_world_size > 1:
            output = SeqAllToAll4D.apply(get_sp_group().ulysses_group, output, self.gather_idx, self.scatter_idx)
        return output

        
from typing import Optional
class AscendLongContextAttention(nn.Module):
    # Single dedicated communication stream shared by all instances (one per transformer
    # block, e.g. 60 in Qwen-Image), so only one `stream2` is allocated per device.
    _shared_comm_stream = None

    def __init__(
            self,
            num_heads: int = 24,
            head_size: int = 128,
            softmax_scale: float | None = None,
            causal: bool = False,
            num_kv_heads: int | None = None,
            attn_type: str = "mindie",
            scatter_idx: int = 2,
            gather_idx: int = 1,
            fa_head_loop: int | None = None,
            **extra_impl_args,
    ) -> None:
        super().__init__()
        from diffsynth_engine.platforms import AscendPlatform

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads

        self.ulysses_pg = get_sp_group().ulysses_group
        self.sp_ulysses_degree = get_sp_group().ulysses_world_size
        self.sp_ring_degree = get_sp_group().ring_world_size


        self.fa_alltoall_overlap = AscendPlatform.fa_alltoall_overlap
        self.fa_alltoall_cut = AscendPlatform.fa_alltoall_cut
        if fa_head_loop is not None:
            self.fa_head_loop = fa_head_loop
        elif self.fa_alltoall_cut > 1:
            self.fa_head_loop = self.fa_alltoall_cut
        elif self.fa_alltoall_overlap > 1:
            self.fa_head_loop = self.fa_alltoall_overlap
        else:
            self.fa_head_loop = self.num_heads // self.sp_ulysses_degree

        if self.fa_alltoall_overlap > 1 and self.fa_alltoall_cut <= 1:
            if AscendLongContextAttention._shared_comm_stream is None:
                AscendLongContextAttention._shared_comm_stream = torch.npu.Stream()
            self.stream2 = AscendLongContextAttention._shared_comm_stream
            self.event = []
            for i in range(self.fa_head_loop):
                self.event.append(torch.npu.Event())

        self.attn_type = str(attn_type) if attn_type is not None else None
        attn_backend = get_attn_backend(attn_type)
        if not attn_backend.supports_head_size(head_size):
            raise ValueError(f"Attention backend {attn_type!r} does not support head size {head_size}.")

        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            **extra_impl_args,
        )

        # TODO: currunt MindIE only support Ulysses
        if self.sp_ring_degree > 1: 
            raise RuntimeError(
                "NPU MindIE attention currently supports Ulysses only "
                f"(sp_ring_degree must be 1, got {self.sp_ring_degree})"
            )


    def _run_attention(self, q, k, v, **attn_kwargs):
        return self.attn_impl.forward(q, k, v, **attn_kwargs)

    @staticmethod
    def all_to_all_4D_pre(input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None):
        assert (
                input.dim() == 4
        ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

        seq_world_size = dist.get_world_size(group)

        if scatter_idx == 2 and gather_idx == 1:
            # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
            bs, shard_seqlen, hc, hs = input.shape
            seqlen = shard_seqlen * seq_world_size
            shard_hc = hc // seq_world_size

            # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
            # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
            input_t = (
                input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
                .transpose(0, 2)
                .contiguous()
            )

            return input_t

        elif scatter_idx == 1 and gather_idx == 2:
            # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
            bs, seqlen, shard_hc, hs = input.shape
            hc = shard_hc * seq_world_size
            shard_seqlen = seqlen // seq_world_size

            # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
            # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
            input_t = (
                input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
                .transpose(0, 3)
                .transpose(0, 1)
                .contiguous()
                .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
            )

            return input_t
        else:
            raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")

    @staticmethod
    def all_to_all_4D_after(input: torch.tensor, output: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1,
                            group=None):
        seq_world_size = dist.get_world_size(group)

        if scatter_idx == 2 and gather_idx == 1:
            bs, shard_seqlen, hc, hs = input.shape

            seqlen = shard_seqlen * seq_world_size
            shard_hc = hc // seq_world_size

            output = output.reshape(seqlen, bs, shard_hc, hs)

            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
            return output
        elif scatter_idx == 1 and gather_idx == 2:
            bs, seqlen, shard_hc, hs = input.shape
            hc = shard_hc * seq_world_size
            shard_seqlen = seqlen // seq_world_size
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(hc, shard_seqlen, bs, hs)

            # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
            output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            return output
        else:
            raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


    @staticmethod
    def split_qkv_by_head(query, key, value, sp_ulysses_degree, loop_time):
        """Split Q/K/V along head dim into chunks for insertcomm / blockattn."""
        _, _, head_count, _ = query.shape
        if head_count % sp_ulysses_degree != 0:
            raise ValueError(
                f"head_count must be divisible by ulysses world size, "
                f"got head_count={head_count}, sp_ulysses_degree={sp_ulysses_degree}"
            )
        heads_per_rank = head_count // sp_ulysses_degree
        if heads_per_rank % loop_time != 0:
            raise ValueError(
                f"heads_per_rank must be divisible by loop_time={loop_time}, "
                f"got heads_per_rank={heads_per_rank}"
            )
        global_chunk_heads = heads_per_rank // loop_time * sp_ulysses_degree
        return (
            query.split(global_chunk_heads, dim=2),
            key.split(global_chunk_heads, dim=2),
            value.split(global_chunk_heads, dim=2),
        )


    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """forward

        Arguments:
            query (torch.Tensor): query input to the layer
            key (torch.Tensor): key input to the layer
            value (torch.Tensor): value input to the layer

        Returns:
            * output (torch.Tensor): context output
        """
        # Check input shapes
        assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        attn_kwargs = {"attn_metadata": attn_metadata}
        attn_kwargs.update(kwargs)
    
        output = None

        if self.fa_alltoall_cut <= 1 and self.fa_alltoall_overlap <= 1:
            # baseline: both 0 / both 1 / a single 1 all mean "not enabled" (1 chunk = no split)
            # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
            # scatter 2, gather 1
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

            out = self._run_attention(query_layer, key_layer, value_layer, **attn_kwargs)
            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, out, self.gather_idx, self.scatter_idx
            )
        elif self.fa_alltoall_cut > 1:  # fa_alltoall_cut
            # Split heads into chunks (loop_time = fa_alltoall_cut), full Ulysses round-trip per chunk.
            q_chunks, k_chunks, v_chunks = self.split_qkv_by_head(
                query, key, value, self.sp_ulysses_degree, self.fa_head_loop
            )
            output_chunks = []
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, q_chunk, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, k_chunk, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, v_chunk, self.scatter_idx, self.gather_idx
                )
                out = self._run_attention(query_layer, key_layer, value_layer, **attn_kwargs)
                out = SeqAllToAll4D.apply(
                    self.ulysses_pg, out, self.gather_idx, self.scatter_idx
                )
                output_chunks.append(out)
            output = torch.cat(output_chunks, dim=2)
        elif self.fa_alltoall_overlap > 1 :  # fa_alltoall_overlap
            # B, S/sp, N/tp, D
            # Refresh the current stream here: __init__ runs at model-build time and may capture a
            # different stream than the one forward actually executes on (e.g. under a stream context,
            # pipeline/CFG stream switching, or CUDA-graph capture). Pinning the build-time stream would
            # break event/stream synchronization in the overlap pipeline.
            self.current_stream = torch.npu.current_stream()
            query_layer_list, key_layer_list, value_layer_list = self.split_qkv_by_head(
                query, key, value, self.sp_ulysses_degree, self.fa_head_loop
            )
            for_loop = len(query_layer_list)

            # scatter 2, gather 1
            output_fa = []
            q_event = torch.npu.Event()
            k_event = torch.npu.Event()
            v_event = torch.npu.Event()
            q_lists, k_lists, v_lists, kv_lists = [], [], [], []

            for i in range(0, for_loop):
                input_q = self.all_to_all_4D_pre(query_layer_list[i], self.scatter_idx, self.gather_idx,
                                            self.ulysses_pg)
                q_event.record()
                with torch.npu.stream(self.stream2):
                    self.stream2.wait_event(q_event)
                    query_layer = torch.empty_like(input_q)
                    dist.all_to_all_single(query_layer, input_q, group=self.ulysses_pg)

                input_k = self.all_to_all_4D_pre(key_layer_list[i], self.scatter_idx, self.gather_idx,
                                            self.ulysses_pg)
                input_v = self.all_to_all_4D_pre(value_layer_list[i], self.scatter_idx, self.gather_idx,
                                            self.ulysses_pg)
                v_event.record()

                with torch.npu.stream(self.stream2):
                    self.stream2.wait_event(v_event)
                    key_layer = torch.empty_like(input_k)
                    dist.all_to_all_single(key_layer, input_k, group=self.ulysses_pg)

                    value_layer = torch.empty_like(input_v)
                    dist.all_to_all_single(value_layer, input_v, group=self.ulysses_pg)
                    k_event.record()

                q_lists.append(query_layer)
                k_lists.append(key_layer)
                v_lists.append(value_layer)

                k_lists[i] = self.all_to_all_4D_after(key_layer_list[i], k_lists[i], self.scatter_idx,
                                                 self.gather_idx, self.ulysses_pg)
                v_lists[i] = self.all_to_all_4D_after(value_layer_list[i], v_lists[i], self.scatter_idx,
                                                 self.gather_idx, self.ulysses_pg)
                q_event.record()
                with torch.npu.stream(self.stream2):
                    self.stream2.wait_event(q_event)
                    self.event[i].record()
                q_lists[i] = self.all_to_all_4D_after(query_layer_list[i], q_lists[i], self.scatter_idx, self.gather_idx, self.ulysses_pg)
                

            for i in range(0, for_loop):
                # fa
                self.current_stream.wait_event(self.event[i])

                out = self._run_attention(q_lists[i], k_lists[i], v_lists[i], **attn_kwargs)
                kv_lists.append(out)
                input_t = self.all_to_all_4D_pre(out, self.gather_idx, self.scatter_idx, self.ulysses_pg)
                q_event.record()

                with torch.npu.stream(self.stream2):
                    self.stream2.wait_event(q_event)
                    output = torch.empty_like(input_t)
                    dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
                    self.event[i].record()
                output_fa.append(output)

            for i in range(for_loop):
                self.current_stream.wait_event(self.event[i])
                output_fa[i] = self.all_to_all_4D_after(kv_lists[i], output_fa[i], self.gather_idx, self.scatter_idx, self.ulysses_pg)
            output = torch.cat(output_fa, dim=2)
        else:
            raise RuntimeError(
                f"Invalid configuration: fa_alltoall_cut={self.fa_alltoall_cut}, fa_alltoall_overlap={self.fa_alltoall_overlap}"
            )
        return output

_ASCEND_LONGCTX_ATTN: Optional[AscendLongContextAttention] = None


def _get_ascend_long_context_attn() -> AscendLongContextAttention:
    global _ASCEND_LONGCTX_ATTN
    if _ASCEND_LONGCTX_ATTN is None:
        _ASCEND_LONGCTX_ATTN = AscendLongContextAttention()
    return _ASCEND_LONGCTX_ATTN