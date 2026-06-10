import torch
import torch.nn.functional as F

from diffsynth_engine.distributed.comm import RingComm
from diffsynth_engine.distributed.parallel_state import get_sp_group
from diffsynth_engine.layers.attention.backends.abstract import AttentionImpl


def ring_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: AttentionImpl,
    **kwargs,
) -> torch.Tensor:
    """Ring Flash Attention: each rank attends its local Q to all K/V shards across
    the ring, overlapping the next step's K/V transfer with the current attention compute.

    Args:
        query: [B, S/P, H, D] local query shard.
        key:   [B, S/P, H, D] local key shard.
        value: [B, S/P, H, D] local value shard.
        attn_impl: AttentionImpl backend (e.g. FlashAttention2Impl).

    Returns:
        [B, S/P, H, D] attention output for the local query shard.
    """
    sp_group = get_sp_group()
    comm = RingComm(sp_group.ring_group)
    world_size = comm.world_size

    out = None
    lse = None

    key = key.contiguous()
    value = value.contiguous()

    for step in range(world_size):
        if step + 1 < world_size:
            next_key = comm.send_recv(key)
            next_value = comm.send_recv(value)
            comm.commit()

        block_out, block_lse = attn_impl.forward_with_lse(query, key, value, **kwargs)
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 < world_size:
            comm.wait()
            key = next_key
            value = next_value

    return out.to(query.dtype)


def _update_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    out_block: torch.Tensor,
    lse_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid-form merge for out and lse.
    Reference: https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795

    Args:
        out:       [B, S_q, H, D] running attention output of Q over all K/V shards merged so far,
                   or None on the first step.
        lse:       [B, S_q, H, 1] running log-sum-exp of attention scores over those same shards,
                   in broadcast layout. None on the first step.
        out_block: [B, S_q, H, D] attention output of Q over the current K/V shard only.
        lse_block: [B, H, S_q] log-sum-exp of attention scores over the current K/V shard only
                   (native attn impl layout, transposed inside).

    Returns:
        out: [B, S_q, H, D] new running attention output, now including the current shard.
        lse: [B, S_q, H, 1] new running log-sum-exp, now including the current shard.
    """
    lse_block = lse_block.transpose(1, 2).unsqueeze(-1)
    if out is None:
        return out_block.float(), lse_block

    out_block = out_block.float()
    out = out - F.sigmoid(lse_block - lse) * (out - out_block)
    lse = lse - F.logsigmoid(lse - lse_block)
    return out, lse
