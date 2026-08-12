import math
from typing import List

import torch
import torch.nn.functional as F

from diffsynth_engine.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    model_parallel_is_initialized,
)


def sequence_parallel_shard(tensors: List[torch.Tensor], seq_dims: List[int]) -> List[torch.Tensor]:
    if not model_parallel_is_initialized() or get_sequence_parallel_world_size() == 1:
        return tensors

    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"

    shard_tensors = []
    for tensor, seq_dim in zip(tensors, seq_dims):
        if tensor is None:
            shard_tensors.append(tensor)
            continue
        seq_len = tensor.size(seq_dim)
        shard_len = math.ceil(seq_len / sp_world_size)
        start = min(shard_len * sp_rank, seq_len)
        length = min(shard_len, seq_len - start)
        chunk = tensor.narrow(seq_dim, start, length)

        if length < shard_len:
            # pad only the local shard instead of the full sequence, to cut the memory peak
            padding = [0] * (2 * tensor.ndim)
            padding[-2 * seq_dim - 1] = shard_len - length
            chunk = F.pad(chunk, padding)
        else:
            # clone so the view does not keep the full sequence storage alive
            chunk = chunk.clone()
        shard_tensors.append(chunk)
    return shard_tensors


def sequence_parallel_unshard(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    seq_lens: List[int],
) -> List[torch.Tensor]:
    if not model_parallel_is_initialized() or get_sequence_parallel_world_size() == 1:
        return tensors

    sp_group = get_sp_group()

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"
    assert len(tensors) == len(seq_lens), "tensors and seq_lens must have the same number of elements"

    unshard_tensors = []
    for tensor, seq_dim, seq_len in zip(tensors, seq_dims, seq_lens):
        tensor = tensor.contiguous()
        unshard = sp_group.all_gather(tensor, dim=seq_dim)
        unshard = unshard.narrow(dim=seq_dim, start=0, length=seq_len)
        unshard_tensors.append(unshard)
    return unshard_tensors
