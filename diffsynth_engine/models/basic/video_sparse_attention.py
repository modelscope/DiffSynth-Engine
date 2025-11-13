import torch
import math
import functools

from diffsynth_engine.utils.flag import VIDEO_SPARSE_ATTN_AVAILABLE
from diffsynth_engine.utils.process_group import get_sp_ulysses_group, get_sp_ring_world_size


vsa_core = None
if VIDEO_SPARSE_ATTN_AVAILABLE:
    try:
        from vsa import video_sparse_attn as vsa_core
    except Exception:
        vsa_core = None

VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(
                    indices[
                        t * ts : min(t * ts + ts, T), h * hs : min(h * hs + hs, H), w * ws : min(w * ws + ws, W)
                    ].flatten()
                )
    index = torch.cat(ls, dim=0)
    return index


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """
    Compute the number of valid (non-padded) tokens inside every
    (ts_t x ts_h x ts_w) tile after padding -- flattened in the order
    (t-tile, h-tile, w-tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = VSA_TILE_SIZE
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Vector with the size of each tile along one dimension."""
        sizes = torch.full((n_tiles,), tile, dtype=torch.int, device=device)
        # size of last (possibly partial) tile
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # broadcast‑multiply to get voxels per tile, then flatten
    block_sizes = (
        t_sizes[:, None, None]  # [n_t, 1,   1]
        * h_sizes[None, :, None]  # [1,   n_h, 1]
        * w_sizes[None, None, :]  # [1,   1,   n_w]
    ).reshape(-1)  # [n_t * n_h * n_w]

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def get_vsa_kwargs(
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    sparsity: float,
    device: torch.device,
):
    dit_seq_shape = (
        latent_shape[0] // patch_size[0],
        latent_shape[1] // patch_size[1],
        latent_shape[2] // patch_size[2],
    )

    num_tiles = (
        math.ceil(dit_seq_shape[0] / VSA_TILE_SIZE[0]),
        math.ceil(dit_seq_shape[1] / VSA_TILE_SIZE[1]),
        math.ceil(dit_seq_shape[2] / VSA_TILE_SIZE[2]),
    )
    total_seq_length = math.prod(dit_seq_shape)

    tile_partition_indices = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    reverse_tile_partition_indices = get_reverse_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    variable_block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
    non_pad_index = get_non_pad_index(variable_block_sizes, math.prod(VSA_TILE_SIZE))

    return {
        "sparsity": sparsity,
        "num_tiles": num_tiles,
        "total_seq_length": total_seq_length,
        "tile_partition_indices": tile_partition_indices,
        "reverse_tile_partition_indices": reverse_tile_partition_indices,
        "variable_block_sizes": variable_block_sizes,
        "non_pad_index": non_pad_index,
    }


def tile(
    x: torch.Tensor,
    num_tiles: tuple[int, int, int],
    tile_partition_indices: torch.LongTensor,
    non_pad_index: torch.LongTensor,
) -> torch.Tensor:
    t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
    h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
    w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]

    x_padded = torch.zeros(
        (x.shape[0], t_padded_size * h_padded_size * w_padded_size, x.shape[-2], x.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    x_padded[:, non_pad_index] = x[:, tile_partition_indices]
    return x_padded


def untile(
    x: torch.Tensor, reverse_tile_partition_indices: torch.LongTensor, non_pad_index: torch.LongTensor
) -> torch.Tensor:
    x = x[:, non_pad_index][:, reverse_tile_partition_indices]
    return x


def video_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    sparsity: float,
    num_tiles: tuple[int, int, int],
    total_seq_length: int,
    tile_partition_indices: torch.LongTensor,
    reverse_tile_partition_indices: torch.LongTensor,
    variable_block_sizes: torch.LongTensor,
    non_pad_index: torch.LongTensor,
):
    if vsa_core is None:
        raise RuntimeError(
            "Video sparse attention (VSA) is not available. "
            "Please install the 'vsa' package and ensure all its dependencies (including pytest) are installed."
        )

    q = tile(q, num_tiles, tile_partition_indices, non_pad_index)
    k = tile(k, num_tiles, tile_partition_indices, non_pad_index)
    v = tile(v, num_tiles, tile_partition_indices, non_pad_index)
    g = tile(g, num_tiles, tile_partition_indices, non_pad_index)

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()

    topk = math.ceil((1 - sparsity) * (total_seq_length / math.prod(VSA_TILE_SIZE)))
    out = vsa_core(
        q,
        k,
        v,
        variable_block_sizes=variable_block_sizes,
        topk=topk,
        block_size=VSA_TILE_SIZE,
        compress_attn_weight=g,
    ).transpose(1, 2)
    out = untile(out, reverse_tile_partition_indices, non_pad_index)
    return out


def distributed_video_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    sparsity: float,
    num_tiles: tuple[int, int, int],
    total_seq_length: int,
    tile_partition_indices: torch.LongTensor,
    reverse_tile_partition_indices: torch.LongTensor,
    variable_block_sizes: torch.LongTensor,
    non_pad_index: torch.LongTensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
):
    from yunchang.comm.all_to_all import SeqAllToAll4D

    ring_world_size = get_sp_ring_world_size()
    assert ring_world_size == 1, "distributed video sparse attention requires ring degree to be 1"
    sp_ulysses_group = get_sp_ulysses_group()

    q = SeqAllToAll4D.apply(sp_ulysses_group, q, scatter_idx, gather_idx)
    k = SeqAllToAll4D.apply(sp_ulysses_group, k, scatter_idx, gather_idx)
    v = SeqAllToAll4D.apply(sp_ulysses_group, v, scatter_idx, gather_idx)
    g = SeqAllToAll4D.apply(sp_ulysses_group, g, scatter_idx, gather_idx)

    out = video_sparse_attn(
        q,
        k,
        v,
        g,
        sparsity,
        num_tiles,
        total_seq_length,
        tile_partition_indices,
        reverse_tile_partition_indices,
        variable_block_sizes,
        non_pad_index,
    )

    out = SeqAllToAll4D.apply(sp_ulysses_group, out, gather_idx, scatter_idx)
    return out
