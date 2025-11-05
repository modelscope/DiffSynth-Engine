import torch
import torch.nn.functional as F


def accumulate(result, new_item):
    if result is None:
        return new_item
    for i, item in enumerate(new_item):
        result[i] += item
    return result


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def pad_and_concat(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    concat_dim: int = 0,
    pad_dim: int = 1,
) -> torch.Tensor:
    """
    Concatenate two tensors along a specified dimension after padding along another dimension.

    Assumes input tensors have shape (b, s, d), where:
    - b: batch dimension
    - s: sequence dimension (may differ)
    - d: feature dimension

    Args:
        tensor1: First tensor with shape (b1, s1, d)
        tensor2: Second tensor with shape (b2, s2, d)
        concat_dim: Dimension to concatenate along, default is 0 (batch dimension)
        pad_dim: Dimension to pad along, default is 1 (sequence dimension)

    Returns:
        Concatenated tensor, shape depends on concat_dim and pad_dim choices
    """
    assert tensor1.dim() == tensor2.dim(), "Both tensors must have the same number of dimensions"
    assert concat_dim != pad_dim, "concat_dim and pad_dim cannot be the same"

    len1, len2 = tensor1.shape[pad_dim], tensor2.shape[pad_dim]
    max_len = max(len1, len2)

    # Calculate the position of pad_dim in the padding list
    # Padding format: from the last dimension, each pair represents (dim_n_left, dim_n_right, ..., dim_0_left, dim_0_right)
    ndim = tensor1.dim()
    padding = [0] * (2 * ndim)
    pad_right_idx = -2 * pad_dim - 1

    if len1 < max_len:
        pad_len = max_len - len1
        padding[pad_right_idx] = pad_len
        tensor1 = F.pad(tensor1, padding, mode="constant", value=0)
    elif len2 < max_len:
        pad_len = max_len - len2
        padding[pad_right_idx] = pad_len
        tensor2 = F.pad(tensor2, padding, mode="constant", value=0)

    # Concatenate along the specified dimension
    return torch.cat([tensor1, tensor2], dim=concat_dim)
