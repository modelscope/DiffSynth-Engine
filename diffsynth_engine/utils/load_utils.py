import json
import os
import re
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from diffsynth_engine.distributed.parallel_state import (
    get_global_rank,
    get_tensor_model_parallel_world_size,
    get_world_group,
    is_tp_group_initialized,
    is_world_group_initialized,
)
from diffsynth_engine.layers.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from diffsynth_engine.layers.tensor_parallel.norm import TensorParallelRMSNorm
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import (
    DIFFUSION_SAFETENSORS_INDEX_NAME,
    DIFFUSION_SAFETENSORS_WEIGHTS_NAME,
    SAFETENSORS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
)

logger = logging.get_logger(__name__)

try:
    from fast_safetensors import load_safetensors as load_file

    FAST_SAFETENSORS_AVAILABLE = True
except ImportError:
    from safetensors.torch import load_file

    FAST_SAFETENSORS_AVAILABLE = False


def load_safetensors(path: str, device: str = "cpu") -> Dict[str, Any]:
    is_rank_zero = not is_world_group_initialized() or get_global_rank() == 0
    start_time = time.perf_counter()
    if FAST_SAFETENSORS_AVAILABLE:
        if is_rank_zero:
            logger.info(f"FastSafetensors loading model from {path}...")
        num_threads = int(os.environ.get("FAST_SAFETENSORS_NUM_THREADS", 16))
        direct_io = os.environ.get("FAST_SAFETENSORS_DIRECT_IO", "False").upper() == "TRUE"
        state_dict = load_file(path, num_threads=num_threads, direct_io=direct_io)
        for k, v in state_dict.items():
            state_dict[k] = v.to(device=device, non_blocking=True)
    else:
        if is_rank_zero:
            logger.info(f"Safetensors loading model from {path}...")
        state_dict = load_file(path, device=device)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    if is_rank_zero:
        logger.info(f"Model loaded in {elapsed_time} ms")
    return state_dict


def _list_shard_files(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path not found: {path}")

    diffusion_index = os.path.join(path, DIFFUSION_SAFETENSORS_INDEX_NAME)
    diffusion_weights = os.path.join(path, DIFFUSION_SAFETENSORS_WEIGHTS_NAME)
    generic_index = os.path.join(path, SAFETENSORS_INDEX_NAME)
    generic_weights = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)

    if os.path.exists(diffusion_index):
        index_file = diffusion_index
    elif os.path.exists(diffusion_weights):
        return [diffusion_weights]
    elif os.path.exists(generic_index):
        index_file = generic_index
    elif os.path.exists(generic_weights):
        return [generic_weights]
    else:
        raise FileNotFoundError(f"Safetensors index or weights file not found in {path}")

    with open(index_file, "r", encoding="utf-8") as file:
        index_dict = json.load(file)
    shard_files = sorted(set(index_dict["weight_map"].values()))
    if not shard_files:
        raise ValueError(f"Weight index {index_file} contains an empty weight_map")
    return [os.path.join(path, name) for name in shard_files]


# tensor parallel


def _slice_tensor(
    tensor: torch.Tensor,
    name: str,
    span: tuple[int, int, int],
) -> torch.Tensor:
    dim, start, length = span
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Cannot shard '{name}' shape={tuple(tensor.shape)} along dim {dim}.")
    if start < 0 or length < 0 or start + length > tensor.shape[dim]:
        raise ValueError(
            f"Invalid shard for '{name}' shape={tuple(tensor.shape)}: dim={dim}, start={start}, length={length}."
        )

    return tensor.narrow(dim, start, length).contiguous()


def _slice_tensor_parallel_weights(
    model: nn.Module,
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    tp_size = get_tensor_model_parallel_world_size() if is_tp_group_initialized() else 1
    if tp_size <= 1:
        return state_dict

    spans: Dict[str, tuple[int, int, int]] = {}
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""

        if isinstance(module, ColumnParallelLinear) and module.tp_size > 1:
            start = module.tp_rank * module.out_features_per_partition
            length = module.out_features_per_partition
            spans[f"{prefix}weight"] = (0, start, length)
            if module.bias is not None:
                spans[f"{prefix}bias"] = (0, start, length)
        elif isinstance(module, RowParallelLinear) and module.tp_size > 1:
            start = module.tp_rank * module.in_features_per_partition
            length = module.in_features_per_partition
            spans[f"{prefix}weight"] = (1, start, length)
        elif isinstance(module, TensorParallelRMSNorm) and module.tp_size > 1 and module.weight is not None:
            start = module.tp_rank * module.hidden_size_per_partition
            length = module.hidden_size_per_partition
            spans[f"{prefix}weight"] = (0, start, length)

    for name, tensor in state_dict.items():
        span = spans.get(name)
        if span is not None:
            state_dict[name] = _slice_tensor(tensor, name, span)

    return state_dict


def load_model_weights(
    model_path: str,
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    broadcast_from_rank0: bool = True,
) -> Dict[str, Any]:
    world_group = get_world_group() if is_world_group_initialized() else None
    is_rank_zero = world_group is None or get_global_rank() == 0
    model_path = os.path.join(model_path, subfolder) if subfolder is not None else model_path

    state_dict: Dict[str, Any] = {}
    for shard_file in _list_shard_files(model_path):
        shard_dict = load_safetensors(shard_file, device=device or "cpu") if is_rank_zero else {}

        if world_group is not None and broadcast_from_rank0:
            shard_dict = world_group.broadcast_tensor_dict(shard_dict, src=0)

        for name, tensor in shard_dict.items():
            if isinstance(tensor, torch.Tensor):
                shard_dict[name] = tensor.to(dtype=dtype, non_blocking=True)
        state_dict.update(shard_dict)

    return state_dict


def prepare_model_weights(
    model: nn.Module,
    model_path: str,
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    broadcast_from_rank0: bool = True,
) -> Dict[str, Any]:
    world_group = get_world_group() if is_world_group_initialized() else None
    is_rank_zero = world_group is None or get_global_rank() == 0
    model_path = os.path.join(model_path, subfolder) if subfolder is not None else model_path

    state_dict: Dict[str, Any] = {}
    for shard_file in _list_shard_files(model_path):
        shard_dict = load_safetensors(shard_file, device=device or "cpu") if is_rank_zero else {}

        if world_group is not None and broadcast_from_rank0:
            shard_dict = world_group.broadcast_tensor_dict(shard_dict, src=0)

        shard_dict = _slice_tensor_parallel_weights(model, shard_dict)

        for name, tensor in shard_dict.items():
            if isinstance(tensor, torch.Tensor):
                shard_dict[name] = tensor.to(dtype=dtype, non_blocking=True)
        state_dict.update(shard_dict)

    return state_dict


def fix_state_dict_key(state_dict: Dict[str, Any], key_mapping: Dict[str, str]) -> Dict[str, Any]:
    _state_dict = {}
    for k, v in state_dict.items():
        for pattern, repl in key_mapping.items():
            k, n = re.subn(pattern, repl, k)
            if n > 0:
                break
        _state_dict[k] = v
    return _state_dict
