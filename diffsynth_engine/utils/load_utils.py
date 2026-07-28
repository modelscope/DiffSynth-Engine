import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from diffsynth_engine.distributed.parallel_state import get_global_rank, get_world_group, is_world_group_initialized
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import (
    DIFFUSION_SAFETENSORS_INDEX_NAME,
    DIFFUSION_SAFETENSORS_WEIGHTS_NAME,
    SAFETENSORS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
)


@dataclass(frozen=True)
class TensorSlice:
    dim: int
    start: int
    end: int


@dataclass(frozen=True)
class TensorSelection:
    slices: Tuple[TensorSlice, ...]
    contiguous: bool = True


TensorSelectionPlan = Dict[str, TensorSelection]

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


def apply_tensor_selection(
    state_dict: Dict[str, Any],
    tensor_selection_plan: TensorSelectionPlan,
) -> Dict[str, Any]:
    for key in list(state_dict.keys()):
        selection = tensor_selection_plan.get(key)
        if selection is None:
            continue

        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            continue

        slices = [slice(None)] * value.dim()
        for tensor_slice in selection.slices:
            dim = tensor_slice.dim if tensor_slice.dim >= 0 else value.dim() + tensor_slice.dim
            if dim < 0 or dim >= value.dim():
                raise ValueError(
                    f"Cannot slice '{key}' shape={tuple(value.shape)} along dim {tensor_slice.dim}: "
                    f"tensor has {value.dim()} dimensions."
                )
            if tensor_slice.start < 0 or tensor_slice.end > value.shape[dim] or tensor_slice.start > tensor_slice.end:
                raise ValueError(
                    f"Invalid slice for '{key}' shape={tuple(value.shape)}: "
                    f"dim={tensor_slice.dim}, start={tensor_slice.start}, end={tensor_slice.end}."
                )
            slices[dim] = slice(tensor_slice.start, tensor_slice.end)

        selected = value[tuple(slices)]
        state_dict[key] = selected.contiguous() if selection.contiguous else selected

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
    shard_names = sorted(set(index_dict["weight_map"].values()))
    if not shard_names:
        raise ValueError(f"Weight index {index_file} contains an empty weight_map")
    return [os.path.join(path, name) for name in shard_names]


def load_model_weights(
    model_path: str,
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    broadcast_from_rank0: bool = True,
    tensor_selection_plan: Optional[TensorSelectionPlan] = None,
) -> Dict[str, Any]:
    world_group = get_world_group() if is_world_group_initialized() else None
    is_rank_zero = world_group is None or get_global_rank() == 0

    resolved_path = os.path.join(model_path, subfolder) if subfolder is not None else model_path
    shard_files = _list_shard_files(resolved_path)

    state_dict: Dict[str, Any] = {}
    for shard_file in shard_files:
        shard_dict = load_safetensors(shard_file) if is_rank_zero else {}

        if world_group is not None and broadcast_from_rank0:
            shard_dict = world_group.broadcast_tensor_dict(shard_dict, src=0)

        if tensor_selection_plan is not None:
            shard_dict = apply_tensor_selection(shard_dict, tensor_selection_plan)

        for key, value in shard_dict.items():
            if isinstance(value, torch.Tensor):
                shard_dict[key] = value.to(device=device, dtype=dtype, non_blocking=True)
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
