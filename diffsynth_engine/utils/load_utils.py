import json
import os
import re
import time
from typing import Any, Dict, Optional

import torch

from diffsynth_engine.distributed.parallel_state import get_global_rank, get_world_group, is_world_group_initialized
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


def load_model_weights(
    model_path: str,
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    broadcast_from_rank0: bool = True,
) -> Dict[str, Any]:
    world_group = get_world_group() if is_world_group_initialized() else None
    is_rank_zero = world_group is None or get_global_rank() == 0

    # rank 0 reads all shards
    state_dict = {}
    if is_rank_zero:
        if subfolder is not None:
            model_path = os.path.join(model_path, subfolder)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        _diffusion_index_file = os.path.join(model_path, DIFFUSION_SAFETENSORS_INDEX_NAME)
        _diffusion_weights_file = os.path.join(model_path, DIFFUSION_SAFETENSORS_WEIGHTS_NAME)
        _index_file = os.path.join(model_path, SAFETENSORS_INDEX_NAME)
        _weights_file = os.path.join(model_path, SAFETENSORS_WEIGHTS_NAME)

        index_file, weights_file = None, None

        if os.path.exists(_diffusion_index_file):
            index_file = _diffusion_index_file
        elif os.path.exists(_diffusion_weights_file):
            weights_file = _diffusion_weights_file
        elif os.path.exists(_index_file):
            index_file = _index_file
        elif os.path.exists(_weights_file):
            weights_file = _weights_file
        else:
            raise FileNotFoundError(f"Safetensors index or weights file not found in {model_path}")

        if index_file is not None:
            with open(index_file, "r", encoding="utf-8") as f:
                index_dict = json.load(f)
            weight_map = index_dict["weight_map"]
            shard_files = sorted(set(weight_map.values()))
            for shard_file in shard_files:
                path = os.path.join(model_path, shard_file)
                shard_dict = load_safetensors(path)
                for k, v in shard_dict.items():
                    shard_dict[k] = v.to(device=device, dtype=dtype, non_blocking=True)
                state_dict.update(shard_dict)
        else:
            state_dict = load_safetensors(weights_file)
            for k, v in state_dict.items():
                state_dict[k] = v.to(device=device, dtype=dtype, non_blocking=True)

    # rank 0 broadcasts full state dict to all other ranks
    if broadcast_from_rank0 and world_group is not None:
        state_dict = world_group.broadcast_tensor_dict(state_dict, src=0)

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
