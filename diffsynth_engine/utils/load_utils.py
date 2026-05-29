import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

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


def _get_cgroup_memory_limit_bytes() -> Optional[int]:
    for cgroup_path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        if not os.path.exists(cgroup_path):
            continue
        with open(cgroup_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if raw in ("max", ""):
            continue
        limit = int(raw)
        # cgroup v2 uses "max" encoded as a very large number on some systems
        if limit > (1 << 62):
            continue
        return limit
    return None


def _estimate_checkpoint_bytes(weight_paths: List[str]) -> int:
    return sum(os.path.getsize(path) for path in weight_paths)


def _should_stream_load(weight_paths: List[str]) -> bool:
    mode = os.environ.get("LOAD_WEIGHTS_STREAMING", "auto").lower()
    if mode in ("1", "true", "yes"):
        return True
    if mode in ("0", "false", "no"):
        return False

    cgroup_limit = _get_cgroup_memory_limit_bytes()
    if cgroup_limit is None:
        return False

    checkpoint_bytes = _estimate_checkpoint_bytes(weight_paths)
    # Host needs headroom beyond raw checkpoint bytes for Python and copy peaks.
    safety_factor = float(os.environ.get("LOAD_WEIGHTS_STREAMING_SAFETY", "1.15"))
    return checkpoint_bytes * safety_factor > cgroup_limit


def load_safetensors(
    path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    start_time = time.perf_counter()
    if FAST_SAFETENSORS_AVAILABLE:
        if is_rank_zero:
            logger.info(f"FastSafetensors loading model from {path}...")
        num_threads = int(os.environ.get("FAST_SAFETENSORS_NUM_THREADS", 16))
        direct_io = os.environ.get("FAST_SAFETENSORS_DIRECT_IO", "False").upper() == "TRUE"
        state_dict = load_file(path, num_threads=num_threads, direct_io=direct_io)
        if dtype is None:
            state_dict = {k: v.to(device=device, non_blocking=True) for k, v in state_dict.items()}
        else:
            state_dict = {
                k: v.to(device=device, dtype=dtype, non_blocking=True) for k, v in state_dict.items()
            }
    else:
        if is_rank_zero:
            logger.info(f"Safetensors loading model from {path}...")
        state_dict = load_file(path, device=device)
        if dtype is not None:
            state_dict = {k: v.to(dtype=dtype, non_blocking=True) for k, v in state_dict.items()}
    elapsed_time = (time.perf_counter() - start_time) * 1000
    if is_rank_zero:
        logger.info(f"Model loaded in {elapsed_time} ms")
    return state_dict


def _resolve_weight_paths(model_path: str) -> List[str]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    _diffusion_index_file = os.path.join(model_path, DIFFUSION_SAFETENSORS_INDEX_NAME)
    _diffusion_weights_file = os.path.join(model_path, DIFFUSION_SAFETENSORS_WEIGHTS_NAME)
    _index_file = os.path.join(model_path, SAFETENSORS_INDEX_NAME)
    _weights_file = os.path.join(model_path, SAFETENSORS_WEIGHTS_NAME)

    if os.path.exists(_diffusion_index_file):
        index_file = _diffusion_index_file
    elif os.path.exists(_index_file):
        index_file = _index_file
    elif os.path.exists(_diffusion_weights_file):
        return [_diffusion_weights_file]
    elif os.path.exists(_weights_file):
        return [_weights_file]
    else:
        raise FileNotFoundError(f"Safetensors index or weights file not found in {model_path}")

    with open(index_file, "r", encoding="utf-8") as f:
        index_dict = json.load(f)
    weight_map = index_dict["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    return [os.path.join(model_path, shard_file) for shard_file in shard_files]


def _move_tensors(
    state_dict: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    if device is None and dtype is None:
        return state_dict
    return {k: v.to(device=device, dtype=dtype, non_blocking=True) for k, v in state_dict.items()}


def _load_weights_streaming(
    module: nn.Module,
    weight_paths: List[str],
    device: Optional[Union[str, torch.device]],
    dtype: Optional[torch.dtype],
    key_mapping: Optional[Dict[str, str]],
) -> None:
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    expected_keys = set(module.state_dict().keys())
    loaded_keys: set[str] = set()
    total_params = 0
    total_size_bytes = 0
    load_device = device if device is not None else "cpu"

    for shard_idx, shard_path in enumerate(weight_paths, start=1):
        if is_rank_zero:
            start_time = time.perf_counter()

        shard_dict = load_safetensors(shard_path, device=load_device, dtype=dtype)
        if key_mapping:
            shard_dict = fix_state_dict_key(shard_dict, key_mapping)

        module.load_state_dict(shard_dict, strict=False, assign=True)
        loaded_keys.update(shard_dict.keys())
        total_params += sum(v.numel() for v in shard_dict.values())
        total_size_bytes += sum(v.numel() * v.element_size() for v in shard_dict.values())

        if is_rank_zero:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"Shard {shard_idx}/{len(weight_paths)} assigned in {elapsed:.2f} ms")

        del shard_dict

    missing_keys = expected_keys - loaded_keys
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing_keys)} parameter(s) required by the model, "
            f"e.g. {sorted(missing_keys)[:5]}"
        )

    if is_rank_zero:
        total_size_gb = total_size_bytes / (1024**3)
        logger.info(
            f"Loaded {total_params:,} parameters ({total_size_gb:.2f} GB) "
            f"from {len(weight_paths)} shard(s) into module (streaming)"
        )


def load_weights_into_module(
    module: nn.Module,
    model_path: str,
    subfolder: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    key_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """Load checkpoint into *module*, streaming shards only when Host RAM is tight."""
    if subfolder is not None:
        model_path = os.path.join(model_path, subfolder)

    weight_paths = _resolve_weight_paths(model_path)
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0

    if _should_stream_load(weight_paths):
        if is_rank_zero:
            checkpoint_gb = _estimate_checkpoint_bytes(weight_paths) / (1024**3)
            cgroup_limit = _get_cgroup_memory_limit_bytes()
            cgroup_gb = cgroup_limit / (1024**3) if cgroup_limit is not None else None
            logger.info(
                f"Using streaming weight load ({checkpoint_gb:.2f} GB checkpoint"
                + (f", cgroup limit {cgroup_gb:.2f} GB)" if cgroup_gb is not None else ")")
            )
        _load_weights_streaming(module, weight_paths, device, dtype, key_mapping)
        return

    if is_rank_zero:
        logger.info("Using bulk weight load")

    state_dict = load_model_weights(model_path, device=device, dtype=dtype)
    if key_mapping:
        state_dict = fix_state_dict_key(state_dict, key_mapping)
    module.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict


def load_model_weights(
    model_path: str,
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    if subfolder is not None:
        model_path = os.path.join(model_path, subfolder)

    weight_paths = _resolve_weight_paths(model_path)
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0

    if len(weight_paths) == 1:
        state_dict = load_safetensors(weight_paths[0])
    else:
        state_dict = {}
        for shard_path in weight_paths:
            state_dict.update(load_safetensors(shard_path))

    if is_rank_zero and device is not None:
        total_params = sum(v.numel() for v in state_dict.values())
        total_size_gb = sum(v.numel() * v.element_size() for v in state_dict.values()) / (1024**3)
        logger.info(f"Moving {total_params:,} parameters ({total_size_gb:.2f} GB) to {device}...")
        start_time = time.perf_counter()
    state_dict = _move_tensors(state_dict, device, dtype)
    if is_rank_zero and device is not None:
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Moved to {device} in {elapsed:.2f} ms")
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
