import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from typing import Dict, List, Set, Type, Union, Optional
from queue import Empty

import diffsynth_engine.models.basic.attention as attention_ops
from diffsynth_engine.utils.platform import empty_cache
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.process_group import (
    PROCESS_GROUP,
    get_cfg_group,
    get_cfg_world_size,
    get_cfg_rank,
    get_cfg_ranks,
    get_sp_group,
    get_sp_world_size,
    get_sp_rank,
    get_sp_ranks,
    get_sp_ulysses_group,
    get_sp_ulysses_world_size,
    get_sp_ulysses_rank,
    get_sp_ulysses_ranks,
    get_sp_ring_group,
    get_sp_ring_world_size,
    get_sp_ring_rank,
    get_sp_ring_ranks,
    get_tp_group,
    get_tp_world_size,
    get_tp_rank,
    get_tp_ranks,
)

logger = logging.get_logger(__name__)


def init_parallel_pgs(
    cfg_degree: int = 1,
    sp_ulysses_degree: int = 1,
    sp_ring_degree: int = 1,
    tp_degree: int = 1,
    rank: int = 0,
    world_size: int = 1,
):
    from yunchang.globals import set_seq_parallel_pg

    sp_degree = sp_ulysses_degree * sp_ring_degree

    assert sp_degree == 1 or tp_degree == 1, "not allowed to enable sequence parallel and tensor parallel together"
    assert world_size == cfg_degree * sp_degree * tp_degree, (
        f"world_size ({world_size}) must be equal to cfg_degree ({cfg_degree}) * sp_degree ({sp_degree}) * tp_degree ({tp_degree})"
    )

    def make_parallel_groups(blocks: List[List[int]], degree: int):
        groups, chunks = [], []
        for block in blocks:
            size = len(block) // degree
            chunk = [block[i * size : (i + 1) * size] for i in range(degree)]
            chunks.extend(chunk)
            groups.extend(list(zip(*chunk)))
        return groups, chunks

    blocks = [list(range(world_size))]
    cfg_groups, cfg_blocks = make_parallel_groups(blocks, cfg_degree)
    for cfg_ranks in cfg_groups:
        if rank in cfg_ranks:
            PROCESS_GROUP.CFG_GROUP = dist.new_group(cfg_ranks)
            PROCESS_GROUP.CFG_RANKS = cfg_ranks

    sp_groups, sp_blocks = make_parallel_groups(cfg_blocks, sp_degree)
    for sp_ranks in sp_groups:
        if rank in sp_ranks:
            PROCESS_GROUP.SP_GROUP = dist.new_group(sp_ranks)
            PROCESS_GROUP.SP_RANKS = sp_ranks

    sp_ulysses_groups, sp_ulysses_blocks = make_parallel_groups(cfg_blocks, sp_ulysses_degree)
    for sp_ulysses_ranks in sp_ulysses_groups:
        if rank in sp_ulysses_ranks:
            PROCESS_GROUP.SP_ULYSSUES_GROUP = dist.new_group(sp_ulysses_ranks)
            PROCESS_GROUP.SP_ULYSSUES_RANKS = sp_ulysses_ranks

    sp_ring_groups, _ = make_parallel_groups(sp_ulysses_blocks, sp_ring_degree)
    for sp_ring_ranks in sp_ring_groups:
        if rank in sp_ring_ranks:
            PROCESS_GROUP.SP_RING_GROUP = dist.new_group(sp_ring_ranks)
            PROCESS_GROUP.SP_RING_RANKS = sp_ring_ranks

    tp_groups, _ = make_parallel_groups(sp_blocks, tp_degree)
    for tp_ranks in tp_groups:
        if rank in tp_ranks:
            PROCESS_GROUP.TP_GROUP = dist.new_group(tp_ranks)
            PROCESS_GROUP.TP_RANKS = tp_ranks

    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)


def clone(data):
    if isinstance(data, dict):
        return {k: clone(v) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [clone(t) for t in data]
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return copy.deepcopy(data)


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [to_device(t, device) for t in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    else:
        return data


def shard_model(
    module: nn.Module,
    device_id: int | torch.device,
    wrap_module_cls: Set[Type[nn.Module]],
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
):
    return FSDP(
        module,
        device_id=device_id,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=wrap_module_cls),
    )


def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: Optional[Union[ParallelStyle, Dict[str, ParallelStyle]]] = None,
):
    _validate_tp_mesh_dim(device_mesh)
    if parallelize_plan is None:
        return module
    if isinstance(parallelize_plan, ParallelStyle):
        return parallelize_plan._apply(module, device_mesh)
    for module_path, parallelize_style in parallelize_plan.items():
        if module_path.strip() == "":
            raise ValueError("Expect module path to be non-empty, but got empty string!")
        try:
            submodule = module.get_submodule(module_path)
            parallelize_style._apply(submodule, device_mesh)
        except AttributeError:
            continue
    return module


NCCL_TIMEOUT_SEC = int(os.environ.get("NCCL_TIMEOUT_SEC", 600))
PARALLEL_FWD_TIMEOUT_SEC = int(os.environ.get("PARALLEL_FWD_TIMEOUT_SEC", 3600))


def _worker_loop(
    rank: int,
    world_size: int,
    queue_in: List[mp.Queue],
    queue_out: mp.Queue,
    cfg_degree: int,
    sp_ulysses_degree: int,
    sp_ring_degree: int,
    tp_degree: int,
    use_fsdp: bool,
    master_port: int = 29500,
    device_type: str = "cuda",
):
    """
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    """
    if device_type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("parallelism is only supported on CUDA devices")

    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        device = torch.device(type=device_type, index=rank)
        torch.cuda.set_device(rank)

        timeout = timedelta(seconds=NCCL_TIMEOUT_SEC)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout,
            world_size=world_size,
            rank=rank,
            device_id=device,
        )
        init_parallel_pgs(
            cfg_degree=cfg_degree,
            sp_ulysses_degree=sp_ulysses_degree,
            sp_ring_degree=sp_ring_degree,
            tp_degree=tp_degree,
            rank=rank,
            world_size=world_size,
        )

        def wrap_for_parallel(module):
            if hasattr(module, "model_names"):
                for model_name in getattr(module, "model_names"):
                    submodule = getattr(module, model_name)
                    if getattr(submodule, "_supports_parallelization", False):
                        setattr(module, model_name, wrap_for_parallel(submodule))
                return module

            if not getattr(module, "_supports_parallelization", False):
                return module

            if tp_degree > 1:
                module = parallelize_module(
                    module=module,
                    device_mesh=DeviceMesh(device_type, torch.tensor(get_tp_ranks())),
                    parallelize_plan=module.get_tp_plan(),
                )
            elif use_fsdp:
                module = shard_model(module, device_id=device, wrap_module_cls=module.get_fsdp_module_cls())
            return module

        module = None
        data, args, kwargs = None, None, None

        while True:
            res = None
            data = queue_in[rank].get()

            if (name := data[0]) == "unload_module":
                module = None
                empty_cache()
            elif name == "load_module":
                init_fn, kwargs = data[1:]
                module = wrap_for_parallel(init_fn(**kwargs))
            elif module is None:
                res = RuntimeError("module is not initialized")
            elif not hasattr(module, name):
                res = AttributeError(f'{module.__class__.__name__} has no attribute "{name}"')
            else:
                args, kwargs = to_device(data[1:], device=device)
                with torch.no_grad():
                    res = getattr(module, name)(*args, **kwargs)

            if rank == 0:
                queue_out.put(res)
            data, args, kwargs = None, None, None
            dist.barrier()
    except Exception:
        import traceback

        msg = traceback.format_exc()
        err = RuntimeError(msg)
        logger.error(f"Error in worker loop (rank {rank}): {msg}")
        if rank == 0:
            queue_out.put(err)  # any exception caught in the worker will be raised to the main process
    finally:
        del module, data, args, kwargs
        torch.cuda.synchronize()
        empty_cache()
        dist.destroy_process_group()


class ParallelWrapper:
    def __init__(
        self,
        cfg_degree: int,
        sp_ulysses_degree: int,
        sp_ring_degree: int,
        tp_degree: int,
        use_fsdp: bool = False,
        master_port: int = 29500,
        device_type: str = "cuda",
    ):
        super().__init__()
        self._module_name = None

        self.world_size = cfg_degree * sp_ulysses_degree * sp_ring_degree * tp_degree
        spawn_ctx = mp.get_context("spawn")
        self.queue_in = [spawn_ctx.Queue() for _ in range(self.world_size)]
        self.queue_out = spawn_ctx.Queue()
        self.ctx = mp.spawn(
            _worker_loop,
            args=(
                self.world_size,
                self.queue_in,
                self.queue_out,
                cfg_degree,
                sp_ulysses_degree,
                sp_ring_degree,
                tp_degree,
                use_fsdp,
                master_port,
                device_type,
            ),
            nprocs=self.world_size,
            join=False,
        )

    def load_module(self, init_fn, **kwargs):
        data = ["load_module", init_fn, kwargs]
        for q in self.queue_in:
            q.put(data)
        try:
            res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
            self._module_name = init_fn.__self__.__name__
        except Empty:
            logger.error("[ParallelWrapper] load_module timeout")
            raise RuntimeError("[ParallelWrapper] load_module timeout")
        except Exception as e:
            logger.error(f"[ParallelWrapper] load_module error: {e}")
            raise RuntimeError(f"[ParallelWrapper] load_module error: {e}")
        logger.info("[ParallelWrapper] load_module done")

    def unload_module(self):
        data = ["unload_module", None, None]
        for q in self.queue_in:
            q.put(data)
        try:
            res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
        except Empty:
            logger.error("[ParallelWrapper] unload_module timeout")
            raise RuntimeError("[ParallelWrapper] unload_module timeout")
        except Exception as e:
            logger.error(f"[ParallelWrapper] unload_module error: {e}")
            raise RuntimeError(f"[ParallelWrapper] unload_module error: {e}")
        logger.info("[ParallelWrapper] unload_module done")

    def __call__(self, *args, **kwargs):
        data = ["__call__", args, kwargs]
        for q in self.queue_in:
            q.put(data)
        try:
            res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
        except Empty:
            logger.error(f"[ParallelWrapper] {self._module_name}.__call__ timeout")
            raise RuntimeError(f"[ParallelWrapper] {self._module_name}.__call__ timeout")
        except Exception as e:
            logger.error(f"[ParallelWrapper] {self._module_name}.__call__ error: {e}")
            raise RuntimeError(f"[ParallelWrapper] {self._module_name}.__call__ error: {e}")
        logger.info(f"[ParallelWrapper] {self._module_name}.__call__ done")
        return res

    def __getattr__(self, name):
        def wrapped_func(*args, **kwargs):
            data = [name, args, kwargs]
            for q in self.queue_in:
                q.put(data)
            try:
                res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
                if isinstance(res, Exception):
                    raise res
            except Empty:
                logger.error(f"[ParallelWrapper] {self._module_name}.{name} timeout")
                raise RuntimeError(f"[ParallelWrapper] {self._module_name}.{name} timeout")
            except Exception as e:
                logger.error(f"[ParallelWrapper] {self._module_name}.{name} error: {e}")
                raise RuntimeError(f"[ParallelWrapper] {self._module_name}.{name} error: {e}")
            logger.info(f"[ParallelWrapper] {self._module_name}.{name} done")
            return res

        return wrapped_func

    def __del__(self):
        # Send terminate signal to all workers
        for p in self.ctx.processes:
            p.terminate()
            p.join()
        for p in self.queue_in:
            p.close()
        self.queue_out.close()


@contextmanager
def cfg_parallel(tensors: List[torch.Tensor], use_cfg=True):
    if get_cfg_world_size() == 1 or not use_cfg:
        yield
        return

    original_tensors = clone(tensors)
    for tensor in tensors:
        if tensor is None:
            continue
        chunks = torch.chunk(tensor, get_cfg_world_size(), dim=0)
        chunk = chunks[get_cfg_rank()]
        chunk = chunk.clone()
        tensor.resize_(chunk.shape)
        tensor.copy_(chunk)
    yield
    for tensor, original_tensor in zip(tensors, original_tensors):
        if original_tensor is None:
            continue
        tensor.resize_(original_tensor.shape)
        tensor.copy_(original_tensor)


def cfg_parallel_unshard(tensors: List[torch.Tensor], use_cfg=True):
    if get_cfg_world_size() == 1 or not use_cfg:
        return tensors

    unshard_tensors = []
    for tensor in tensors:
        unshard = torch.zeros((get_cfg_world_size(), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(unshard, tensor, group=get_cfg_group())
        unshard_tensors.append(unshard)
    return unshard_tensors


@contextmanager
def sequence_parallel(
    tensors: List[torch.Tensor] | None = None,
    seq_dims: List[int] | None = None,
):
    if get_sp_world_size() == 1:
        yield
        return

    tensors = [] if tensors is None else tensors
    seq_dims = [] if seq_dims is None else seq_dims
    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"

    original_tensors = clone(tensors)
    for tensor, seq_dim in zip(tensors, seq_dims):
        if tensor is None:
            continue
        # pad seq_len to multiple of sp_world_size
        # TODO: long_context_attention does not support attn_mask, may cause loss of numerical precision with padding
        seq_len = tensor.size(seq_dim)
        pad_len = math.ceil(seq_len / get_sp_world_size()) * get_sp_world_size() - seq_len
        padding = [0] * (2 * tensor.ndim)
        padding[-2 * seq_dim - 1] = pad_len
        padded_tensor = F.pad(tensor, padding)

        chunks = torch.chunk(padded_tensor, get_sp_world_size(), dim=seq_dim)
        chunk = chunks[get_sp_rank()]
        chunk = chunk.clone()
        tensor.resize_(chunk.shape)
        tensor.copy_(chunk)

    origin_attention = attention_ops.attention
    attention_ops.attention = attention_ops.long_context_attention
    yield
    for tensor, original_tensor in zip(tensors, original_tensors):
        if original_tensor is None:
            continue
        tensor.resize_(original_tensor.shape)
        tensor.copy_(original_tensor)

    attention_ops.attention = origin_attention


def sequence_parallel_unshard(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    seq_lens: List[int],
) -> List[torch.Tensor]:
    if get_sp_world_size() == 1:
        return tensors

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"
    assert len(tensors) == len(seq_lens), "tensors and seq_lens must have the same number of elements"
    unshard_tensors = []
    for tensor, seq_dim, seq_len in zip(tensors, seq_dims, seq_lens):
        unshard = [torch.zeros_like(tensor) for _ in range(get_sp_world_size())]
        dist.all_gather(unshard, tensor, group=get_sp_group())
        unshard = torch.cat(unshard, dim=seq_dim).narrow(dim=seq_dim, start=0, length=seq_len)
        unshard_tensors.append(unshard)
    return unshard_tensors


__all__ = [
    "ParallelWrapper",
    "cfg_parallel",
    "cfg_parallel_unshard",
    "sequence_parallel",
    "sequence_parallel_unshard",
]
