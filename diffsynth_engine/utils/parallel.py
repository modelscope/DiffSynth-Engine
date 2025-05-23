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
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from typing import Callable, Dict, List, Union, Optional
from queue import Empty

import diffsynth_engine.models.basic.attention as attention_ops
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.CFG_GROUP: Optional[dist.ProcessGroup] = None
        self.SP_GROUP: Optional[dist.ProcessGroup] = None
        self.TP_GROUP: Optional[dist.ProcessGroup] = None

        self.CFG_RANKS: List[int] = []
        self.SP_RANKS: List[int] = []
        self.TP_RANKS: List[int] = []


PROCESS_GROUP = ProcessGroupSingleton()


def get_cfg_group():
    return PROCESS_GROUP.CFG_GROUP


def get_cfg_world_size():
    return PROCESS_GROUP.CFG_GROUP.size()


def get_cfg_rank():
    return PROCESS_GROUP.CFG_GROUP.rank()


def get_cfg_ranks():
    return PROCESS_GROUP.CFG_RANKS


def get_sp_group():
    return PROCESS_GROUP.SP_GROUP


def get_sp_world_size():
    return PROCESS_GROUP.SP_GROUP.size()


def get_sp_rank():
    return PROCESS_GROUP.SP_GROUP.rank()


def get_sp_ranks():
    return PROCESS_GROUP.SP_RANKS


def get_tp_group():
    return PROCESS_GROUP.TP_GROUP


def get_tp_world_size():
    return PROCESS_GROUP.TP_GROUP.size()


def get_tp_rank():
    return PROCESS_GROUP.TP_GROUP.rank()


def get_tp_ranks():
    return PROCESS_GROUP.TP_RANKS


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
        cfg_group = dist.new_group(cfg_ranks)
        if rank in cfg_ranks:
            PROCESS_GROUP.CFG_GROUP = cfg_group
            PROCESS_GROUP.CFG_RANKS = cfg_ranks

    sp_groups, sp_blocks = make_parallel_groups(cfg_blocks, sp_degree)
    for sp_ranks in sp_groups:
        group = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            PROCESS_GROUP.SP_GROUP = group
            PROCESS_GROUP.SP_RANKS = sp_ranks

    tp_groups, _ = make_parallel_groups(sp_blocks, tp_degree)
    for tp_ranks in tp_groups:
        group = dist.new_group(tp_ranks)
        if rank in tp_ranks:
            PROCESS_GROUP.TP_GROUP = group
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
        return data.to(device)
    else:
        return data


def split_and_get(data, num, dim, index):
    if isinstance(data, dict):
        return {k: split_and_get(v, num, dim, index) for k, v in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [split_and_get(t, num, dim, index) for t in data]
    if isinstance(data, torch.Tensor):
        if data.shape[dim] < num:
            raise ValueError(f"data.shape[{dim}] ({data.shape[dim]}) < num ({num}), split failed")
        return torch.split(data, data.shape[dim] // num, dim)[index]
    return data


def shard_model(
    module: nn.Module,
    device_id: int,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    wrap_module_names: Optional[List[str]] = None,
):
    wrap_module_names = wrap_module_names or []

    def wrap_fn(m):
        for name in wrap_module_names:
            submodule = getattr(module, name)
            if isinstance(submodule, nn.ModuleList) and m in submodule:
                return True
            elif not isinstance(submodule, nn.ModuleList) and m is submodule:
                return True
        return False

    return FSDP(
        module,
        device_id=device_id,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=wrap_fn),
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
PARALLEL_FWD_TIMEOUT_SEC = int(os.environ.get("PARALLEL_FWD_TIMEOUT_SEC", 300))
PARALLEL_LORA_TIMEOUT_SEC = int(os.environ.get("PARALLEL_LORA_TIMEOUT_SEC ", 60))


def _worker_loop(
    rank: int,
    world_size: int,
    queue_in: mp.Queue,
    queue_out: mp.Queue,
    module: nn.Module,
    cfg_degree: int,
    sp_ulysses_degree: int,
    sp_ring_degree: int,
    tp_degree: int,
    shard_fn: Optional[Callable] = None,
    master_port: int = 29500,
    device: str = "cuda",
):
    """
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    """
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.cuda.set_device(rank)

        timeout = timedelta(seconds=NCCL_TIMEOUT_SEC)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout,
            world_size=world_size,
            rank=rank,
        )
        init_parallel_pgs(
            cfg_degree=cfg_degree,
            sp_ulysses_degree=sp_ulysses_degree,
            sp_ring_degree=sp_ring_degree,
            tp_degree=tp_degree,
            rank=rank,
            world_size=world_size,
        )

        if tp_degree > 1:
            module = parallelize_module(
                module=module,
                device_mesh=DeviceMesh(device, torch.tensor(get_tp_ranks())),
                parallelize_plan=module.get_tp_plan(),
            ).to(device)
        elif shard_fn:
            module = shard_fn(module=module, device_id=rank)
        else:
            module = module.to(device)

        while True:
            if rank == 0:
                kwargs = queue_in.get()
                data = [kwargs]
            else:
                data = [None]
            dist.broadcast_object_list(data, src=0)
            kwargs = clone(data[0])
            del data

            y = None
            if kwargs.get("method", None) == "load_loras":
                module.load_loras(lora_args=kwargs["lora_args"], fused=kwargs["fused"])
            elif kwargs.get("method", None) == "unload_loras":
                module.unload_loras()
            else:
                kwargs = to_device(kwargs, device)
                kwargs = split_and_get(kwargs, get_cfg_world_size(), 0, get_cfg_rank())
                with torch.no_grad():
                    y = module(**kwargs)
                if get_sp_rank() == 0 and get_tp_rank() == 0:
                    gathered = torch.zeros((get_cfg_world_size(), *y.shape[1:]), dtype=y.dtype, device=y.device)
                    dist.all_gather_into_tensor(gathered, y, group=get_cfg_group())
                    y = gathered

            if rank == 0:
                queue_out.put(y)
            dist.barrier()
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error in worker loop (rank {rank}): {e}")
        queue_out.put(e)  # any exception caught in the worker will be raised to the main process
    finally:
        del module
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


class ParallelModel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        cfg_degree: int,
        sp_ulysses_degree: int,
        sp_ring_degree: int,
        tp_degree: int,
        shard_fn: Optional[Callable] = None,
        master_port: int = 29500,
        device: str = "cuda",
    ):
        super().__init__()
        self.world_size = cfg_degree * sp_ulysses_degree * sp_ring_degree * tp_degree
        self.device = device
        self.queue_in = mp.Queue()
        self.queue_out = mp.Queue()
        self.ctx = mp.spawn(
            _worker_loop,
            args=(
                self.world_size,
                self.queue_in,
                self.queue_out,
                module,
                cfg_degree,
                sp_ulysses_degree,
                sp_ring_degree,
                tp_degree,
                shard_fn,
                master_port,
                device,
            ),
            nprocs=self.world_size,
            join=False,
        )

    def load_loras(self, lora_args: List[Dict[str, any]], fused: bool = True):
        self.queue_in.put(
            {
                "method": "load_loras",
                "lora_args": lora_args,
                "fused": fused,
            }
        )
        try:
            res = self.queue_out.get(timeout=PARALLEL_LORA_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
        except Empty:
            logger.error("ParallelModel load LoRA timeout")
            raise RuntimeError("ParallelModel load LoRA timeout")
        except Exception as e:
            logger.error(f"ParallelModel load LoRA error: {e}")
            raise RuntimeError(f"ParallelModel load LoRA error: {e}")
        logger.info("ParallelModel load LoRA done")

    def unload_loras(self):
        self.queue_in.put({"method": "unload_loras"})
        try:
            res = self.queue_out.get(timeout=PARALLEL_LORA_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
        except Empty:
            logger.error("ParallelModel unload LoRA timeout")
            raise RuntimeError("ParallelModel unload LoRA timeout")
        except Exception as e:
            logger.error(f"ParallelModel unload LoRA error: {e}")
            raise RuntimeError(f"ParallelModel unload LoRA error: {e}")
        logger.info("ParallelModel unload LoRA done")

    def forward(self, **kwargs):
        self.queue_in.put(kwargs)
        try:
            res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
            if isinstance(res, Exception):
                raise res
        except Empty:
            logger.error("ParallelModel forward timeout")
            raise RuntimeError("ParallelModel forward timeout")
        except Exception as e:
            logger.error(f"ParallelModel forward error: {e}")
            raise RuntimeError(f"ParallelModel forward error: {e}")
        return res

    def __del__(self):
        # Send terminate signal to all workers
        for p in self.ctx.processes:
            p.terminate()
            p.join()
        self.queue_in.close()
        self.queue_out.close()


_sequence_parallel_enabled = False


@contextmanager
def sequence_parallel(
    tensors: List[torch.Tensor] | None = None,
    seq_dims: List[int] | None = None,
    enabled: bool = False,
):
    if not enabled:
        yield
        return

    tensors = [] if tensors is None else tensors
    seq_dims = [] if seq_dims is None else seq_dims
    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"

    for tensor, seq_dim in zip(tensors, seq_dims):
        # pad seq_len to multiple of sp_world_size
        # TODO: long_context_attention does not support attn_mask, may cause loss of numerical precision with padding
        seq_len = tensor.size(seq_dim)
        pad_len = math.ceil(seq_len / get_sp_world_size()) * get_sp_world_size() - seq_len
        padding = [0] * (2 * tensor.ndim)
        padding[-2 * seq_dim - 1] = pad_len
        padded_tensor = F.pad(tensor, padding)

        chunks = torch.chunk(padded_tensor, get_sp_world_size(), dim=seq_dim)
        chunk = chunks[get_sp_rank()]
        tensor.resize_(chunk.shape)
        tensor.copy_(chunk)

    global _sequence_parallel_enabled
    _sequence_parallel_enabled = True
    origin_attention = attention_ops.attention
    attention_ops.attention = attention_ops.long_context_attention
    yield
    _sequence_parallel_enabled = False
    attention_ops.attention = origin_attention


def sequence_parallel_unshard(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    seq_lens: List[int],
) -> List[torch.Tensor]:
    if not _sequence_parallel_enabled:
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


__all__ = ["ParallelModel", "sequence_parallel", "sequence_parallel_unshard"]
