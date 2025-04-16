import os
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from yunchang.globals import Singleton, set_seq_parallel_pg

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.SP_GROUP: dist.ProcessGroup = None
        self.CFG_GROUP: dist.ProcessGroup = None


PROCESS_GROUP = ProcessGroupSingleton()


def get_sp_group():
    return PROCESS_GROUP.SP_GROUP


def get_sp_world_size():
    return PROCESS_GROUP.SP_GROUP.size()


def get_sp_rank():
    return PROCESS_GROUP.SP_GROUP.rank()


def get_cfg_group():
    return PROCESS_GROUP.CFG_GROUP


def get_cfg_world_size():
    return PROCESS_GROUP.CFG_GROUP.size()


def get_cfg_rank():
    return PROCESS_GROUP.CFG_GROUP.rank()


def init_parallel_pgs(
    sp_ulysses_degree: int = 1,
    sp_ring_degree: int = 1,
    cfg_degree: int = 1,
    rank: int = 0,
    world_size: int = 1,
):
    sp_degree = sp_ulysses_degree * sp_ring_degree

    assert world_size == sp_degree * cfg_degree, (
        f"world_size ({world_size}) must be equal to sp_degree ({sp_degree}) * cfg_degree ({cfg_degree})"
    )

    num_sp_pgs = world_size // sp_degree
    num_cfg_pgs = world_size // cfg_degree
    for i in range(num_sp_pgs):
        sp_ranks = list(range(i * sp_degree, (i + 1) * sp_degree))
        group = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            PROCESS_GROUP.SP_GROUP = group
    for i in range(num_cfg_pgs):
        cfg_ranks = list(range(i, sp_degree * cfg_degree, sp_degree))
        group = dist.new_group(cfg_ranks)
        if rank in cfg_ranks:
            PROCESS_GROUP.CFG_GROUP = group

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


NCCL_TIMEOUT_SEC = int(os.environ.get("NCCL_TIMEOUT_SEC", 600))
PARALLEL_FWD_TIMEOUT_SEC = int(os.environ.get("PARALLEL_FWD_TIMEOUT_SEC", 300))


def _worker_loop(
    rank: int,
    world_size: int,
    queue_in: mp.Queue,
    queue_out: mp.Queue,
    module: nn.Module,
    sp_ulysses_degree: int = 1,
    sp_ring_degree: int = 1,
    cfg_degree: int = 1,
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
        init_parallel_pgs(sp_ulysses_degree, sp_ring_degree, cfg_degree, rank, world_size)
        module = module.to(device)
        while True:
            if rank == 0:
                kwargs = queue_in.get()
                data = [kwargs]
            else:
                data = [None]
            dist.broadcast_object_list(data, src=0)
            kwargs = to_device(clone(data[0]), device)
            kwargs = split_and_get(kwargs, get_cfg_world_size(), 0, get_cfg_rank())
            del data
            with torch.no_grad():
                res = module(**kwargs)
            if get_sp_rank() == 0:
                gathered = torch.zeros((get_cfg_world_size(), *res.shape[1:]), dtype=res.dtype, device=res.device)
                dist.all_gather_into_tensor(gathered, res, group=get_cfg_group())
                res = gathered
            if rank == 0:
                queue_out.put(res)
            dist.barrier()
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error in worker loop (rank {rank}): {e}")
    finally:
        del module
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


class ParallelModel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        sp_ulysses_degree: int = 4,
        sp_ring_degree: int = 1,
        cfg_degree: int = 2,
        master_port: int = 29500,
        device: str = "cuda",
    ):
        super().__init__()
        self.world_size = sp_ulysses_degree * sp_ring_degree * cfg_degree
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
                sp_ulysses_degree,
                sp_ring_degree,
                cfg_degree,
                master_port,
                device,
            ),
            nprocs=self.world_size,
            join=False,
        )

    def forward(self, **kwargs):
        self.queue_in.put(kwargs)
        res = self.queue_out.get(timeout=PARALLEL_FWD_TIMEOUT_SEC)
        return res

    def __del__(self):
        # Send terminate signal to all workers
        for p in self.ctx.processes:
            p.terminate()
            p.join()
        self.queue_in.close()
        self.queue_out.close()


__all__ = ["ParallelModel"]
