"""
Process group management for distributed training.

This module provides singleton-based process group management for distributed training,
including support for CFG parallelism, sequence parallelism (Ulysses + Ring), and tensor parallelism.
"""

import torch.distributed as dist
from typing import Optional, List


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.CFG_GROUP: Optional[dist.ProcessGroup] = None
            self.SP_GROUP: Optional[dist.ProcessGroup] = None
            self.SP_ULYSSUES_GROUP: Optional[dist.ProcessGroup] = None
            self.SP_RING_GROUP: Optional[dist.ProcessGroup] = None
            self.TP_GROUP: Optional[dist.ProcessGroup] = None

            self.CFG_RANKS: List[int] = []
            self.SP_RANKS: List[int] = []
            self.SP_ULYSSUES_RANKS: List[int] = []
            self.SP_RING_RANKS: List[int] = []
            self.TP_RANKS: List[int] = []

            self.initialized = True


PROCESS_GROUP = ProcessGroupSingleton()


# CFG parallel group functions
def get_cfg_group():
    return PROCESS_GROUP.CFG_GROUP


def get_cfg_world_size():
    return PROCESS_GROUP.CFG_GROUP.size() if PROCESS_GROUP.CFG_GROUP is not None else 1


def get_cfg_rank():
    return PROCESS_GROUP.CFG_GROUP.rank() if PROCESS_GROUP.CFG_GROUP is not None else 0


def get_cfg_ranks():
    return PROCESS_GROUP.CFG_RANKS


# Sequence parallel group functions
def get_sp_group():
    return PROCESS_GROUP.SP_GROUP


def get_sp_world_size():
    return PROCESS_GROUP.SP_GROUP.size() if PROCESS_GROUP.SP_GROUP is not None else 1


def get_sp_rank():
    return PROCESS_GROUP.SP_GROUP.rank() if PROCESS_GROUP.SP_GROUP is not None else 0


def get_sp_ranks():
    return PROCESS_GROUP.SP_RANKS


# Sequence parallel Ulysses group functions
def get_sp_ulysses_group():
    return PROCESS_GROUP.SP_ULYSSUES_GROUP


def get_sp_ulysses_world_size():
    return PROCESS_GROUP.SP_ULYSSUES_GROUP.size() if PROCESS_GROUP.SP_ULYSSUES_GROUP is not None else 1


def get_sp_ulysses_rank():
    return PROCESS_GROUP.SP_ULYSSUES_GROUP.rank() if PROCESS_GROUP.SP_ULYSSUES_GROUP is not None else 0


def get_sp_ulysses_ranks():
    return PROCESS_GROUP.SP_ULYSSUES_RANKS


# Sequence parallel Ring group functions
def get_sp_ring_group():
    return PROCESS_GROUP.SP_RING_GROUP


def get_sp_ring_world_size():
    return PROCESS_GROUP.SP_RING_GROUP.size() if PROCESS_GROUP.SP_RING_GROUP is not None else 1


def get_sp_ring_rank():
    return PROCESS_GROUP.SP_RING_GROUP.rank() if PROCESS_GROUP.SP_RING_GROUP is not None else 0


def get_sp_ring_ranks():
    return PROCESS_GROUP.SP_RING_RANKS


# Tensor parallel group functions
def get_tp_group():
    return PROCESS_GROUP.TP_GROUP


def get_tp_world_size():
    return PROCESS_GROUP.TP_GROUP.size() if PROCESS_GROUP.TP_GROUP is not None else 1


def get_tp_rank():
    return PROCESS_GROUP.TP_GROUP.rank() if PROCESS_GROUP.TP_GROUP is not None else 0


def get_tp_ranks():
    return PROCESS_GROUP.TP_RANKS


__all__ = [
    "PROCESS_GROUP",
    "get_cfg_group",
    "get_cfg_world_size",
    "get_cfg_rank",
    "get_cfg_ranks",
    "get_sp_group",
    "get_sp_world_size",
    "get_sp_rank",
    "get_sp_ranks",
    "get_sp_ulysses_group",
    "get_sp_ulysses_world_size",
    "get_sp_ulysses_rank",
    "get_sp_ulysses_ranks",
    "get_sp_ring_group",
    "get_sp_ring_world_size",
    "get_sp_ring_rank",
    "get_sp_ring_ranks",
    "get_tp_group",
    "get_tp_world_size",
    "get_tp_rank",
    "get_tp_ranks",
]
