import multiprocessing as mp
import os

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from diffsynth_engine.registry import get_pipeline_class
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.torch_profiler import TorchProfiler

logger = logging.get_logger(__name__)


class Worker:
    def __init__(
        self,
        local_rank: int,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        pipeline_config: PipelineConfig,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.pipeline_config = pipeline_config

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)

        cfg_degree = 2 if pipeline_config.use_cfg_parallel else 1
        sp_ulysses_degree = pipeline_config.sp_ulysses_degree
        sp_ring_degree = pipeline_config.sp_ring_degree
        sp_degree = sp_ulysses_degree * sp_ring_degree
        tp_degree = pipeline_config.tp_degree
        vae_parallel_size = world_size if pipeline_config.use_vae_parallel else 0
        initialize_model_parallel(
            classifier_free_guidance_degree=cfg_degree,
            sequence_parallel_degree=sp_degree,
            ulysses_degree=sp_ulysses_degree,
            ring_degree=sp_ring_degree,
            tensor_parallel_degree=tp_degree,
            vae_parallel_size=vae_parallel_size,
        )

        pipeline_class_name = self.pipeline_config.pipeline_class_name
        pipeline_class = get_pipeline_class(pipeline_class_name)
        self.pipeline = pipeline_class.from_pretrained(self.pipeline_config)

    def __call__(self, **kwargs):
        return self.pipeline(**kwargs)

    def start_profile(self, **kwargs):
        path = kwargs.get("path", ".")
        profile_rank0_only = kwargs.get("profile_rank0_only", True)
        return TorchProfiler.start(path, profile_rank0_only=profile_rank0_only)

    def stop_profile(self, **kwargs):
        result = TorchProfiler.stop()
        get_world_group().barrier()
        return result

    def __getattr__(self, name):
        pipeline = self.__dict__.get("pipeline")
        if pipeline is None:
            raise AttributeError(f"'{name}' attribute not found and pipeline is not initialized")
        return getattr(pipeline, name)


def run_worker_loop(
    local_rank: int,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    conn: mp.connection.Connection,
    pipeline_config: PipelineConfig,
):
    try:
        worker = Worker(
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            pipeline_config=pipeline_config,
        )

        logger.info(f"Worker process {rank} is ready")
        conn.send(
            {
                "status": "ready",
            }
        )

        world_group = get_world_group()

        while True:
            should_reply = rank == 0
            try:
                if rank == 0:
                    data = conn.recv()
                    world_group.broadcast_tensor_dict(data, src=0)
                else:
                    data = world_group.broadcast_tensor_dict(src=0)

                method = data.get("method")
                kwargs = data.get("kwargs", {})

                if method == "shutdown":
                    break

                output_rank = data.get("output_rank", 0)
                should_reply = output_rank is None or output_rank == rank

                output = getattr(worker, method)(**kwargs)
                if should_reply:
                    conn.send(
                        {
                            "status": "success",
                            "output": output,
                        }
                    )
                world_group.barrier()
            except EOFError as e:
                logger.error(f"Worker process {rank} connection closed: {e}", exc_info=True)
                if should_reply:
                    conn.send(
                        {
                            "status": "error",
                            "error": str(e),
                        }
                    )
                break
            except Exception as e:
                logger.error(f"Worker process {rank} error: {e}", exc_info=True)
                if should_reply:
                    conn.send(
                        {
                            "status": "error",
                            "error": str(e),
                        }
                    )
    except Exception as e:
        logger.error(f"Worker process {rank} error: {e}", exc_info=True)
        conn.send(
            {
                "status": "error",
                "error": str(e),
            }
        )
    finally:
        logger.info(f"Worker process {rank} is exiting")
        conn.close()
        destroy_model_parallel()
        destroy_distributed_environment()
