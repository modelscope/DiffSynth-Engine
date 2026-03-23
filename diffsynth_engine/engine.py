import torch.multiprocessing as mp
from torch.cuda import set_device

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.pipelines.utils import (
    get_pipeline_class,
    get_pipeline_class_name,
)
from diffsynth_engine.utils import logging
from diffsynth_engine.worker import run_worker_loop

logger = logging.get_logger(__name__)


class DiffSynthEngine:
    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config
        self.pipeline = None
        self.workers = None
        self.conns = None

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | PipelineConfig, **kwargs):
        if isinstance(model_path_or_config, str):
            pipeline_config = PipelineConfig(model_path=model_path_or_config)
        else:
            pipeline_config = model_path_or_config

        pipeline_class_name = pipeline_config.pipeline_class_name
        if pipeline_class_name is None:
            logger.info(f"pipeline_class_name is not set, infer from {pipeline_config.model_path}...")
            pipeline_class_name = get_pipeline_class_name(pipeline_config.model_path)
            pipeline_config.pipeline_class_name = pipeline_class_name
            logger.info(f"pipeline_class_name is set to {pipeline_class_name}")

        instance = cls(pipeline_config)

        num_workers = pipeline_config.parallelism
        master_port = kwargs.get("master_port", 29500)
        if num_workers > 1:
            instance._init_workers(num_workers, master_port)
        else:
            instance._init_pipeline()
        return instance

    def _init_workers(self, num_workers: int, master_port: int):
        logger.info(f"Initializing {num_workers} workers...")

        set_device(0)

        self.workers = []
        self.conns = []

        ctx = mp.get_context("spawn")
        for rank in range(num_workers):
            conn_main, conn_worker = ctx.Pipe(duplex=True)

            process = ctx.Process(
                target=run_worker_loop,
                args=(
                    rank,  # local_rank
                    rank,  # rank
                    num_workers,  # world_size
                    master_port,  # master_port
                    conn_worker,  # conn
                    self.pipeline_config,  # pipeline_config
                ),
                name=f"diffsynth-worker-{rank}",
                daemon=True,
            )
            process.start()

            self.workers.append(process)
            self.conns.append(conn_main)

        for rank, conn in enumerate(self.conns):
            result = conn.recv()
            if result["status"] != "ready":
                raise RuntimeError(f"Worker {rank} failed to start: {result.get('error', 'Unknown error')}")
        logger.info("All workers are ready")

    def _init_pipeline(self):
        logger.info("Initializing pipeline...")
        pipeline_class = get_pipeline_class(self.pipeline_config.pipeline_class_name)
        self.pipeline = pipeline_class.from_pretrained(self.pipeline_config)

    def generate(self, **kwargs):
        if self.workers is not None:
            return self._generate(**kwargs)
        else:
            return self.pipeline(**kwargs)

    def _generate(self, **kwargs):
        # TODO: health check and timeout
        self.conns[0].send({"method": "__call__", "kwargs": kwargs})

        result = self.conns[0].recv()

        if result["status"] != "success":
            raise RuntimeError(f"Generation failed: {result.get('error', 'Unknown error')}")

        return result["output"]

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.workers is not None:
            logger.info("Shutting down workers...")

            self.conns[0].send({"method": "shutdown"})

            for process in self.workers:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()

            for conn in self.conns:
                conn.close()

            self.workers = None
            self.conns = None
