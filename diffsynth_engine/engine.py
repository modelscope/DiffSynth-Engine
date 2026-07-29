from typing import Any

import torch.multiprocessing as mp
from torch.cuda import set_device

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.registry import (
    get_pipeline_class,
    get_pipeline_class_name,
)
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.torch_profiler import TorchProfiler
from diffsynth_engine.worker import run_worker_loop

logger = logging.get_logger(__name__)


class DiffSynthEngine:
    @classmethod
    def from_pretrained(cls, model_path_or_config: str | PipelineConfig, **kwargs):
        pipeline_config = _resolve_pipeline_config(model_path_or_config)
        num_workers = pipeline_config.parallelism
        master_addr = kwargs.get("master_addr", "localhost")
        master_port = kwargs.get("master_port", 29500)
        nnodes = kwargs.get("nnodes", 1)
        node_rank = kwargs.get("node_rank", 0)

        if num_workers > 1:
            return DistributedEngine(pipeline_config, num_workers, master_addr, master_port, nnodes, node_rank)
        return LocalEngine(pipeline_config)

    def generate(self, **kwargs):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def start_profile(self, path: str = ".", profile_rank0_only: bool = True):
        raise NotImplementedError

    def stop_profile(self):
        raise NotImplementedError

    # LoRA APIs

    def load_loras(self, lora_args: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
        """Load LoRA weights and patch to the target module's LoRA layers.

        Args:
            lora_args: One LoRA argument dict or a list of LoRA argument dicts.
                lora_id: Unique LoRA model id.
                path: Safetensors file path to load.
                target_module: Pipeline module name to patch. If omitted or
                    None, the pipeline default target module is used.
                scale: Initial LoRA scale. If omitted, 1.0 is used.

        Returns:
            LoRA ids that were successfully loaded.
        """
        raise NotImplementedError

    def unload_loras(self, lora_ids: str | list[str] | None = None) -> None:
        """Unload LoRA weights that are not merged.

        Args:
            lora_ids: LoRA id or LoRA ids to unload.
                If None, unload all loaded LoRAs that are not merged.
        """
        raise NotImplementedError

    def set_active_loras(self, lora_ids: str | list[str], scales: float | list[float] | None = None) -> None:
        """Set selected LoRAs active and deactivate other unmerged LoRAs.

        Args:
            lora_ids: LoRA id or LoRA ids to set active.
            scales: Optional scale override for selected LoRAs.
                If float, apply the same scale to every selected LoRA.
                If list[float], apply one scale per LoRA id; its length must
                    match ``lora_ids``.
                If None, keep each selected LoRA's current scale.
        """
        raise NotImplementedError

    def activate_loras(self, lora_ids: str | list[str], scales: float | list[float] | None = None) -> None:
        """Activate selected LoRAs without changing other LoRA statuses.

        Args:
            lora_ids: LoRA id or LoRA ids to activate.
            scales: Optional scale override for activated LoRAs.
                If float, apply the same scale to every selected LoRA.
                If list[float], apply one scale per LoRA id; its length must
                    match ``lora_ids``.
                If None, keep each selected LoRA's current scale.
        """
        raise NotImplementedError

    def deactivate_loras(self, lora_ids: str | list[str] | None = None) -> None:
        """Deactivate LoRAs while keeping their weights loaded.

        Args:
            lora_ids: LoRA id or LoRA ids to deactivate.
                If None, deactivate all loaded LoRAs.
        """
        raise NotImplementedError

    def merge_loras(self, target_module: str | None = None, chunked: bool = False, high_precision: bool = True) -> None:
        """Merge active LoRA weights into base weights.

        Args:
            target_module: Target module to merge.
                If None, merge active LoRAs in all converted target modules.
            chunked: If True, merge in chunks to limit peak memory usage.
            high_precision: If True, compute merge in float32 for better numerical accuracy.
        """
        raise NotImplementedError

    def unmerge_loras(self, target_module: str | None = None) -> None:
        """Undo merged LoRAs and discard their LoRA refs.

        Args:
            target_module: Target module to unmerge.
                If None, unmerge LoRAs in all converted target modules.
        """
        raise NotImplementedError

    def reset_loras(self, target_module: str | None = None) -> None:
        """Reset LoRA status and restore base weights for selected modules.

        Args:
            target_module: Target module to reset.
                If None, reset LoRA status in all converted target modules.
        """
        raise NotImplementedError

    def list_loras(self, lora_ids: str | list[str] | None = None) -> list[dict[str, Any]]:
        """List loaded LoRAs and their current status.

        Args:
            lora_ids: LoRA id or LoRA ids to list.
                If None, list all loaded LoRAs.

        Returns:
            List of dicts with keys: lora_id, path, target_module, scale, status.
        """
        raise NotImplementedError

    def __del__(self):
        self.shutdown()


def _resolve_pipeline_config(model_path_or_config: str | PipelineConfig) -> PipelineConfig:
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

    return pipeline_config


class LocalEngine(DiffSynthEngine):
    def __init__(self, pipeline_config: PipelineConfig):
        logger.info("Initializing pipeline...")
        pipeline_class = get_pipeline_class(pipeline_config.pipeline_class_name)
        self.pipeline = pipeline_class.from_pretrained(pipeline_config)

    def generate(self, **kwargs):
        return self.pipeline(**kwargs)

    def shutdown(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    def load_loras(self, lora_args: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
        return self.pipeline.load_loras(lora_args=lora_args)

    def unload_loras(self, lora_ids: str | list[str] | None = None) -> None:
        return self.pipeline.unload_loras(lora_ids=lora_ids)

    def set_active_loras(self, lora_ids: str | list[str], scales: float | list[float] | None = None) -> None:
        return self.pipeline.set_active_loras(lora_ids=lora_ids, scales=scales)

    def activate_loras(self, lora_ids: str | list[str], scales: float | list[float] | None = None) -> None:
        return self.pipeline.activate_loras(lora_ids=lora_ids, scales=scales)

    def deactivate_loras(self, lora_ids: str | list[str] | None = None) -> None:
        return self.pipeline.deactivate_loras(lora_ids=lora_ids)

    def merge_loras(self, target_module: str | None = None, chunked: bool = False, high_precision: bool = True) -> None:
        return self.pipeline.merge_loras(target_module=target_module, chunked=chunked, high_precision=high_precision)

    def unmerge_loras(self, target_module: str | None = None) -> None:
        return self.pipeline.unmerge_loras(target_module=target_module)

    def reset_loras(self, target_module: str | None = None) -> None:
        return self.pipeline.reset_loras(target_module=target_module)

    def list_loras(self, lora_ids: str | list[str] | None = None) -> list[dict[str, Any]]:
        return self.pipeline.list_loras(lora_ids=lora_ids)

    def start_profile(self, path: str = ".", profile_rank0_only: bool = True):
        TorchProfiler.start(path, profile_rank0_only=profile_rank0_only)

    def stop_profile(self):
        return _collect_profile_results([TorchProfiler.stop()])


class DistributedEngine(DiffSynthEngine):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        num_workers: int,
        master_addr: str = "localhost",
        master_port: int = 29500,
        nnodes: int = 1,
        node_rank: int = 0,
    ):
        if nnodes <= 0:
            raise ValueError(f"nnodes must be positive, got {nnodes}")
        if not 0 <= node_rank < nnodes:
            raise ValueError(f"node_rank must be in [0, {nnodes}), got {node_rank}")
        if num_workers % nnodes != 0:
            raise ValueError(f"num_workers ({num_workers}) must be a multiple of nnodes ({nnodes})")

        nproc_per_node = num_workers // nnodes
        rank_offset = node_rank * nproc_per_node
        self.node_rank = node_rank

        logger.info(
            f"Initializing {nproc_per_node} workers on node {node_rank} "
            f"(world_size={num_workers}, master={master_addr}:{master_port})..."
        )

        set_device(0)

        self.workers = []
        self.conns = []

        ctx = mp.get_context("spawn")
        for local_rank in range(nproc_per_node):
            global_rank = rank_offset + local_rank
            conn_main, conn_worker = ctx.Pipe(duplex=True)

            process = ctx.Process(
                target=run_worker_loop,
                args=(
                    local_rank,  # local_rank
                    global_rank,  # rank
                    num_workers,  # world_size
                    master_addr,  # master_addr
                    master_port,  # master_port
                    conn_worker,  # conn
                    pipeline_config,  # pipeline_config
                ),
                name=f"diffsynth-worker-{global_rank}",
                daemon=True,
            )
            process.start()

            self.workers.append(process)
            self.conns.append(conn_main)

        for i, conn in enumerate(self.conns):
            result = conn.recv()
            if result["status"] != "ready":
                global_rank = rank_offset + i
                raise RuntimeError(f"Worker {global_rank} failed to start: {result.get('error', 'Unknown error')}")
        logger.info(f"All workers on node {node_rank} are ready")

    def _dispatch(self, method: str, output_rank: int | None = 0, **kwargs):
        self.conns[0].send(
            {
                "method": method,
                "output_rank": output_rank,
                "kwargs": kwargs or {},
            }
        )

        if output_rank is None:
            outputs = []
            for rank, conn in enumerate(self.conns):
                result = conn.recv()
                if result["status"] != "success":
                    raise RuntimeError(f"{method} failed on rank {rank}: {result.get('error', 'Unknown error')}")
                outputs.append(result["output"])
            return outputs

        result = self.conns[output_rank].recv()
        if result["status"] != "success":
            raise RuntimeError(f"{method} failed on rank {output_rank}: {result.get('error', 'Unknown error')}")

        return result["output"]

    def generate(self, **kwargs):
        return self._dispatch("__call__", output_rank=0, **kwargs)

    def load_loras(self, lora_args: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
        return self._dispatch("load_loras", lora_args=lora_args)

    def unload_loras(self, lora_ids: str | list[str] | None = None) -> None:
        return self._dispatch("unload_loras", lora_ids=lora_ids)

    def set_active_loras(self, lora_ids: str | list[str], scales: float | list[float] | None = None) -> None:
        return self._dispatch("set_active_loras", lora_ids=lora_ids, scales=scales)

    def activate_loras(self, lora_ids: str | list[str], scales: float | list[float] | None = None) -> None:
        return self._dispatch("activate_loras", lora_ids=lora_ids, scales=scales)

    def deactivate_loras(self, lora_ids: str | list[str] | None = None) -> None:
        return self._dispatch("deactivate_loras", lora_ids=lora_ids)

    def merge_loras(self, target_module: str | None = None, chunked: bool = False, high_precision: bool = True) -> None:
        return self._dispatch(
            "merge_loras", target_module=target_module, chunked=chunked, high_precision=high_precision
        )

    def unmerge_loras(self, target_module: str | None = None) -> None:
        return self._dispatch("unmerge_loras", target_module=target_module)

    def reset_loras(self, target_module: str | None = None) -> None:
        return self._dispatch("reset_loras", target_module=target_module)

    def list_loras(self, lora_ids: str | list[str] | None = None) -> list[dict[str, Any]]:
        return self._dispatch("list_loras", lora_ids=lora_ids)

    def shutdown(self):
        if self.workers is not None:
            logger.info("Shutting down workers...")

            try:
                self.conns[0].send({"method": "shutdown"})
            except (BrokenPipeError, OSError):
                pass

            for process in self.workers:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()

            for conn in self.conns:
                conn.close()

            self.workers = None
            self.conns = None

    def start_profile(self, path: str = ".", profile_rank0_only: bool = True):
        self._dispatch("start_profile", output_rank=0, path=path, profile_rank0_only=profile_rank0_only)

    def stop_profile(self):
        outputs = self._dispatch("stop_profile", output_rank=None)
        return _collect_profile_results(outputs)


def _collect_profile_results(outputs: list):
    results = {"traces": []}
    for output in outputs:
        if not isinstance(output, dict):
            continue

        trace = output.get("trace")
        if trace:
            results["traces"].append(trace)

    logger.info("Profile traces: %s", results["traces"])
    return results
