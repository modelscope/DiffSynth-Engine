"""NPU profiling hook for multi-worker distributed inference.

All profiler configuretion values are provided by the caller (script side).
This module only contains the mapping mechanism from a raw dict to
torch_npu.profiler — it does NOT store any config values.
"""

import logging

logger = logging.get_logger(__name__)


def _resolve_profiler_enums(cfg: dict) -> dict:
    """Map int profiler config values to torch_npu enum types."""
    import torch_npu

    resolved = dict(cfg)

    level = cfg["profiler_level"]
    if isinstance(level, int):
        resolved["profiler_level"] = (
            torch_npu.profiler.ProfilerLevel.Level0,
            torch_npu.profiler.ProfilerLevel.Level1,
            torch_npu.profiler.ProfilerLevel.Level2,
        )[level]

    metrics = cfg["aic_metrics"]
    if isinstance(metrics, int):
        resolved["aic_metrics"] = (
            torch_npu.profiler.AiCMetrics.PipeUtilization,
            torch_npu.profiler.AiCMetrics.AiCoreUtilization,
            torch_npu.profiler.AiCMetrics.L2Cache,
            torch_npu.profiler.AiCMetrics.Memory,
        )[metrics]

    activities = cfg["activities"]
    if activities and isinstance(activities[0], int):
        activity_map = {
            0: torch_npu.profiler.ProfilerActivity.CPU,
            1: torch_npu.profiler.ProfilerActivity.NPU,
        }
        resolved["activities"] = [activity_map[a] for a in activities]

    return resolved


def setup_block_profiling(profiling_tag: str, profiler_config: dict, rank: int):
    """Create a torch_npu.profiler from *profiler_config* and register it.

    The profiler will auto-start/step/stop inside the first 4 transformer
    blocks (the hook points already exist in transformer_qwenimage.py).

    Args:
        profiling_tag: trace output sub-directory or full path prefix.
        profiler_config: PROFILER_CONFIG dict using only int/bool/str/float.
        rank: worker rank; non‑0 skips so we only profile one replica.
    """
    if not profiling_tag or not profiler_config:
        return
    if rank != 0:
        return

    import torch_npu

    from diffsynth_engine.models.qwen_image.transformer_qwenimage import (
        register_multi_block_profiler,
        reset_multi_block_profiler,
    )

    reset_multi_block_profiler()

    cfg = _resolve_profiler_enums(profiler_config)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=cfg["profiler_level"],
        aic_metrics=cfg["aic_metrics"],
        data_simplification=cfg["data_simplification"],
        msprof_tx=cfg["msprof_tx"],
        l2_cache=cfg["l2_cache"],
        op_attr=cfg["op_attr"],
        record_op_args=cfg["record_op_args"],
    )
    profiler = torch_npu.profiler.profile(
        activities=cfg["activities"],
        schedule=torch_npu.profiler.schedule(
            wait=cfg["schedule_wait"],
            warmup=cfg["schedule_warmup"],
            active=cfg["schedule_active"],
            repeat=cfg["schedule_repeat"],
            skip_first=cfg["schedule_skip_first"],
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_tag),
        record_shapes=cfg["record_shapes"],
        profile_memory=cfg["profile_memory"],
        with_stack=cfg["with_stack"],
        with_flops=cfg["with_flops"],
        experimental_config=experimental_config,
    )
    register_multi_block_profiler(profiler)
    logger.info("[Profiling] tag=%s", profiling_tag)


def teardown_block_profiling():
    """Reset profiling state so a new profiler can be registered."""
    from diffsynth_engine.models.qwen_image.transformer_qwenimage import (
        reset_multi_block_profiler,
    )

    reset_multi_block_profiler()
