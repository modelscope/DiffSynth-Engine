import torch
import torch.distributed as dist
from tqdm import tqdm

from diffsynth_engine.configs import PipelineConfig


class Pipeline:
    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config
        self.device = pipeline_config.device
        self._ensure_npu_device()

    def _ensure_npu_device(self):
        """Set NPU device context so that MindIE-SD custom ops can execute correctly.

        MindIE-SD fused operators (adaln_v2, rotary_position_embedding, attention_forward,
        etc.) rely on the current-device context (torch.npu.current_device()) to launch
        kernels, unlike torch_npu built-in ops which derive the target device from the
        input tensor's .device attribute.  If set_device is not called, the current device
        defaults to 0 regardless of where the tensors reside, causing a vector-core
        exception (507035) on any device other than 0.

        The check "npu" in device_str keeps this path entirely inert for CUDA / ROCm /
        CPU backends — no torch_npu import, no side effects.
        """
        device_str = str(self.device)
        if "npu" not in device_str:
            return
        try:
            import torch_npu

            parts = device_str.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else 0
            torch_npu.npu.set_device(device_id)
        except (ImportError, ValueError):
            # torch_npu not installed or device string unparseable — silently skip.
            # Inference will fall back to standard ops (AdaLayerNorm fallback path).
            pass

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | PipelineConfig):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(self._progress_bar_config)
        if "disable" not in progress_bar_config:
            is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
            progress_bar_config["disable"] = not is_rank_zero

        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    # TODO: preprocess & postprocess & LoRA
