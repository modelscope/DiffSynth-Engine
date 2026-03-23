import torch
import torch.distributed as dist
from tqdm import tqdm

from diffsynth_engine.configs import PipelineConfig


class Pipeline:
    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config
        self.device = pipeline_config.device

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
