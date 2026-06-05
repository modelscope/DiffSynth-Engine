import json
import os

from diffsynth_engine.pipelines.base import Pipeline
from diffsynth_engine.plugins import load_general_plugins
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import MODEL_INDEX_NAME
from diffsynth_engine.utils.import_utils import LazyImport

logger = logging.get_logger(__name__)

_DIFFSYNTH_PIPELINES: dict[str, str] = {
    "QwenImagePipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage:QwenImagePipeline",
    "QwenImageEditPipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage_edit:QwenImageEditPipeline",
    "QwenImageEditPlusPipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage_edit_plus:QwenImageEditPlusPipeline",
    "QwenImageLayeredPipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage_layered:QwenImageLayeredPipeline",
}

PIPELINE_REGISTRY: dict[str, LazyImport] = {}


def register_pipeline(name: str, target: str) -> None:
    """Register a pipeline for lazy import.

    `target` must be a "module_name:class_name" string. The class is not
    imported until `get_pipeline_class(name)` is called.
    """
    module_name, class_name = target.split(":", 1)
    PIPELINE_REGISTRY[name] = LazyImport(module_name, class_name)


def _register_builtin_pipelines() -> None:
    for name, pipeline_cls in _DIFFSYNTH_PIPELINES.items():
        register_pipeline(name, pipeline_cls)


def get_pipeline_class_name(model_path: str) -> str:
    model_index_path = os.path.join(model_path, MODEL_INDEX_NAME)
    if not os.path.exists(model_index_path):
        raise FileNotFoundError(f"Model index file not found: {model_index_path}")

    with open(model_index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    if "_class_name" not in model_index:
        raise KeyError(f"_class_name field not found in {model_index_path}")

    return model_index["_class_name"]


def get_pipeline_class(name: str) -> type[Pipeline]:
    if not PIPELINE_REGISTRY:
        _register_builtin_pipelines()
        load_general_plugins()
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Pipeline class {name!r} not found. Available pipelines: {sorted(PIPELINE_REGISTRY)}")
    return PIPELINE_REGISTRY[name].load()
