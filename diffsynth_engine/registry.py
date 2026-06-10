import json
import os
from functools import cache
from typing import TYPE_CHECKING

from diffsynth_engine.plugins import load_general_plugins
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import MODEL_INDEX_NAME
from diffsynth_engine.utils.import_utils import LazyImport

if TYPE_CHECKING:
    from diffsynth_engine.layers.attention.backends.abstract import AttentionBackend
    from diffsynth_engine.pipelines.base import Pipeline

logger = logging.get_logger(__name__)

_PIPELINES: dict[str, str] = {
    "QwenImagePipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage:QwenImagePipeline",
    "QwenImageEditPipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage_edit:QwenImageEditPipeline",
    "QwenImageEditPlusPipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage_edit_plus:QwenImageEditPlusPipeline",
    "QwenImageLayeredPipeline": "diffsynth_engine.pipelines.qwen_image.pipeline_qwenimage_layered:QwenImageLayeredPipeline",
}

_ATTENTION_BACKENDS: dict[str, str] = {
    "aiter": "diffsynth_engine.layers.attention.backends.aiter:AiterBackend",
    "aiter_fp8": "diffsynth_engine.layers.attention.backends.aiter:AiterFP8Backend",
    "fa2": "diffsynth_engine.layers.attention.backends.flash_attn_2:FlashAttention2Backend",
    "fa3": "diffsynth_engine.layers.attention.backends.flash_attn_3:FlashAttention3Backend",
    "fa3_fp8": "diffsynth_engine.layers.attention.backends.flash_attn_3:FlashAttention3FP8Backend",
    "fa4": "diffsynth_engine.layers.attention.backends.flash_attn_4:FlashAttention4Backend",
    "sage2": "diffsynth_engine.layers.attention.backends.sage_attn_2:SageAttention2Backend",
    "sage3": "diffsynth_engine.layers.attention.backends.sage_attn_3:SageAttention3Backend",
    "sdpa": "diffsynth_engine.layers.attention.backends.sdpa:SDPABackend",
    "sparge": "diffsynth_engine.layers.attention.backends.sparge_attn:SpargeAttentionBackend",
}

PIPELINE_REGISTRY: dict[str, LazyImport] = {}
ATTENTION_BACKEND_REGISTRY: dict[str, LazyImport] = {}
_pipeline_registry_initialized = False
_attention_backend_registry_initialized = False


def register_pipeline(name: str, target: str) -> None:
    """Register a pipeline for lazy import.

    `target` must be a "module_name:class_name" string. The class is not
    imported until `get_pipeline_class(name)` is called.
    """
    if name in PIPELINE_REGISTRY:
        logger.warning(f"Pipeline {name!r} already exists, skipping registration")
        return

    module_name, class_name = target.split(":", 1)
    PIPELINE_REGISTRY[name] = LazyImport(module_name, class_name)


def register_attention_backend(attn_type: str, target: str) -> None:
    """Register an attention backend for lazy import.

    `target` must be a "module_name:class_name" string. The class is not
    imported until `get_attn_backend(attn_type)` is called.
    """
    if attn_type in ATTENTION_BACKEND_REGISTRY:
        logger.warning(f"Attention backend {attn_type!r} already exists, skipping registration")
        return

    module_name, class_name = target.split(":", 1)
    ATTENTION_BACKEND_REGISTRY[attn_type] = LazyImport(module_name, class_name)


def _register_builtin_pipelines() -> None:
    for name, pipeline_cls in _PIPELINES.items():
        register_pipeline(name, pipeline_cls)


def _register_builtin_attention_backends() -> None:
    for attn_type, backend_cls in _ATTENTION_BACKENDS.items():
        register_attention_backend(attn_type, backend_cls)


def get_pipeline_class_name(model_path: str) -> str:
    model_index_path = os.path.join(model_path, MODEL_INDEX_NAME)
    if not os.path.exists(model_index_path):
        raise FileNotFoundError(f"Model index file not found: {model_index_path}")

    with open(model_index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    if "_class_name" not in model_index:
        raise KeyError(f"_class_name field not found in {model_index_path}")

    return model_index["_class_name"]


def get_pipeline_class(name: str) -> type["Pipeline"]:
    global _pipeline_registry_initialized
    if not _pipeline_registry_initialized:
        _register_builtin_pipelines()
        load_general_plugins()
        _pipeline_registry_initialized = True

    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Pipeline class {name!r} not found. Available pipelines: {sorted(PIPELINE_REGISTRY)}")
    return PIPELINE_REGISTRY[name].load()


@cache
def get_attn_backend(attn_type: str | None = None) -> type["AttentionBackend"]:
    global _attention_backend_registry_initialized
    if not _attention_backend_registry_initialized:
        _register_builtin_attention_backends()
        load_general_plugins()
        _attention_backend_registry_initialized = True

    if attn_type is None:
        attn_type = "sdpa"
    if attn_type not in ATTENTION_BACKEND_REGISTRY:
        available_backends = sorted(ATTENTION_BACKEND_REGISTRY)
        raise ValueError(f"Attention backend {attn_type!r} not found. Available backends: {available_backends}")
    selected_backend = ATTENTION_BACKEND_REGISTRY[attn_type].load()
    selected_backend.check_availability()
    return selected_backend
