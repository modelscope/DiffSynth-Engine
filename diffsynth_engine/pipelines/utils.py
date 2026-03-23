import importlib
import json
import os
import pkgutil
from typing import Dict, Type

from diffsynth_engine.pipelines.base import Pipeline
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import MODEL_INDEX_NAME

logger = logging.get_logger(__name__)


def _build_pipeline_class_map() -> Dict[str, str]:
    pipeline_class_map = {}
    module = importlib.import_module("diffsynth_engine.pipelines")

    for _, name, ispkg in pkgutil.iter_modules(module.__path__, "diffsynth_engine.pipelines."):
        if not ispkg:
            continue

        try:
            submodule = importlib.import_module(name)
            if not hasattr(submodule, "__all__"):
                continue

            for class_name in submodule.__all__:
                if not hasattr(submodule, class_name):
                    continue

                cls = getattr(submodule, class_name)
                if isinstance(cls, type) and issubclass(cls, Pipeline):
                    pipeline_class_map[class_name] = name
        except (ImportError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to import {name}: {e}", exc_info=True)
            continue

    return pipeline_class_map


_PIPELINE_CLASS_MAP = _build_pipeline_class_map()


def get_pipeline_class_name(model_path: str) -> str:
    model_index_path = os.path.join(model_path, MODEL_INDEX_NAME)
    if not os.path.exists(model_index_path):
        raise FileNotFoundError(f"Model index file not found: {model_index_path}")

    with open(model_index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    if "_class_name" not in model_index:
        raise KeyError(f"_class_name field not found in {model_index_path}")

    return model_index["_class_name"]


def get_pipeline_class(pipeline_class_name: str) -> Type[Pipeline]:
    if pipeline_class_name in _PIPELINE_CLASS_MAP:
        module_path = _PIPELINE_CLASS_MAP[pipeline_class_name]
        module = importlib.import_module(module_path)
        if hasattr(module, pipeline_class_name):
            pipeline_class = getattr(module, pipeline_class_name)
            if not issubclass(pipeline_class, Pipeline):
                raise ValueError(f"Class {pipeline_class_name} from {module_path} is not a subclass of Pipeline")
            return pipeline_class
    raise ValueError(
        f"Pipeline class '{pipeline_class_name}' not found. Available pipelines: {list(_PIPELINE_CLASS_MAP.keys())}"
    )
