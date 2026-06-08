# References:
# - https://github.com/vllm-project/vllm-omni/blob/v0.20.0/vllm_omni/plugins/__init__.py

from collections.abc import Callable
from importlib.metadata import entry_points

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

# entry point group name
DIFFSYNTH_DEFAULT_PLUGINS_GROUP = "diffsynth_engine.general_plugins"

_plugins_loaded = False


def load_plugins_by_group(group: str) -> dict[str, Callable]:
    """Discover external plugins via entry points."""
    plugins: dict[str, Callable] = {}
    for ep in entry_points(group=group):
        try:
            func = ep.load()
            plugins[ep.name] = func
        except Exception:
            logger.exception("Failed to load plugin %r (%s)", ep.name, ep.value)
            continue
        logger.info("Loaded plugin %r from %s", ep.name, ep.value)

    return plugins


def load_general_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    plugins = load_plugins_by_group(DIFFSYNTH_DEFAULT_PLUGINS_GROUP)
    # execute the loaded functions of general plugins
    for name, func in plugins.items():
        try:
            func()
            logger.info("Executed general plugin %r", name)
        except Exception:
            logger.warning("Failed to execute general plugin %r", name, exc_info=True)
