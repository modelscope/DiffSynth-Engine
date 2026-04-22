# Copied from https://github.com/sgl-project/sglang

import importlib


def is_npu_available():
    """Detect if NPU is available using mindiesd.utils.is_npu_available.

    Falls back to manual detection if mindiesd is not available.
    """
    mindiesd_spec = importlib.util.find_spec("mindiesd")
    if mindiesd_spec is not None:
        try:
            from mindiesd.utils import is_npu_available as mindiesd_is_npu_available

            return mindiesd_is_npu_available()
        except (ImportError, AttributeError):
            pass

    # Fallback to manual detection
    if importlib.util.find_spec("torch_npu") is None:
        return False
    try:
        import torch

        import torch_npu

        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


class LazyImport:
    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)
