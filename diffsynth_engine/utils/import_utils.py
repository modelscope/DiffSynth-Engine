# Copied from https://github.com/sgl-project/sglang

import importlib


class LazyImport:
    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self.load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self.load()
        return module(*args, **kwargs)
