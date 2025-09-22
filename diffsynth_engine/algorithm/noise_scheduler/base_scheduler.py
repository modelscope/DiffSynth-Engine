import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class BaseScheduler:
    def __init__(self):
        self._initial_params = {}

    def store_initial_config(self, **params):
        self._initial_params.update(params)

    def update_scheduler_config(self, **kwargs):
        for param_name, new_value in kwargs.items():
            if hasattr(self, param_name) and getattr(self, param_name) != new_value:
                setattr(self, param_name, new_value)

    def restore_scheduler_config(self):
        for param_name, initial_value in self._initial_params.items():
            if hasattr(self, param_name) and getattr(self, param_name) != initial_value:
                setattr(self, param_name, initial_value)

    def schedule(self, num_inference_steps: int):
        raise NotImplementedError()
