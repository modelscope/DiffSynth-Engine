import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class BaseScheduler:
    def __init__(self):
        self._initial_params = {}

    def store_initial_config(self):
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                self._initial_params[attr_name] = getattr(self, attr_name)

    def update_scheduler_config(self, config_dict):
        for param_name, new_value in config_dict.items():
            if hasattr(self, param_name) and getattr(self, param_name) != new_value:
                setattr(self, param_name, new_value)

    def restore_scheduler_config(self):
        for param_name, initial_value in self._initial_params.items():
            if hasattr(self, param_name):
                setattr(self, param_name, initial_value)

    def schedule(self, num_inference_steps: int):
        raise NotImplementedError()
