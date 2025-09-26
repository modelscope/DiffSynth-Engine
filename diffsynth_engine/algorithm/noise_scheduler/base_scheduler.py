import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class BaseScheduler:
    def __init__(self):
        self._initial_params = {}

    def store_initial_config(self):
        self._initial_params = {attr_name: attr_value for attr_name, attr_value in vars(self).items()}

    def update_scheduler_config(self, config_dict):
        for param_name, new_value in config_dict.items():
            if hasattr(self, param_name):
                setattr(self, param_name, new_value)

    def restore_scheduler_config(self):
        current_attrs = set(vars(self).keys())
        initial_attrs = set(self._initial_params.keys())

        for param_name, initial_value in self._initial_params.items():
            setattr(self, param_name, initial_value)

        for attr_name in current_attrs - initial_attrs:
            delattr(self, attr_name)

    def schedule(self, num_inference_steps: int):
        raise NotImplementedError()
