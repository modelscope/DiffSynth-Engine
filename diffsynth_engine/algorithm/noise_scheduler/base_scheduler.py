import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class BaseScheduler:
    def __init__(self):
        self._stored_config = {}

    def store_config(self):
        self._stored_config = {
            config_name: config_value
            for config_name, config_value in vars(self).items()
            if not config_name.startswith("_")
        }

    def update_config(self, config_dict):
        for config_name, new_value in config_dict.items():
            if hasattr(self, config_name):
                setattr(self, config_name, new_value)

    def restore_config(self):
        for config_name, config_value in self._stored_config.items():
            setattr(self, config_name, config_value)

    def schedule(self, num_inference_steps: int):
        raise NotImplementedError()
