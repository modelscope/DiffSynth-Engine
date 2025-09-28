import torch
import math

from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import BaseScheduler


def append(x, value):
    return torch.cat([x, x.new_ones([1]) * value])


class RecifitedFlowScheduler(BaseScheduler):
    def __init__(
        self,
        shift=1.0,
        sigma_min=None,
        sigma_max=None,
        num_train_timesteps=1000,
        use_dynamic_shifting=False,
        shift_terminal=None,
        exponential_shift_mu=None,
    ):
        super().__init__()
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps
        self.use_dynamic_shifting = use_dynamic_shifting
        self.shift_terminal = shift_terminal
        # static mu for distill model
        self.exponential_shift_mu = exponential_shift_mu
        self.store_config()

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def _t_to_sigma(self, t):
        return t / self.num_train_timesteps

    def _time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _shift_sigma(self, sigma: torch.Tensor, shift: float):
        return shift * sigma / (1 + (shift - 1) * sigma)

    def _stretch_shift_to_terminal(self, sigma: torch.Tensor):
        one_minus_z = 1 - sigma
        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
        return 1 - (one_minus_z / scale_factor)

    def schedule(
        self,
        num_inference_steps: int,
        mu: float | None = None,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        append_value: float = 0,
    ):
        sigma_min = sigma_min if self.sigma_min is None else self.sigma_min
        sigma_max = sigma_max if self.sigma_max is None else self.sigma_max
        sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps)
        if self.exponential_shift_mu is not None:
            mu = self.exponential_shift_mu
        if self.use_dynamic_shifting:
            sigmas = self._time_shift(mu, 1.0, sigmas)  # FLUX
        else:
            sigmas = self._shift_sigma(sigmas, self.shift)
        if self.shift_terminal is not None:
            sigmas = self._stretch_shift_to_terminal(sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = append(sigmas, append_value)
        return sigmas, timesteps
