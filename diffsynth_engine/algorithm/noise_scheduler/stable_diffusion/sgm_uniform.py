import torch

from diffsynth_engine.algorithm.noise_scheduler.stable_diffusion.linear import ScaledLinearScheduler
from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero


class SGMUniformScheduler(ScaledLinearScheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, num_inference_steps: int):
        # suppose sigma_min and sigma_max is default value
        timesteps = torch.linspace(999, 0, num_inference_steps + 1)[:-1]
        sigmas = [self.t_to_sigma(timestep) for timestep in timesteps]
        sigmas = torch.FloatTensor(sigmas).to(self.device)
        sigmas = append_zero(sigmas)
        return sigmas, timesteps
