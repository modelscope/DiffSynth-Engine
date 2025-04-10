import torch

from diffsynth_engine.algorithm.noise_scheduler.stable_diffusion.linear import ScaledLinearScheduler
from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero


class DDIMScheduler(ScaledLinearScheduler):
    """
    Implemented based on: https://arxiv.org/pdf/2010.02502.pdf
    """

    def __init__(self):
        super().__init__()

    def schedule(self, num_inference_steps: int):
        inner_sigmas = self.get_sigmas()
        sigmas = []
        ss = max(len(inner_sigmas) // num_inference_steps, 1)
        for i in range(1, len(inner_sigmas), ss):
            sigmas.append(float(inner_sigmas[i]))
        sigmas = sigmas[::-1]
        sigmas = torch.FloatTensor(sigmas).to(self.device)
        timesteps = self.sigma_to_t(sigmas)
        sigmas = append_zero(sigmas)
        return sigmas, timesteps
