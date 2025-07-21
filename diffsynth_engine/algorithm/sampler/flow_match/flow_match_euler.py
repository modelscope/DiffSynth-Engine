import torch


class FlowMatchEulerSampler:
    def initialize(self, init_latents, timesteps, sigmas, mask=None):
        self.init_latents = init_latents
        self.timesteps = timesteps
        self.sigmas = sigmas
        self.mask = mask

    def step(self, latents, model_outputs, i):
        dt = self.sigmas[i + 1] - self.sigmas[i]
        latents = latents.to(dtype=torch.float32)
        latents = latents + model_outputs * dt
        latents = latents.to(dtype=model_outputs.dtype)
        if self.mask is not None:
            latents = latents * self.mask + self.init_latents * (1 - self.mask)
        return latents

    def add_noise(self, latents, noise, sigma):
        return (1 - sigma) * latents + noise * sigma
