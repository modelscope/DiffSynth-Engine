import torch


class FlowMatchEulerSampler:
    def initialize(self, sigmas):
        self.sigmas = sigmas

    def step(self, latents, model_outputs, i):
        dt = self.sigmas[i + 1] - self.sigmas[i]
        latents = latents.to(dtype=torch.float32)
        latents = latents + model_outputs * dt
        latents = latents.to(dtype=model_outputs.dtype)
        return latents

    def add_noise(self, latents, noise, sigma):
        return (1 - sigma) * latents + noise * sigma
