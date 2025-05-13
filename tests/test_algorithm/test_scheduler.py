import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine.algorithm.noise_scheduler import (
    ScaledLinearScheduler,
    ExponentialScheduler,
    KarrasScheduler,
    BetaScheduler,
    DDIMScheduler,
    SGMUniformScheduler,
)
from diffsynth_engine.algorithm.noise_scheduler.flow_match.recifited_flow import RecifitedFlowScheduler
from diffsynth_engine.pipelines.flux_image import calculate_shift


class TestScheduler(ImageTestCase):
    def test_linear_scheduler(self):
        scheduler = ScaledLinearScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/scaled_linear_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, expect_tensors["timesteps"])

    def test_karras_scheduler(self):
        scheduler = KarrasScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/karras_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_exponential_scheduler(self):
        scheduler = ExponentialScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/exponential_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_beta_scheduler(self):
        scheduler = BetaScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/beta_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_recifited_flow_scheduler(self):
        # FLUX
        width = 1024
        height = 1024
        num_inference_steps = 20
        scheduler = RecifitedFlowScheduler(use_dynamic_shifting=True)
        sigmas, timesteps = scheduler.schedule(
            num_inference_steps,
            mu=calculate_shift(width // 16 * height // 16),
            sigma_min=1.0 / num_inference_steps,
            sigma_max=1.0,
        )
        expect_tensors = self.get_expect_tensor("algorithm/recifited_flow_20steps_flux.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, expect_tensors["timesteps"])

    def test_ddim_scheduler(self):
        scheduler = DDIMScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/ddim_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_sgm_uniform_scheduler(self):
        scheduler = SGMUniformScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/sgm_uniform_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))


if __name__ == "__main__":
    unittest.main()
