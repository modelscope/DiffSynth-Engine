import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine.algorithm.sampler import EulerSampler, FlowMatchEulerSampler
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler, RecifitedFlowScheduler
from diffsynth_engine.pipelines.flux_image import calculate_shift


class TestSampler(ImageTestCase):
    def test_euler_sampler(self):
        num_inference_steps = 20
        scheduler = ScaledLinearScheduler()
        sigmas, timesteps = scheduler.schedule(num_inference_steps)
        sampler = EulerSampler()
        sampler.initialize(None, timesteps, sigmas, None)
        expect_tensor = self.get_expect_tensor("algorithm/euler_i10.safetensors")
        origin_sample = expect_tensor["origin_sample"]
        model_output = expect_tensor["model_output"]
        prev_sample = expect_tensor["prev_sample"]
        results = sampler.step(origin_sample, model_output, 10)
        self.assertTensorEqual(results, prev_sample)

    def test_flow_match_sampler(self):
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

        sampler = FlowMatchEulerSampler()
        sampler.initialize(None, timesteps, sigmas, None)

        expect_tensor = self.get_expect_tensor("algorithm/flow_match_euler_i10.safetensors")
        origin_sample = expect_tensor["origin_sample"]
        model_output = expect_tensor["model_output"]
        prev_sample = expect_tensor["prev_sample"]

        results = sampler.step(origin_sample, model_output, 10)
        self.assertTensorEqual(results, prev_sample)


if __name__ == "__main__":
    unittest.main()
