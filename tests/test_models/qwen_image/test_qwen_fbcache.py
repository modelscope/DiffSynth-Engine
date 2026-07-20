import inspect
import unittest

from diffsynth_engine.models.qwen_image.qwen_image_dit_fbcache import QwenImageDiTFBCache


class TestQwenFBCacheContract(unittest.TestCase):
    def test_forward_matches_current_qwen_conditioning_contract(self):
        parameters = inspect.signature(QwenImageDiTFBCache.forward).parameters
        for name in (
            "edit",
            "text_seq_lens",
            "context_latents",
            "entity_text",
            "entity_seq_lens",
            "entity_masks",
            "attn_kwargs",
        ):
            self.assertIn(name, parameters)

    def test_cache_state_resets_for_each_pipeline_call(self):
        model = QwenImageDiTFBCache.__new__(QwenImageDiTFBCache)
        model.step_count = 9
        model.num_inference_steps = 9
        model.refresh_cache_status(30, num_cache_streams=2)
        self.assertEqual(model.step_count, 0)
        self.assertEqual(model.num_inference_steps, 30)

        model._cache_states[0]["step_count"] = 4
        model._cache_states[1]["step_count"] = 7
        model.set_cache_stream(0)
        self.assertEqual(model.step_count, 4)
        model.set_cache_stream(1)
        self.assertEqual(model.step_count, 7)


if __name__ == "__main__":
    unittest.main()
