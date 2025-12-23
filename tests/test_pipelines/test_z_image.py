import unittest
import torch

from diffsynth_engine import ZImagePipelineConfig
from diffsynth_engine.pipelines import ZImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestZImagePipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        config = ZImagePipelineConfig(
            model_path=fetch_model("Tongyi-MAI/Z-Image-Turbo", path="transformer/*.safetensors"),
            encoder_path=fetch_model("Tongyi-MAI/Z-Image-Turbo", path="text_encoder/*.safetensors"),
            vae_path=fetch_model("Tongyi-MAI/Z-Image-Turbo", path="vae/*.safetensors"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.float32,
            batch_cfg=True,
        )
        cls.pipe = ZImagePipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
        image = self.pipe(
            prompt=prompt,
            #            negative_prompt="ugly",
            cfg_scale=0.0,
            width=1024,
            height=1024,
            num_inference_steps=9,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "z_image/z_image.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
