import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine import fetch_model
from diffsynth_engine.pipelines.sd_image import SDImagePipeline


class TestSDImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model(
            "muse/v1-5-pruned-emaonly", revision="20240118200020", path="v1-5-pruned-emaonly.safetensors"
        )
        cls.pipe = SDImagePipeline.from_pretrained(model_path)

    def test_txt2img(self):
        image = self.pipe(prompt="beautiful girl", width=512, height=512, num_inference_steps=20, seed=42)
        self.assertImageEqualAndSaveFailed(image, "sd/sd_txt2img.png", threshold=0.99)

    def test_inpainting(self):
        image = self.pipe(
            prompt="a beautiful girl with green hair",
            input_image=self.get_input_image("test_image.png"),
            mask_image=self.get_input_image("mask_image.png"),
            denoising_strength=0.8,
            width=1024,
            height=1024,
            num_inference_steps=20,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "sd/sd_inpainting.png", threshold=0.99)

    def test_unfused_lora(self):
        lora_model_path = fetch_model("MusePublic/148_lora_SD_1_5", revision="765", path="765.safetensors")
        self.pipe.load_loras([(lora_model_path, 0.8)])
        image = self.pipe(
            prompt="a girl, drawing",
            width=512,
            height=512,
            num_inference_steps=20,
            seed=42,
        )
        self.pipe.unload_loras()
        self.assertImageEqualAndSaveFailed(image, "sd/sd_lora.png", threshold=0.99)

    def test_fused_lora(self):
        lora_model_path = fetch_model("MusePublic/148_lora_SD_1_5", revision="765", path="765.safetensors")
        self.pipe.load_loras([(lora_model_path, 0.8)], fused=True)
        image = self.pipe(
            prompt="a girl, drawing",
            width=512,
            height=512,
            num_inference_steps=20,
            seed=42,
        )
        self.pipe.unload_loras()
        self.assertImageEqualAndSaveFailed(image, "sd/sd_lora.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
