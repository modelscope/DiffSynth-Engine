from ..common.test_case import ImageTestCase
from diffsynth_engine import fetch_model
from diffsynth_engine.pipelines.sdxl_image import SDXLImagePipeline


class TestSDXLImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/sd_xl_base_1.0", revision="20240425120250", path="sd_xl_base_1.0.safetensors")
        cls.pipe = SDXLImagePipeline.from_pretrained(model_path)

    def test_txt2img(self):
        image = self.pipe(
            prompt="a beautiful girl", width=1024, height=1024, num_inference_steps=20, seed=42, clip_skip=2
        )

        self.assertImageEqualAndSaveFailed(image, "sdxl/sdxl_txt2img.png", threshold=0.99)

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
            clip_skip=2,
        )
        self.assertImageEqualAndSaveFailed(image, "sdxl/sdxl_inpainting.png", threshold=0.99)

    def test_unfused_lora(self):
        lora_model_path = fetch_model("MusePublic/89_lora_SD_XL", revision="532", path="532.safetensors")
        self.pipe.load_loras([(lora_model_path, 0.8)])
        image = self.pipe(
            prompt="a beautiful girl, chibi",
            width=1024,
            height=1024,
            num_inference_steps=20,
            seed=42,
            clip_skip=2,
        )
        self.pipe.unload_loras()
        self.assertImageEqualAndSaveFailed(image, "sdxl/sdxl_lora.png", threshold=0.99)

    def test_fused_lora(self):
        lora_model_path = fetch_model("MusePublic/89_lora_SD_XL", revision="532", path="532.safetensors")
        self.pipe.load_loras([(lora_model_path, 0.8)], fused=True)
        image = self.pipe(
            prompt="a beautiful girl, chibi",
            width=1024,
            height=1024,
            num_inference_steps=20,
            seed=42,
            clip_skip=2,
        )
        self.pipe.unload_loras()
        self.assertImageEqualAndSaveFailed(image, "sdxl/sdxl_lora.png", threshold=0.99)
