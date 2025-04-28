from ..common.test_case import ImageTestCase
from diffsynth_engine.pipelines import FluxImagePipeline, FluxModelConfig
from diffsynth_engine import fetch_model


class TestFLUXImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors")
        cls.pipe = FluxImagePipeline.from_pretrained(model_path).eval()

    def test_txt2img(self):
        image = self.pipe(
            prompt="A cat holding a sign that says hello world",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_txt2img.png", threshold=0.99)

    def test_fused_lora(self):
        lora_model_path = fetch_model("MAILAND/Merjic-Maria", revision="v1.0", path="12.safetensors")
        self.pipe.load_loras([(lora_model_path, 0.8)], fused=True)
        image = self.pipe(
            prompt="1 girl, maria",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.pipe.unload_loras()
        self.assertImageEqualAndSaveFailed(image, "flux/flux_lora.png", threshold=0.99)

    def test_unfused_lora(self):
        lora_model_path = fetch_model("MAILAND/Merjic-Maria", revision="v1.0", path="12.safetensors")
        self.pipe.load_loras([(lora_model_path, 0.8)])
        image = self.pipe(
            prompt="1 girl, maria",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.pipe.unload_loras()
        self.assertImageEqualAndSaveFailed(image, "flux/flux_lora.png", threshold=0.98)


class TestFLUXGGUF(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("city96/FLUX.1-dev-gguf", path="flux1-dev-Q4_K_S.gguf")
        t5_path = fetch_model("city96/t5-v1_1-xxl-encoder-gguf", path="t5-v1_1-xxl-encoder-Q4_K_S.gguf")
        config = FluxModelConfig(
            dit_path=model_path,
            t5_path=t5_path,
        )
        cls.pipe = FluxImagePipeline.from_pretrained(config).eval()

    def test_gguf_inference(self):
        image = self.pipe(
            prompt="A cat holding a sign that says hello world",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_txt2img.png", threshold=0.85)
