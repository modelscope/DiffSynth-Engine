import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine import Hunyuan3DShapePipeline, fetch_model


class TestHunyuan3DShape(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/Hunyuan3d-2.1-Shape", path="dit.safetensors")
        cls.pipe = Hunyuan3DShapePipeline.from_pretrained(model_path)

    def test_hunyuan3d_shape(self):
        image = self.get_input_image("demo.png")
        mesh = self.pipe(
            image,
            num_inference_steps=50,
            guidance_scale=5.0,
            seed=42,
        )
        mesh.export("test_mesh.glb")


if __name__ == "__main__":
    unittest.main()
