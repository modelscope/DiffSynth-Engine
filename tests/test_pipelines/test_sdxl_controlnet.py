import unittest
import torch

from tests.common.test_case import ImageTestCase
from diffsynth_engine import fetch_model, SDXLImagePipeline, SDXLControlNetUnion, ControlNetParams


class TestSDXLImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/sd_xl_base_1.0", revision="20240425120250", path="sd_xl_base_1.0.safetensors")
        cls.pipe = SDXLImagePipeline.from_pretrained(model_path)

    def test_canny(self):
        canny_image = self.get_input_image("canny.png")
        controlnet = SDXLControlNetUnion.from_pretrained(
            fetch_model("AI-ModelScope/controlnet-union-sdxl-1.0-promax", path="diffusion_pytorch_model.safetensors"),
            device="cuda:0",
            dtype=torch.float16,
        )
        output_image = self.pipe(
            prompt="A young girl stands gracefully at the edge of a serene beach, her long, flowing hair gently tousled by the sea breeze. She wears a soft, pastel-colored dress that complements the tranquil blues and greens of the coastal scenery. The golden hues of the setting sun cast a warm glow on her face, highlighting her serene expression. The background features a vast, azure ocean with gentle waves lapping at the shore, surrounded by distant cliffs and a clear, cloudless sky. The composition emphasizes the girl's serene presence amidst the natural beauty, with a balanced blend of warm and cool tones.",
            height=canny_image.height,
            width=canny_image.width,
            num_inference_steps=30,
            seed=42,
            controlnet_params=ControlNetParams(
                model=controlnet,
                scale=1.0,
                control_end=1.0,
                image=canny_image,
                processor_name="canny",
            ),
        )
        self.assertImageEqualAndSaveFailed(output_image, "flux/flux_union_pro_canny.png", threshold=0.7)


if __name__ == "__main__":
    unittest.main()
