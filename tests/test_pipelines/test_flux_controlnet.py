from ..common.test_case import ImageTestCase
from diffsynth_engine import FluxImagePipeline, FluxControlNet, ControlNetParams, fetch_model
import torch


class TestFluxControlNet(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors")
        cls.pipe = FluxImagePipeline.from_pretrained(model_path).eval()

    def test_union_control(self):
        canny_image = self.get_input_image("canny.png")
        controlnet = FluxControlNet.from_pretrained(
            fetch_model("LiblibAI/FLUX.1-dev-ControlNet-Union-Pro-2.0", path="diffusion_pytorch_model.safetensors"),
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        output_image = self.pipe(
            prompt="A young girl stands gracefully at the edge of a serene beach, her long, flowing hair gently tousled by the sea breeze. She wears a soft, pastel-colored dress that complements the tranquil blues and greens of the coastal scenery. The golden hues of the setting sun cast a warm glow on her face, highlighting her serene expression. The background features a vast, azure ocean with gentle waves lapping at the shore, surrounded by distant cliffs and a clear, cloudless sky. The composition emphasizes the girl's serene presence amidst the natural beauty, with a balanced blend of warm and cool tones.",
            height=canny_image.height,
            width=canny_image.width,
            num_inference_steps=30,
            seed=42,
            controlnet_params=ControlNetParams(
                model=controlnet,
                scale=0.7,
                control_end=1.0,
                image=canny_image,
            ),
        )
        self.assertImageEqualAndSaveFailed(output_image, "flux/flux_union_pro_canny.png")
