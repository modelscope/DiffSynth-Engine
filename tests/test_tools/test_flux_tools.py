from ..common.test_case import ImageTestCase
from diffsynth_engine import fetch_model, FluxInpaintingTool, FluxOutpaintingTool


class TestFluxTools(ImageTestCase):
    def test_inpainting(self):
        inpainting_tool = FluxInpaintingTool(
            fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors")
        )
        mask_image = self.get_input_image("mask_image.png")
        input_image = self.get_input_image("test_image.png")
        output_image = inpainting_tool(
            image=input_image,
            mask=mask_image,
            prompt="a beautiful girl with green hair",
            strength=0.9,
        )
        self.assertImageEqualAndSaveFailed(output_image, "flux/flux_inpainting.png")

    def test_outpainting(self):
        outpainting_tool = FluxOutpaintingTool(
            fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors")
        )
        input_image = self.get_input_image("test_image.png")
        output_image = outpainting_tool(
            image=input_image,
            prompt="blue sky",
            scale=2.0,
            strength=0.9,
        )
        self.assertImageEqualAndSaveFailed(output_image, "flux/flux_outpainting.png")
