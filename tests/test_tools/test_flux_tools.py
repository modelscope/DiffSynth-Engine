import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine import (
    fetch_model,
    FluxInpaintingTool,
    FluxOutpaintingTool,
    FluxReduxRefTool,
    FluxIPAdapterRefTool,
)


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

    def test_ipadapter_ref(self):
        reference_tool = FluxIPAdapterRefTool(
            fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors")
        )
        input_image = self.get_input_image("wukong_1024_1024.png")
        output_image = reference_tool(
            ref_image=input_image,
            ref_scale=0.4,
            prompt="A rugged man, dressed in a weathered leather jacket and dusty boots, rides a powerful chestnut horse through the rolling hills. ",
        )
        self.assertImageEqualAndSaveFailed(output_image, "flux/flux_ipadapter_ref.png")

    def test_redux_ref(self):
        reference_tool = FluxReduxRefTool(
            fetch_model("muse/flux-with-vae", revision="20240902173035", path="flux1-dev-with-vae.safetensors"),
            load_text_encoder=False,
        )
        input_image = self.get_input_image("robot.png")
        output_image = reference_tool(ref_image=input_image, ref_scale=1.0, num_inference_steps=50, seed=0)
        self.assertImageEqualAndSaveFailed(output_image, "flux/flux_redux_ref.png")


if __name__ == "__main__":
    unittest.main()
