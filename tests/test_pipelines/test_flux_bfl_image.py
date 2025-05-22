import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine.pipelines import FluxImagePipeline
from diffsynth_engine.pipelines.flux_image import ControlType, ControlNetParams
from diffsynth_engine.processor.canny_processor import CannyProcessor
from diffsynth_engine.processor.depth_processor import DepthProcessor
from diffsynth_engine import fetch_model
from PIL import Image


class TestFLUXBFLImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_canny_txt2img(self) -> None:
        self.canny_processor = CannyProcessor("cuda:0")
        self.canny_model_path = fetch_model(
            "AI-ModelScope/FLUX.1-Canny-dev", revision="master", path="flux1-canny-dev.safetensors"
        )

        self.pipe = FluxImagePipeline.from_pretrained(self.canny_model_path, control_type=ControlType.bfl_canny).eval()
        width, height = 1024, 1024
        image = self.get_input_image("test_image.png").resize((width, height), Image.LANCZOS)
        control_image = self.canny_processor(image)
        controlnet_params = ControlNetParams(
            model=None,
            control_type=ControlType.bfl_canny,
            scale=1.0,
            image=[control_image],
        )
        image = self.pipe(
            prompt="a beautiful girl with green hair",
            width=width,
            height=height,
            num_inference_steps=50,
            seed=self.seed,
            controlnet_params=[controlnet_params],
            flux_guidance_scale=30,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_bfl_canny.png", threshold=0.99)

    def test_depth_txt2img(self):
        self.depth_processor = DepthProcessor("cuda:0")
        self.depth_model_path = fetch_model(
            "AI-ModelScope/FLUX.1-Depth-dev", revision="master", path="flux1-depth-dev.safetensors"
        )

        self.pipe = FluxImagePipeline.from_pretrained(self.depth_model_path, control_type=ControlType.bfl_depth).eval()
        width, height = 1024, 1024
        image = self.get_input_image("robot.png").resize((width, height), Image.LANCZOS)
        control_image = self.depth_processor(image)
        controlnet_params = ControlNetParams(
            model=None,
            control_type=ControlType.bfl_depth,
            scale=1.0,
            image=[control_image],
        )
        image = self.pipe(
            prompt="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
            width=width,
            height=height,
            num_inference_steps=30,
            seed=self.seed,
            controlnet_params=[controlnet_params],
            flux_guidance_scale=10,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_bfl_depth.png", threshold=0.99)

    def test_fill_txt2img(self):
        self.fill_model_path = fetch_model(
            "AI-ModelScope/FLUX.1-Fill-dev", revision="master", path="flux1-fill-dev.safetensors"
        )

        self.pipe = FluxImagePipeline.from_pretrained(self.fill_model_path, control_type=ControlType.bfl_fill).eval()
        width, height = 1232, 1632
        image = self.get_input_image("cup.png").resize((width, height), Image.LANCZOS)
        mask = self.get_input_image("cup_mask.png").resize((width, height), Image.LANCZOS)
        controlnet_params = ControlNetParams(
            model=None,
            control_type=ControlType.bfl_fill,
            scale=1.0,
            image=[image],
            mask=[mask],
        )
        image = self.pipe(
            prompt="a white paper cup",
            width=width,
            height=height,
            num_inference_steps=50,
            seed=self.seed,
            controlnet_params=[controlnet_params],
            flux_guidance_scale=30,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_bfl_fill.png", threshold=0.99)


if __name__ == "__main__":
    unittest.main()
