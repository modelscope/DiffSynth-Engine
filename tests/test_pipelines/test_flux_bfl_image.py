import unittest

from tests.common.test_case import ImageTestCase
from diffsynth_engine.pipelines import FluxImagePipeline
from diffsynth_engine.pipelines.flux_image import ControlType, ControlNetParams
from diffsynth_engine.processor.canny_processor import CannyProcessor
from diffsynth_engine.processor.depth_processor import DepthProcessor

from diffsynth_engine import fetch_model


class TestFLUXBFLCannyImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        cls.canny_processor = CannyProcessor("cuda:0")
        cls.canny_model_path = fetch_model(
            "AI-ModelScope/FLUX.1-Canny-dev", revision="master", path="flux1-canny-dev.safetensors"
        )
        cls.pipe = FluxImagePipeline.from_pretrained(cls.canny_model_path, control_type=ControlType.bfl_control).eval()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.pipe

    def test_canny_txt2img(self) -> None:
        width, height = 1024, 1024
        image = self.get_input_image("test_image.png")
        control_image = self.canny_processor(image)
        controlnet_params = ControlNetParams(
            scale=1.0,
            image=control_image,
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


class TestFLUXBFLDepthImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        cls.depth_processor = DepthProcessor("cuda:0")
        cls.depth_model_path = fetch_model(
            "AI-ModelScope/FLUX.1-Depth-dev", revision="master", path="flux1-depth-dev.safetensors"
        )
        cls.pipe = FluxImagePipeline.from_pretrained(cls.depth_model_path, control_type=ControlType.bfl_control).eval()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.pipe

    def test_depth_txt2img(self):
        width, height = 1024, 1024
        image = self.get_input_image("robot.png")
        control_image = self.depth_processor(image)
        controlnet_params = ControlNetParams(
            scale=1.0,
            image=control_image,
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


class TestFLUXBFLFillImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fill_model_path = fetch_model(
            "AI-ModelScope/FLUX.1-Fill-dev", revision="master", path="flux1-fill-dev.safetensors"
        )
        cls.pipe = FluxImagePipeline.from_pretrained(cls.fill_model_path, control_type=ControlType.bfl_fill).eval()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.pipe

    def test_fill_txt2img(self):
        width, height = 1232, 1632
        image = self.get_input_image("cup.png")
        mask = self.get_input_image("cup_mask.png")
        controlnet_params = ControlNetParams(
            scale=1.0,
            image=image,
            mask=mask,
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
