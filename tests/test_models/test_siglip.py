import torch
from diffsynth_engine.models.components.siglip import SiglipImageEncoder
from diffsynth_engine import fetch_model
from tests.common.test_case import ImageTestCase


class TestSiglipImageEncoder(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/google-siglip-so400m-patch14-384", path="model.safetensors")
        cls.image_encoder = SiglipImageEncoder.from_pretrained(model_path, device="cuda:0", dtype=torch.bfloat16)

    def test_siglip_image_encoder(self):
        image = self.get_input_image("test_image.png")
        expect = self.get_expect_tensor("test_siglip_image_encoder.safetensors")
        result = self.image_encoder(image)
        self.assertTensorEqual(result.cpu(), expect["output"])
