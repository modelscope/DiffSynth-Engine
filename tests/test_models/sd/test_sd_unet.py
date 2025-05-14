import unittest
import torch

from diffsynth_engine.models.sd.sd_unet import SDUNet
from tests.common.test_case import ImageTestCase
from diffsynth_engine import fetch_model


class TestSDUNet(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model(
            "muse/v1-5-pruned-emaonly", revision="20240118200020", path="v1-5-pruned-emaonly.safetensors"
        )
        cls.unet = SDUNet.from_pretrained(model_path, device="cuda:0", dtype=torch.float16)

    def test_txt2img(self):
        t = self.get_expect_tensor("sd/sd_unet.safetensors")
        x = t["x"].to(device="cuda:0")
        timestep = t["timestep"].to(device="cuda:0")
        context = t["context"].to(device="cuda:0")
        output = self.unet(x, timestep, context).to(device="cpu")
        self.assertTensorEqual(output, t["output"], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
