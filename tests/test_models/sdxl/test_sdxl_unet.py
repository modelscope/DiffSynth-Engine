import torch

from diffsynth_engine.models.sdxl.sdxl_unet import SDXLUNet
from tests.common.test_case import ImageTestCase
from diffsynth_engine import fetch_model


class TestSDXLUNet(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("muse/sd_xl_base_1.0", revision="20240425120250", path="sd_xl_base_1.0.safetensors")
        cls.unet = SDXLUNet.from_pretrained(model_path, device="cuda:0", dtype=torch.float16)

    def test_txt2img(self):
        t = self.get_expect_tensor("sdxl/sdxl_unet.safetensors")
        x = t["x"].to(device="cuda:0")
        timestep = t["timesteps"].to(device="cuda:0")
        context = t["context"].to(device="cuda:0")
        y = t["y"].to(device="cuda:0")
        output = self.unet(x, timestep, context, y).to(device="cpu")
        self.assertTensorEqual(output, t["output"], atol=1e-3, rtol=1e-3)
