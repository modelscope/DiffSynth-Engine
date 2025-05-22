import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import repeat

from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.onnx import OnnxModel
from diffsynth_engine.utils.channels import HWC3


MODEL_ID = "muse/depth_anything_detector"
REVISION = "20240801180053"
MODEL_NAME = "depth_anything_detector.onnx"


class DepthProcessor:
    def __init__(self, device):
        self.device = device
        model_path = fetch_model(model_uri=MODEL_ID, revision=REVISION, path=MODEL_NAME)
        self.model = OnnxModel(model_path, device=self.device)

    def _image_preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize((518, 518))
        image = np.asarray(image, dtype=np.uint8)
        image = HWC3(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))  # h w c -> c h w
        image = np.ascontiguousarray(image).astype(np.float32)
        image = image[None]  # c h w -> b c h w
        return image

    def __call__(self, img: Image.Image) -> torch.Tensor:
        image = img
        w, h = image.size
        image = self._image_preprocess(image)
        depth = self.model(image)
        depth = torch.from_numpy(depth)
        depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)
        depth = repeat(depth, "b 1 ... -> b 3 ...")
        depth = depth / 127.5 - 1.0
        return depth


if __name__ == "__main__":
    depth_processor = DepthProcessor("cuda")
    img = Image.open("/home/admin/workspace/aop_lab/app_source/DiffSynth-Engine/tests/data/input/cup.png")
    depth = depth_processor(img)
    print(depth.shape)
