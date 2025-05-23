import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize, resize, to_pil_image


from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.onnx import OnnxModel


MODEL_ID = "muse/depth_anything_detector"
REVISION = "20240801180053"
MODEL_NAME = "depth_anything_detector.onnx"


class DepthProcessor:
    def __init__(self, device):
        self.device = device
        model_path = fetch_model(model_uri=MODEL_ID, revision=REVISION, path=MODEL_NAME)
        self.model = OnnxModel(model_path, device=self.device)

    def _image_preprocess(self, image: Image.Image) -> np.ndarray:
        image = resize(image, (518, 518))
        image = to_tensor(image)
        image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.unsqueeze(0).contiguous()
        return image.numpy()

    def __call__(self, img: Image.Image) -> Image.Image:
        image = img
        w, h = image.size
        image = self._image_preprocess(image)
        depth = self.model(image)
        depth = torch.from_numpy(depth)
        depth: torch.Tensor = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)
        depth = depth.squeeze(0).squeeze(0)
        # 确保张量在 [0, 255] 范围内，并转换为 uint8 类型
        depth = torch.clamp(depth, 0, 255).byte()
        # 转换为 PIL Image 对象
        depth = to_pil_image(depth)
        return depth
