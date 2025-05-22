import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat

from diffsynth_engine.utils.channels import HWC3


class CannyProcessor:
    def __init__(
        self,
        device,
        low_threshold: int = 50,
        high_threshold: int = 200,
    ):
        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, image: Image.Image) -> Image.Image:
        image = np.asarray(image, dtype=np.uint8)
        image = HWC3(image)
        output_image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        return output_image

    def encode(self, canny: Image.Image) -> torch.Tensor:
        # Convert back to torch tensor and reshape
        canny = torch.from_numpy(canny).float() / 127.5 - 1.0
        canny = rearrange(canny, "h w -> 1 1 h w")
        canny = repeat(canny, "b 1 ... -> b 3 ...")

        return canny.to(self.device)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.encode(self.process(image))
