import cv2
import numpy as np
from PIL import Image


class CannyProcessor:
    def __init__(
        self,
        device,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ):
        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        image = np.array(image.convert("RGB"), dtype=np.uint8)
        output_image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        output_image = Image.fromarray(output_image)
        return output_image
