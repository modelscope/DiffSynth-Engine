import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def compute_normalized_ssim(image1: Image.Image, image2: Image.Image):
    image1_arr = np.array(image1)
    image2_arr = np.array(image2)
    if image1.mode == "RGB" or image1.mode == "RGBA":
        channel_axis = 2
    else:
        channel_axis = None
    ssim = structural_similarity(image1_arr, image2_arr, channel_axis=channel_axis)
    ssim_normalized = (ssim + 1) / 2

    return ssim_normalized
