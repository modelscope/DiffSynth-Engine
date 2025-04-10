import os
import random
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def make_deterministic(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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
