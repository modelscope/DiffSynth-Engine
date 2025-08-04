import os
import random
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from typing import Dict, List

from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.utils.gguf import load_gguf_checkpoint


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


def load_model_checkpoint(
    checkpoint_path: str | List[str], device: str = "cpu", dtype: torch.dtype = torch.float16
) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_path, str):
        checkpoint_path = [checkpoint_path]
    state_dict = {}
    for path in checkpoint_path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is not a file")
        elif path.endswith(".safetensors"):
            state_dict_ = load_file(path, device=device)
            for key, value in state_dict_.items():
                state_dict[key] = value.to(dtype)

        elif path.endswith(".gguf"):
            state_dict.update(**load_gguf_checkpoint(path, device=device, dtype=dtype))
        else:
            raise ValueError(f"{path} is not a .safetensors or .gguf file")
    return state_dict
