import torch
import numpy as np
from PIL import Image


def tensor_to_image(t: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """
    Convert a tensor to an image.
    """
    # b c h w
    if t.dim() == 4:
        t = t[0]
    t = t.permute(1, 2, 0).float().cpu().numpy()
    if denormalize:
        t = (t + 1) / 2
    t = (t.clip(0, 1) * 255).astype(np.uint8)

    if t.shape[2] == 1:
        mode = "L"
        t = t[..., 0]
    elif t.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    return Image.fromarray(t, mode=mode)
