from typing import List

import numpy as np
import torch
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


def compute_video_ms_ssim(
    input_frames: List[Image.Image],
    expect_frames: List[Image.Image],
) -> float:
    """Compute the mean MS-SSIM score between two frame sequences.

    Each frame is converted to a ``[1, C, H, W]`` float tensor in ``[0, 1]``
    and scored with ``MultiScaleStructuralSimilarityIndexMeasure``.  The
    returned value is the average MS-SSIM across all frame pairs.
    """
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    scores: List[float] = []
    for pred_frame, target_frame in zip(input_frames, expect_frames):
        pred_array = np.array(pred_frame).astype(np.float32)
        target_array = np.array(target_frame).astype(np.float32)

        # Normalize to [0, 1]: only divide by 255 when the data is in uint8 range
        if pred_array.max() > 1.0:
            pred_array = pred_array / 255.0
        if target_array.max() > 1.0:
            target_array = target_array / 255.0

        pred_tensor = torch.from_numpy(pred_array)
        target_tensor = torch.from_numpy(target_array)

        # [H, W, C] -> [1, C, H, W]
        pred_tensor = pred_tensor.permute(2, 0, 1).unsqueeze(0)
        target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0)

        score = ms_ssim_metric(pred_tensor, target_tensor)
        scores.append(score.item())

    return float(np.mean(scores))
