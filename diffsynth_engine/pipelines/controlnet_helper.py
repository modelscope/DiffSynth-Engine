import torch
import torch.nn as nn
from typing import List, Union, Optional
from PIL import Image
from dataclasses import dataclass

ImageType = Union[Image.Image, torch.Tensor, List[Image.Image], List[torch.Tensor]]


@dataclass
class ControlNetParams:
    image: ImageType
    scale: float = 1.0
    model: Optional[nn.Module] = None
    mask: Optional[ImageType] = None
    control_start: float = 0
    control_end: float = 1
    processor_name: Optional[str] = None  # only used for sdxl controlnet union now


def accumulate(result, new_item):
    if result is None:
        return new_item
    for i, item in enumerate(new_item):
        result[i] += item
    return result
