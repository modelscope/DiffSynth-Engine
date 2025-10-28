from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from typing import List, Union, Optional
from PIL import Image

ImageType = Union[Image.Image, torch.Tensor, List[Image.Image], List[torch.Tensor]]


# FLUX ControlType
class ControlType(Enum):
    normal = "normal"
    bfl_control = "bfl_control"
    bfl_fill = "bfl_fill"
    bfl_kontext = "bfl_kontext"

    def get_in_channel(self):
        if self in [ControlType.normal, ControlType.bfl_kontext]:
            return 64
        elif self == ControlType.bfl_control:
            return 128
        elif self == ControlType.bfl_fill:
            return 384


@dataclass
class ControlNetParams:
    image: ImageType
    scale: float = 1.0
    model: Optional[nn.Module] = None
    mask: Optional[ImageType] = None
    control_start: float = 0
    control_end: float = 1
    processor_name: Optional[str] = None  # only used for sdxl controlnet union now


class QwenImageControlType(Enum):
    eligen = "eligen"
    in_context = "in_context"


@dataclass
class QwenImageControlNetParams:
    image: ImageType
    model: str
    control_type: QwenImageControlType
    scale: float = 1.0
