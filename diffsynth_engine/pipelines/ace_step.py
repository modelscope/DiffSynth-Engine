import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms.functional import pil_to_tensor
from typing import Callable, List, Optional
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.configs import ACEStepPipelineConfig
from diffsynth_engine.models.ace_step.ace_dit import ACEStepDiT
from diffsynth_engine.models.ace_step.vae.music_dcae import MusicDCAE
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.utils import logging


logger = logging.get_logger(__name__)


class ACEStepMusicPipeline(BasePipeline):
    def __init__(
        self,
        config: ACEStepPipelineConfig,
        dit: ACEStepDiT,
        vae: MusicDCAE,
    ):
        pass