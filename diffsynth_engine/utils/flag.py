import importlib
import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


# 无损
FLASH_ATTN_3_AVAILABLE = importlib.util.find_spec("flash_attn_interface") is not None
if FLASH_ATTN_3_AVAILABLE:
    logger.info("Flash attention 3 is available")
else:
    logger.info("Flash attention 3 is not available")

FLASH_ATTN_2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
if FLASH_ATTN_2_AVAILABLE:
    logger.info("Flash attention 2 is available")
else:
    logger.info("Flash attention 2 is not available")

XFORMERS_AVAILABLE = importlib.util.find_spec("xformers") is not None
if XFORMERS_AVAILABLE:
    logger.info("XFormers is available")
else:
    logger.info("XFormers is not available")

SDPA_AVAILABLE = hasattr(torch.nn.functional, "scaled_dot_product_attention")
if SDPA_AVAILABLE:
    logger.info("Torch SDPA is available")
else:
    logger.info("Torch SDPA is not available")


# 有损
SAGE_ATTN_AVAILABLE = importlib.util.find_spec("sageattention") is not None
if SAGE_ATTN_AVAILABLE:
    logger.info("Sage attention is available")
else:
    logger.info("Sage attention is not available")

SPARGE_ATTN_AVAILABLE = importlib.util.find_spec("spas_sage_attn") is not None
if SPARGE_ATTN_AVAILABLE:
    logger.info("Sparge attention is available")
else:
    logger.info("Sparge attention is not available")
