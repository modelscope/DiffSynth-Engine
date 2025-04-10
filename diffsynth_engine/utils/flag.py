import importlib
import logging

logger = logging.getLogger(__name__)
# 无损
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
XFORMERS_AVAILABLE = False
SDPA_AVAILABLE = False
# 有损
SAGE_ATTN_AVAILABLE = False
SPARGE_ATTN_AVAILABLE = False


try:
    FLASH_ATTN_3_AVAILABLE = importlib.util.find_spec("flash_attn_interface") is not None
    logger.info("Flash attention 3 is available")
except ModuleNotFoundError:
    logger.info("Flash attention 3 is not available")

try:
    FLASH_ATTN_2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
    logger.info("Flash attention 2 is available")
except ModuleNotFoundError:
    logger.info("Flash attention 2 is not available")

try:
    XFORMERS_AVAILABLE = importlib.util.find_spec("xformers") is not None
    logger.info("XFormers is available")
except ModuleNotFoundError:
    logger.info("XFormers is not available")

try:
    SDPA_AVAILABLE = importlib.util.find_spec("torch.nn.functional.scaled_dot_product_attention") is not None
    logger.info("Torch SDPA is available")
except ModuleNotFoundError:
    logger.info("Torch SDPA is not available")


try:
    SAGE_ATTN_AVAILABLE = importlib.util.find_spec("sageattention") is not None
    logger.info("Sage attention is available")
except ModuleNotFoundError:
    logger.info("Sage attention is not available")

try:
    SPARGE_ATTN_AVAILABLE = importlib.util.find_spec("spas_sage_attn") is not None
    logger.info("Sparge attention is available")
except ModuleNotFoundError:
    logger.info("Sparge attention is not available")
