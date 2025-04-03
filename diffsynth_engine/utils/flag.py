import logging
logger = logging.getLogger(__name__)
# 无损
FLASH_ATTN_3_AVAILABLE=False
FLASH_ATTN_2_AVAILABLE=False
XFORMERS_AVAILABLE=False
SDPA_AVAILABLE=False
# 有损
SAGE_ATTN_AVAILABLE=False
SPARGE_ATTN_AVAILABLE=False



try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
    logger.info("Flash attention 3 is available")
except ModuleNotFoundError:
    logger.info("Flash attention 3 is not available")

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
    logger.info("Flash attention 2 is available")
except ModuleNotFoundError:
    logger.info("Flash attention 2 is not available")

try:
    import xformers
    XFORMERS_AVAILABLE = True
    logger.info("XFormers is available")
except ModuleNotFoundError:
    logger.info("XFormers is not available")

try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
    logger.info("Torch SDPA is available")
except ModuleNotFoundError:
    logger.info("Torch SDPA is not available")


try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
    logger.info("Sage attention is available")
except ModuleNotFoundError:
    logger.info("Sage attention is not available")

try:
    from spas_sage_attn import spas_sage2_attn_meansim_cuda
    SPARGE_ATTN_AVAILABLE = True
    logger.info("Sparge attention is available")
except ModuleNotFoundError:
    logger.info("Sparge attention is not available")
