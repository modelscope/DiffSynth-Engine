import importlib
import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


# 无损
FLASH_ATTN_4_AVAILABLE = importlib.util.find_spec("flash_attn.cute.interface") is not None
if FLASH_ATTN_4_AVAILABLE:
    logger.info("Flash attention 4 is available")
else:
    logger.info("Flash attention 4 is not available")

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

AITER_AVAILABLE = importlib.util.find_spec("aiter") is not None
if AITER_AVAILABLE:
    logger.info("Aiter is available")
else:
    logger.info("Aiter is not available")

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

VIDEO_SPARSE_ATTN_AVAILABLE = importlib.util.find_spec("vsa") is not None
if VIDEO_SPARSE_ATTN_AVAILABLE:
    logger.info("Video sparse attention is available")
else:
    logger.info("Video sparse attention is not available")

NUNCHAKU_AVAILABLE = importlib.util.find_spec("nunchaku") is not None
NUNCHAKU_IMPORT_ERROR = None
if NUNCHAKU_AVAILABLE:
    logger.info("Nunchaku is available")
else:
    logger.info("Nunchaku is not available")
    import sys
    torch_version = getattr(torch, "__version__", "unknown")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    NUNCHAKU_IMPORT_ERROR = (
        "\n\n"
        "ERROR: This model requires the 'nunchaku' library for quantized inference, but it is not installed.\n"
        "'nunchaku' is not available on PyPI and must be installed manually.\n\n"
        "Please follow these steps:\n"
        "1. Visit the nunchaku releases page: https://github.com/nunchaku-tech/nunchaku/releases\n"
        "2. Find the wheel (.whl) file that matches your environment:\n"
        f"   - PyTorch version: {torch_version}\n"
        f"   - Python version: {python_version}\n"
        f"   - Operating System: {sys.platform}\n"
        "3. Copy the URL of the correct wheel file.\n"
        "4. Install it using pip, for example:\n"
        "   pip install nunchaku @ https://.../your_specific_nunchaku_file.whl\n"
    )