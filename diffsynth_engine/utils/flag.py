import importlib
import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


def check_module_available(module_path: str) -> bool:
    try:
        return importlib.util.find_spec(module_path) is not None
    except (ModuleNotFoundError, AttributeError, ValueError):
        return False


# 无损
FLASH_ATTN_4_AVAILABLE = check_module_available("flash_attn.cute.interface")
if FLASH_ATTN_4_AVAILABLE:
    logger.info("Flash attention 4 is available")
else:
    logger.info("Flash attention 4 is not available")

FLASH_ATTN_3_AVAILABLE = check_module_available("flash_attn_interface")
if FLASH_ATTN_3_AVAILABLE:
    logger.info("Flash attention 3 is available")
else:
    logger.info("Flash attention 3 is not available")

FLASH_ATTN_2_AVAILABLE = check_module_available("flash_attn")
if FLASH_ATTN_2_AVAILABLE:
    logger.info("Flash attention 2 is available")
else:
    logger.info("Flash attention 2 is not available")

XFORMERS_AVAILABLE = check_module_available("xformers")
if XFORMERS_AVAILABLE:
    logger.info("XFormers is available")
else:
    logger.info("XFormers is not available")

SDPA_AVAILABLE = hasattr(torch.nn.functional, "scaled_dot_product_attention")
if SDPA_AVAILABLE:
    logger.info("Torch SDPA is available")
else:
    logger.info("Torch SDPA is not available")

AITER_AVAILABLE = check_module_available("aiter")
if AITER_AVAILABLE:
    logger.info("Aiter is available")
else:
    logger.info("Aiter is not available")

# 有损
SAGE_ATTN_AVAILABLE = check_module_available("sageattention")
if SAGE_ATTN_AVAILABLE:
    logger.info("Sage attention is available")
else:
    logger.info("Sage attention is not available")

SPARGE_ATTN_AVAILABLE = check_module_available("spas_sage_attn")
if SPARGE_ATTN_AVAILABLE:
    logger.info("Sparge attention is available")
else:
    logger.info("Sparge attention is not available")

VIDEO_SPARSE_ATTN_AVAILABLE = check_module_available("vsa")
if VIDEO_SPARSE_ATTN_AVAILABLE:
    logger.info("Video sparse attention is available")
else:
    logger.info("Video sparse attention is not available")

NUNCHAKU_AVAILABLE = check_module_available("nunchaku")
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