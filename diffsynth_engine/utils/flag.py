import importlib
import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


def check_module_available(module_path: str, module_name: str = None) -> bool:
    try:
        available = importlib.util.find_spec(module_path) is not None
    except (ModuleNotFoundError, AttributeError, ValueError):
        available = False

    if module_name:
        if available:
            logger.info(f"{module_name} is available")
        else:
            logger.info(f"{module_name} is not available")

    return available


# 无损
FLASH_ATTN_4_AVAILABLE = check_module_available("flash_attn.cute.interface", "Flash attention 4")
FLASH_ATTN_3_AVAILABLE = check_module_available("flash_attn_interface", "Flash attention 3")
FLASH_ATTN_2_AVAILABLE = check_module_available("flash_attn", "Flash attention 2")
XFORMERS_AVAILABLE = check_module_available("xformers", "XFormers")
AITER_AVAILABLE = check_module_available("aiter", "Aiter")

SDPA_AVAILABLE = hasattr(torch.nn.functional, "scaled_dot_product_attention")
if SDPA_AVAILABLE:
    logger.info("Torch SDPA is available")
else:
    logger.info("Torch SDPA is not available")


# 有损
SAGE_ATTN_AVAILABLE = check_module_available("sageattention", "Sage attention")
SPARGE_ATTN_AVAILABLE = check_module_available("spas_sage_attn", "Sparge attention")
VIDEO_SPARSE_ATTN_AVAILABLE = check_module_available("vsa", "Video sparse attention")

NUNCHAKU_AVAILABLE = check_module_available("nunchaku", "Nunchaku")
NUNCHAKU_IMPORT_ERROR = None
if not NUNCHAKU_AVAILABLE:
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