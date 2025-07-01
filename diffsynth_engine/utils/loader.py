import os
import time
import diffsynth_engine.utils.logging as logging
from safetensors.torch import save_file as _save_file

logger = logging.get_logger(__name__)
try:
    from fast_safetensors import load_safetensors

    use_fast_safetensors = True
except ImportError:
    from safetensors.torch import load_file as _load_file

    use_fast_safetensors = False


def load_file(path: str | os.PathLike, device: str = "cpu"):
    if use_fast_safetensors:
        logger.info(f"FastSafetensors load model from {path}")
        start_time = time.time()
        result = load_safetensors(
            str(path),
            num_threads=int(os.environ.get("FAST_SAFETENSORS_NUM_THREADS", 16)),
            direct_io=(os.environ.get("FAST_SAFETENSORS_DIRECT_IO", "False").upper() == "TRUE"),
        )
        logger.info(f"FastSafetensors Load Model End. Time: {time.time() - start_time:.2f}s")
        return {k: v.to(device) for k, v in result.items()}
    else:
        logger.info(f"Safetensors load model from {path}")
        start_time = time.time()
        result = _load_file(path, device=device)
        logger.info(f"Safetensors Load Model End. Time: {time.time() - start_time:.2f}s")
        return result


save_file = _save_file
