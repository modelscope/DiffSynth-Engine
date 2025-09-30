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


def load_file(path: str | os.PathLike, device: str = "cpu", need_metadata: bool = False):
    if use_fast_safetensors:
        logger.info(f"FastSafetensors load model from {path}")
        start_time = time.time()
        result = load_safetensors(
            str(path),
            num_threads=int(os.environ.get("FAST_SAFETENSORS_NUM_THREADS", 16)),
            direct_io=(os.environ.get("FAST_SAFETENSORS_DIRECT_IO", "False").upper() == "TRUE"),
        )
        logger.info(f"FastSafetensors Load Model End. Time: {time.time() - start_time:.2f}s")
        state_dict = {k: v.to(device) for k, v in result.items()}

        if need_metadata:
            # FastSafetensors不直接支持metadata，需要用标准safetensors获取
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            return state_dict, metadata
        else:
            return state_dict
    else:
        logger.info(f"Safetensors load model from {path}")
        start_time = time.time()

        if need_metadata:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                state_dict = {k: f.get_tensor(k).to(device) for k in f.keys()}
                metadata = f.metadata()
            logger.info(f"Safetensors Load Model End. Time: {time.time() - start_time:.2f}s")
            return state_dict, metadata
        else:
            result = _load_file(path, device=device)
            logger.info(f"Safetensors Load Model End. Time: {time.time() - start_time:.2f}s")
            return result


save_file = _save_file
