import os
import time
import diffsynth_engine.utils.logging as logging
from safetensors.torch import save_file as _save_file

logger = logging.get_logger(__name__)
try:
    from fast_safetensors import load_safetensors

    use_fast_safetensors = True
except ImportError:
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

        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k).to(device) for k in f.keys()}
            if need_metadata:
                metadata = f.metadata()

        logger.info(f"Safetensors Load Model End. Time: {time.time() - start_time:.2f}s")

        if need_metadata:
            return state_dict, metadata
        else:
            return state_dict


save_file = _save_file
