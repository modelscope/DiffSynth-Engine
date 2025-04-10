import os

try:
    from fast_safetensors import load_safetensors

    use_fast_safetensors = True
except ImportError:
    from safetensors.torch import load_file as _load_file

    use_fast_safetensors = False


def load_file(path: str, device: str = "cpu"):
    if use_fast_safetensors:
        return load_safetensors(path, num_threads=os.environ.get("FAST_SAFETENSORS_NUM_THREADS", 16))
    else:
        return _load_file(path, device=device)
