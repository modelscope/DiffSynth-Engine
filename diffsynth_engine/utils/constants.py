import os
from pathlib import Path

HOME = Path.home()
DIFFSYNTH_CACHE = os.environ.get("DIFFSYNTH_CACHE", os.path.join(HOME, ".cache", "diffsynth"))
DIFFSYNTH_FILELOCK_DIR = os.environ.get(
    "DIFFSYNTH_FILELOCK_DIR", os.path.join(HOME, ".cache", "diffsynth", "filelocks")
)

CONFIG_NAME = "config.json"
MODEL_INDEX_NAME = "model_index.json"
DIFFUSION_SAFETENSORS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
DIFFUSION_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
SAFETENSORS_WEIGHTS_NAME = "model.safetensors"

IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", 600))
