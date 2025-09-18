import os
from pathlib import Path

HOME = Path.home()

DIFFSYNTH_CACHE = os.environ.get("DIFFSYNTH_CACHE", os.path.join(HOME, ".cache", "diffsynth"))

DIFFSYNTH_FILELOCK_DIR = os.environ.get(
    "DIFFSYNTH_FILELOCK_DIR", os.path.join(HOME, ".cache", "diffsynth", "filelocks")
)

MS_HUB_OFFLINE = os.getenv("MS_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")
