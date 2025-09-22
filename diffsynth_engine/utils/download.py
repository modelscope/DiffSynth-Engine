import os
import shutil
import tqdm
import tempfile
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse
import requests
import glob

from modelscope import snapshot_download
from modelscope.hub.api import HubApi
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.lock import HeartbeatFileLock
from diffsynth_engine.utils.env import DIFFSYNTH_FILELOCK_DIR, DIFFSYNTH_CACHE, MS_HUB_OFFLINE
from diffsynth_engine.utils.constants import MB

logger = logging.get_logger(__name__)


MODEL_SOURCES = ["modelscope", "civitai"]

# Global registry for custom fetch function
_CUSTOM_MODELSCOPE_FETCHER = None


def register_fetch_modelscope_model(fetch_func):
    """
    Register a global custom fetch function for ModelScope models.

    Args:
        fetch_func (callable): Custom fetch function that should accept the same parameters
                               as fetch_modelscope_model and return the model path(s)
    """
    global _CUSTOM_MODELSCOPE_FETCHER
    _CUSTOM_MODELSCOPE_FETCHER = fetch_func
    logger.info("Registered global custom ModelScope fetcher")


def reset_fetch_modelscope_model():
    """
    Reset the global custom fetch function for ModelScope models.
    """
    global _CUSTOM_MODELSCOPE_FETCHER
    _CUSTOM_MODELSCOPE_FETCHER = None
    logger.info("Reset global custom ModelScope fetcher")


def fetch_model(
    model_uri: str,
    revision: Optional[str] = None,
    path: Optional[str | List[str]] = None,
    access_token: Optional[str] = None,
    source: str = "modelscope",
    fetch_safetensors: bool = True,  # TODO: supports other formats like GGUF
) -> str | List[str]:
    if source == "modelscope":
        return fetch_modelscope_model(model_uri, revision, path, access_token, fetch_safetensors)
    if source == "civitai":
        return fetch_civitai_model(model_uri)
    raise ValueError(f'source should be one of {MODEL_SOURCES} but got "{source}"')


def fetch_modelscope_model(
    model_id: str,
    revision: Optional[str] = None,
    path: Optional[str | List[str]] = None,
    access_token: Optional[str] = None,
    fetch_safetensors: bool = True,
) -> str:
    # Check if there's a global custom fetcher registered
    if _CUSTOM_MODELSCOPE_FETCHER is not None:
        logger.info(f"Using global custom fetcher for model: {model_id}")
        return _CUSTOM_MODELSCOPE_FETCHER(model_id, revision, path, access_token, fetch_safetensors)

    lock_file_name = f"modelscope.{model_id.replace('/', '--')}.{revision if revision else '__version'}.lock"
    lock_file_path = os.path.join(DIFFSYNTH_FILELOCK_DIR, lock_file_name)
    ensure_directory_exists(lock_file_path)
    if access_token is not None:
        api = HubApi()
        api.login(access_token)
    with HeartbeatFileLock(lock_file_path):
        directory = os.path.join(DIFFSYNTH_CACHE, "modelscope", model_id, revision if revision else "__version")
        dirpath = snapshot_download(
            model_id, revision=revision, local_dir=directory, allow_patterns=path, local_files_only=MS_HUB_OFFLINE
        )

    if isinstance(path, str):
        path = glob.glob(os.path.join(dirpath, path))
        path = path[0] if len(path) == 1 else path
    elif isinstance(path, list):
        path = [os.path.join(dirpath, p) for p in path]
    else:
        path = dirpath

    if isinstance(path, str) and os.path.isdir(path) and fetch_safetensors:
        return _fetch_safetensors(path)
    return path


def fetch_civitai_model(model_url: str) -> str:
    """
    https://civitai.com/models/4384?modelVersionId=128713
    https://civitai.com/models/4384
    https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16
    """
    try:
        requests.get("https://civitai.com", timeout=3)
    except Exception:
        raise ValueError("Failed to access Civitai, please check your network connection.")

    parsed_url = urlparse(model_url)
    if "/models/" in parsed_url.path:
        if parsed_url.query:
            model_version_id = parsed_url.query.split("=")[-1]
            download_url = f"https://civitai.com/api/download/models/{model_version_id}"
        else:
            model_id = parsed_url.path.split("/")[-1]
            result = requests.get(
                f"https://civitai.com/api/v1/models/{model_id}", headers={"Content-Type": "application/json"}
            ).json()
            model_version_id = result["modelVersions"][0]["id"]
            download_url = result["modelVersions"][0]["downloadUrl"]
    elif "/api/download/models/" in parsed_url.path:
        model_version_id = parsed_url.path.split("/")[-1]
        download_url = model_url
    else:
        raise ValueError("Invalid Civitai model URL")
    CIVITAI_CACHE = os.path.join(DIFFSYNTH_CACHE, "civitai")
    ensure_directory_exists(CIVITAI_CACHE)
    filename = requests.get(f"https://civitai.com/api/v1/model-versions/{model_version_id}").json()["files"][0]["name"]
    filepath = os.path.join(CIVITAI_CACHE, filename)

    if os.path.exists(filepath):
        logger.info(f"File {filename} already exists, file path: {filepath}")
        return filepath
    response = requests.get(
        download_url, stream=True, timeout=4 * 60 * 60, headers={"Content-Type": "application/json"}
    )  # 4h
    if response.status_code >= 400:
        raise RuntimeError(f"Download {filename} failed, please check the model url")
    total_bytes = int(response.headers.get("content-length", 0))
    bar = tqdm.tqdm(desc=f"Download {filename}", total=total_bytes, unit="B", unit_divisor=1024, unit_scale=True)
    with tempfile.NamedTemporaryFile() as f:
        for chunk in response.iter_content(chunk_size=16 * MB):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        shutil.copy(f.name, filepath)
        bar.close()
    # 提示文件下载完成，并显示文件路径
    logger.info(f"Download {filename} completed, file path: {filepath}")
    return filepath


def ensure_directory_exists(filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


def _fetch_safetensors(dirpath: str) -> str | List[str]:
    all_safetensors = []
    for filename in os.listdir(dirpath):
        if filename.endswith(".safetensors"):
            all_safetensors.append(os.path.join(dirpath, filename))
    if len(all_safetensors) == 0:
        logger.error(f"No safetensors file found in {dirpath}")
        return dirpath
    elif len(all_safetensors) == 1:
        all_safetensors = all_safetensors[0]
        logger.info(f"Fetch safetensors file {all_safetensors}")
    else:
        logger.info(f"Fetch safetensors files {all_safetensors}")
    return all_safetensors
