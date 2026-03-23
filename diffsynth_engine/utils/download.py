import os
from pathlib import Path
from typing import List, Optional

from huggingface_hub import login
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as ms_snapshot_download
from modelscope.hub.api import HubApi

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.constants import DIFFSYNTH_CACHE, DIFFSYNTH_FILELOCK_DIR
from diffsynth_engine.utils.lock import HeartbeatFileLock

logger = logging.get_logger(__name__)


MODEL_SOURCES = ["huggingface", "modelscope"]


def fetch_model(
    model_id: str,
    revision: Optional[str] = None,
    path: Optional[str | List[str]] = None,
    access_token: Optional[str] = None,
    local_files_only: bool = False,
    source: str = "modelscope",
) -> str | List[str]:
    if source == "huggingface":
        return fetch_huggingface_model(model_id, revision, path, access_token, local_files_only)
    if source == "modelscope":
        return fetch_modelscope_model(model_id, revision, path, access_token, local_files_only)
    raise ValueError(f'source should be one of {MODEL_SOURCES} but got "{source}"')


def fetch_huggingface_model(
    model_id: str,
    revision: Optional[str] = None,
    path: Optional[str | List[str]] = None,
    access_token: Optional[str] = None,
    local_files_only: bool = False,
) -> str | List[str]:
    lock_file_name = f"huggingface.{model_id.replace('/', '--')}{'.' + revision if revision else ''}.lock"
    lock_file_path = os.path.join(DIFFSYNTH_FILELOCK_DIR, lock_file_name)
    ensure_directory_exists(lock_file_path)
    if access_token is not None:
        login(access_token)
    with HeartbeatFileLock(lock_file_path):
        local_dir = os.path.join(DIFFSYNTH_CACHE, "huggingface", model_id)
        local_dir = os.path.join(local_dir, revision) if revision else local_dir
        return hf_snapshot_download(
            model_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=path,
            local_files_only=local_files_only,
        )


def fetch_modelscope_model(
    model_id: str,
    revision: Optional[str] = None,
    path: Optional[str | List[str]] = None,
    access_token: Optional[str] = None,
    local_files_only: bool = False,
) -> str:
    lock_file_name = f"modelscope.{model_id.replace('/', '--')}{'.' + revision if revision else ''}.lock"
    lock_file_path = os.path.join(DIFFSYNTH_FILELOCK_DIR, lock_file_name)
    ensure_directory_exists(lock_file_path)
    if access_token is not None:
        api = HubApi()
        api.login(access_token)
    with HeartbeatFileLock(lock_file_path):
        local_dir = os.path.join(DIFFSYNTH_CACHE, "modelscope", model_id)
        local_dir = os.path.join(local_dir, revision) if revision else local_dir
        return ms_snapshot_download(
            model_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=path,
            local_files_only=local_files_only,
        )


def ensure_directory_exists(filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
