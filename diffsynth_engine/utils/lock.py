import math
import threading
from types import TracebackType
from typing import Optional

from flufl.lock import Lock

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class HeartbeatFileLock:
    def __init__(self, lock_file_path: str, heartbeat_interval: float = 10):
        self.lock_file_path = lock_file_path
        self.heartbeat_interval = heartbeat_interval
        self.lifetime = math.ceil(heartbeat_interval + 1)
        self.heartbeat_thread = None
        self.stop_event = threading.Event()
        self.lock = None

    def _heartbeat(self):
        while not self.stop_event.is_set():
            self.lock.refresh(lifetime=self.lifetime)
            self.stop_event.wait(self.heartbeat_interval - 1)

    def acquire(self):
        self.lock = Lock(self.lock_file_path, lifetime=self.lifetime)
        self.lock.lock()

        self.heartbeat_thread = threading.Thread(target=self._heartbeat)
        self.heartbeat_thread.start()

    def release(self):
        if self.lock is not None:
            self.lock.unlock(unconditionally=True)
        self._release()

    def _release(self):
        if self.heartbeat_thread is not None:
            self.stop_event.set()
            self.heartbeat_thread.join()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ):
        self._release()

    def __del__(self):
        self.release()
