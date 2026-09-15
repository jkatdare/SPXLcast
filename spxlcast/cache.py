"""Tiny pickle cache with a time-to-live, so repeated runs do not hammer Yahoo/FRED."""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from typing import Any, Callable, Optional


class Cache:
    def __init__(self, directory: str, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        if enabled:
            os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        safe = "".join(ch if ch.isalnum() else "_" for ch in key)[:40]
        return os.path.join(self.directory, f"{safe}_{digest}.pkl")

    def get(self, key: str, ttl_hours: float) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours > ttl_hours:
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            with open(self._path(key), "wb") as fh:
                pickle.dump(value, fh)
        except Exception:
            pass

    def get_or_fetch(self, key: str, ttl_hours: float, fetch: Callable[[], Any]) -> Any:
        hit = self.get(key, ttl_hours)
        if hit is not None:
            return hit
        value = fetch()
        if value is not None:
            self.put(key, value)
        return value
