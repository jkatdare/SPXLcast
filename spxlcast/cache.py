"""Tiny pickle cache with a time-to-live, so repeated runs do not hammer Yahoo/FRED.

Rules that matter:
* Empty results (``{}``, ``[]``, ``None``) are never stored and never served: a failed or offline
  fetch must not poison the cache for the rest of the TTL.
* Writes are atomic (temp file + ``os.replace``) so a concurrent reader never sees a torn pickle.
* ``read_enabled=False`` forces fresh fetches but still stores them, so a ``--refresh`` run leaves
  a fresh cache behind instead of a stale one.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import time
from typing import Any, Callable, Optional


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


class Cache:
    def __init__(self, directory: str, enabled: bool = True, read_enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        self.read_enabled = read_enabled
        if enabled:
            os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        safe = "".join(ch if ch.isalnum() else "_" for ch in key)[:40]
        return os.path.join(self.directory, f"{safe}_{digest}.pkl")

    def mtime(self, key: str) -> Optional[float]:
        """Unix time the entry was written, or None if absent."""
        path = self._path(key)
        return os.path.getmtime(path) if self.enabled and os.path.exists(path) else None

    def get(self, key: str, ttl_hours: float) -> Optional[Any]:
        if not self.enabled or not self.read_enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours > ttl_hours:
            return None
        try:
            with open(path, "rb") as fh:
                value = pickle.load(fh)
        except Exception:
            return None
        return None if _is_empty(value) else value

    def put(self, key: str, value: Any) -> None:
        if not self.enabled or _is_empty(value):
            return
        path = self._path(key)
        try:
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".tmp_", suffix=".pkl")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(value, fh)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)  # type: ignore[name-defined]
            except Exception:
                pass

    def get_or_fetch(self, key: str, ttl_hours: float, fetch: Callable[[], Any]) -> Any:
        hit = self.get(key, ttl_hours)
        if hit is not None:
            return hit
        value = fetch()
        self.put(key, value)   # no-op for empty results
        return value

    def prune(self, max_age_hours: float) -> int:
        """Delete entries older than ``max_age_hours`` (archives are kept). Returns the number removed."""
        if not self.enabled or not os.path.isdir(self.directory):
            return 0
        removed = 0
        cutoff = time.time() - max_age_hours * 3600.0
        for name in os.listdir(self.directory):
            path = os.path.join(self.directory, name)
            try:
                if name.endswith(".pkl") and not name.startswith("archive") and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
        return removed
