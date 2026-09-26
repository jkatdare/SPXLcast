"""Minimal .env loader (no third-party dependency).

Reads ``KEY = value`` lines from a ``.env`` file in the current directory or the project root and
puts them into ``os.environ`` without overriding variables that are already set. Secrets such as
``FRED_API_KEY`` therefore live outside the code and outside git (``.env`` is git-ignored).
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional


def parse_env(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            out[key] = value
    return out


def load_dotenv(path: Optional[str] = None) -> Dict[str, str]:
    """Load the first .env found (explicit path, cwd, project root). Returns what was loaded."""
    candidates = [Path(path)] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for candidate in candidates:
        if candidate.is_file():
            loaded = parse_env(candidate.read_text(encoding="utf-8"))
            for k, v in loaded.items():
                os.environ.setdefault(k, v)
            return loaded
    return {}


def fred_api_key() -> Optional[str]:
    load_dotenv()
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key or None


@functools.lru_cache(maxsize=1)
def build_id() -> str:
    """The code behind this run, logged with every forecast: ``SPXLCAST_BUILD`` (the git commit the
    container image was built from), else the checkout's commit (``-dirty`` with uncommitted
    changes), else ``unknown``. Shortened to 12 characters."""
    env = os.environ.get("SPXLCAST_BUILD", "").strip()
    if env and env != "unknown":
        return env[:12]
    here = Path(__file__).resolve().parent

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=5)

    try:
        head = git("rev-parse", "--short=12", "HEAD")
        if head.returncode == 0 and head.stdout.strip():
            dirty = git("diff", "--quiet", "HEAD", "--").returncode == 1
            return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"
