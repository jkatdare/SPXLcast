"""Minimal .env loader (no third-party dependency).

Reads ``KEY = value`` lines from a ``.env`` file in the current directory or the project root and
puts them into ``os.environ`` without overriding variables that are already set. Secrets such as
``FRED_API_KEY`` therefore live outside the code and outside git (``.env`` is git-ignored).
"""
from __future__ import annotations

import os
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
