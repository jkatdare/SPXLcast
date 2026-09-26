"""Append-only archive of every run: inputs, simulator arguments, outputs and scored headlines.

Yahoo only serves the latest headlines, so a story that is not stored when it is seen is lost. The
archive is the only way to build the history needed to calibrate the news tilt (or to train a
better sentiment model) later. Each run also stores the exact arguments passed to the simulator,
so any logged forecast can be reproduced with the same build (see ``replay``).

    <root>/runs/YYYY-MM-DD/HHMMSSZ.json.gz   one file per run: inputs, simulator arguments, forecast
    <root>/news/YYYY-MM.jsonl                 one line per story, written at its first sighting

The hosted job writes it to the file share (``/data/archive``). The status page does not serve it,
because it holds publishers' headline text.
"""
from __future__ import annotations

import gzip
import json
import os
import platform
from datetime import datetime, timedelta, timezone
from importlib import metadata
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from .config import MODEL_VERSION
from .env import build_id
from .montecarlo import SimulationResult, simulate
from .pipeline import _clean, forecast_to_dict
from .sentiment import _norm_title, is_market_wide

ARRAY_ARGS = ("mu_annual", "sigma_annual")


def _utc(stamp: str) -> datetime:
    t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "yfinance", "vaderSentiment"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


# ---------------------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------------------
def run_record(fc, prices: Optional[List[float]] = None) -> Dict[str, Any]:
    """Everything needed to audit or replay one run."""
    snap = fc.snap
    market_last = {}
    for ticker, df in snap.prices.items():
        s = df["Close"].dropna() if "Close" in df else None
        if s is not None and len(s):
            market_last[ticker] = [str(s.index[-1].date()), float(s.iloc[-1])]
    fred_last = {}
    for sid, s in snap.fred.items():
        s = s.dropna()
        if len(s):
            fred_last[sid] = [str(s.index[-1].date()), float(s.iloc[-1])]
    holdings = None
    if snap.holdings is not None and not snap.holdings.empty:
        holdings = snap.holdings[["symbol", "name", "weight"]].to_dict("records")
    forecast = forecast_to_dict(fc, prices)
    return _clean({
        "run_at": forecast["run_at"],
        "model_version": MODEL_VERSION,
        "build": build_id(),
        "packages": _versions(),
        "data_quality": fc.data_quality(),
        "inputs": {
            "intraday": snap.intraday,
            "fetched_at": {k: v.isoformat() for k, v in snap.fetched_at.items() if v is not None},
            "notes": list(snap.notes),
            "market_last": market_last,     # ticker -> [date, close]
            "fred_last": fred_last,         # series -> [date, value]
            "holdings": holdings,
        },
        "sim_inputs": fc.sim_inputs,
        "rating_sim_inputs": fc.rating_sim_inputs,
        "forecast": forecast,
    })


def write_run(root: str, record: Dict[str, Any]) -> str:
    stamp = _utc(record["run_at"]).astimezone(timezone.utc)
    folder = os.path.join(root, "runs", stamp.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, stamp.strftime("%H%M%SZ") + ".json.gz")
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(record, fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)      # a crash never leaves a truncated record behind
    return path


def load_run(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def replay(record: Dict[str, Any], which: str = "sim_inputs") -> SimulationResult:
    """Re-run the simulation of an archived run (``which="rating_sim_inputs"`` for the untilted run
    behind the rating, when the run had one). Identical to the original with the same build."""
    kwargs = dict(record[which])
    for key in ARRAY_ARGS:
        kwargs[key] = np.asarray(kwargs[key], dtype=float)
    return simulate(**kwargs)


# ---------------------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------------------
def story_key(url: str, title: str, published: str) -> str:
    """Stable identity of a story across runs: its URL, else its title and publication date."""
    if url:
        return url.split("#", 1)[0]
    return f"{_norm_title(title)}|{published[:10]}"


def news_records(sentiment, run_at: str) -> List[Dict[str, Any]]:
    out = []
    for x in getattr(sentiment, "scored", None) or []:
        it = x.item
        published = it.published.isoformat()
        out.append({
            "key": story_key(it.url, it.title, published),
            "first_seen": run_at,
            "published": published,
            "provider": it.provider,
            "url": it.url,
            "title": it.title,
            "summary": it.summary,
            "feeds": sorted(set(x.feeds)),
            "score": round(float(x.score), 4),          # the model's score at first sighting
            "relevance": round(float(x.relevance), 4),  # feed-independent relevance weight
            "market_wide": is_market_wide(it.title),
        })
    return out


def _keys(path: str) -> Set[str]:
    keys: Set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                keys.add(json.loads(line)["key"])
            except (ValueError, KeyError, TypeError):
                continue      # tolerate a torn last line
    return keys


def append_news(root: str, records: Iterable[Dict[str, Any]], build: str = "", model_version: str = "") -> int:
    """Append the stories not archived before. Stories are at most a week old when scored, so
    checking this month's and last month's files is enough. Returns the number added."""
    records = list(records)
    if not records:
        return 0
    seen_at = _utc(records[0]["first_seen"]).astimezone(timezone.utc)
    month = seen_at.strftime("%Y-%m")
    previous = (seen_at.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    folder = os.path.join(root, "news")
    os.makedirs(folder, exist_ok=True)
    seen = _keys(os.path.join(folder, f"{previous}.jsonl")) | _keys(os.path.join(folder, f"{month}.jsonl"))
    added = 0
    with open(os.path.join(folder, f"{month}.jsonl"), "a", encoding="utf-8") as fh:
        for rec in records:
            if rec["key"] in seen:
                continue
            seen.add(rec["key"])
            fh.write(json.dumps({**rec, "build": build, "model_version": model_version}, ensure_ascii=False) + "\n")
            added += 1
    return added


def archive_run(fc, root: str, prices: Optional[List[float]] = None) -> Dict[str, Any]:
    """Archive one run: the run record, then any headlines not seen before."""
    record = run_record(fc, prices)
    path = write_run(root, record)
    added = 0
    if fc.sentiment is not None:
        added = append_news(root, news_records(fc.sentiment, record["run_at"]),
                            record["build"], record["model_version"])
    return {"run": path, "new_stories": added}
