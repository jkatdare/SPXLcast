"""Per-minute live layer for the hosted status page.

The scheduled job rebuilds the full forecast hourly (a Monte Carlo run needs data, a minute of CPU
and FRED). Between runs this module keeps the page current every minute of the regular session
without re-simulating: the simulated distribution is a distribution of *returns* from the base
spot, so when SPXL trades at a new price every simulated price scales by ``spot / base_spot``.
Fixed dollar levels (the price checks, the buy-limit ladder) are re-read off the fine percentile
grids that ``forecast_to_dict`` stores for the terminal price, the path minimum and the path
maximum. The rating is a function of returns only, so it is unchanged until the next full run.

Every minute during the session the loop fetches the latest quote, writes ``output/live.json``
and appends one row to ``logs/spot_log.csv`` (a minute-by-minute price record for later
backtests). Outside the session it idles.

    python -m spxlcast serve --root /data --live          # status page + minute loop
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import NY_TZ, ny_now, session_state

log = logging.getLogger(__name__)

# Fine percentile grid stored with every forecast: dense enough to interpolate any price level.
GRID_PERCENTILES: Tuple[float, ...] = (0.1, 0.5) + tuple(float(p) for p in range(1, 100)) + (99.5, 99.9)

SPOT_LOG_COLUMNS = ["ts_utc", "ny_time", "spot", "session", "base_run_at", "base_spot", "rating", "score"]


# ---------------------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------------------
def fetch_spot(ticker: str) -> Optional[float]:
    """Latest trade price from Yahoo (fast_info, falling back to the last 1-minute bar)."""
    import yfinance as yf  # imported here so the pure re-pricing functions stay import-light

    t = yf.Ticker(ticker)
    try:
        fi = t.fast_info
        price = fi["lastPrice"] if "lastPrice" in fi else getattr(fi, "last_price", None)
        if price is not None and np.isfinite(float(price)) and float(price) > 0:
            return float(price)
    except Exception as exc:  # noqa: BLE001 - any Yahoo hiccup just means "try the other way"
        log.debug("fast_info failed for %s: %s", ticker, exc)
    try:
        h = t.history(period="1d", interval="1m", auto_adjust=False, actions=False)
        if h is not None and not h.empty:
            price = float(h["Close"].dropna().iloc[-1])
            if np.isfinite(price) and price > 0:
                return price
    except Exception as exc:  # noqa: BLE001
        log.debug("1m history failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------------------
# Re-pricing off the stored grids
# ---------------------------------------------------------------------------------------
def _cdf(prices: List[float], pcts: List[float], x: float) -> float:
    """Percent of simulated values at or below ``x`` given a (price, percentile) grid, 0..100."""
    if not prices:
        return float("nan")
    if x <= prices[0]:
        return 0.0
    if x >= prices[-1]:
        return 100.0
    return float(np.interp(x, prices, pcts))


def reprice(base: Dict, spot: float, now: Optional[datetime] = None) -> Dict:
    """Restate a stored forecast (``forecast_to_dict`` output) at a new spot price."""
    base_spot = float(base["spot"])
    if not (np.isfinite(spot) and spot > 0 and np.isfinite(base_spot) and base_spot > 0):
        raise ValueError("spot and base spot must be positive finite prices")
    ratio = spot / base_spot
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ny = pd.Timestamp(now_utc).tz_convert(NY_TZ)
    _, is_open = session_state(ny)

    levels: List[float] = sorted(float(p) for p in (base.get("price_lookup") or {}).keys())
    horizons: Dict[str, Dict] = {}
    for h, f in base["forecast"].items():
        grid = f.get("grid")
        out: Dict = {
            "horizon": int(f["horizon"]),
            "quantile_prices": {k: float(v) * ratio for k, v in f["quantile_prices"].items()},
            # return-based statistics do not move with the spot
            "median_return": f["median_return"], "mean_return": f["mean_return"],
            "p_positive": f["p_positive"], "p_beat_rf": f["p_beat_rf"],
            "p_drawdown_20": f["p_drawdown_20"], "p_up_20": f["p_up_20"],
        }
        if grid:
            pcts = list(grid["percentiles"])
            term = [v * ratio for v in grid["terminal"]]
            lo = [v * ratio for v in grid["path_min"]]
            hi = [v * ratio for v in grid["path_max"]]
            out["price_lookup"] = [{
                "price": p, "vs_spot": p / spot - 1.0,
                "percentile": _cdf(term, pcts, p),
                "p_end_above": 1.0 - _cdf(term, pcts, p) / 100.0,
                "p_touch_below": _cdf(lo, pcts, p) / 100.0,       # P(path min <= p)
                "p_touch_above": 1.0 - _cdf(hi, pcts, p) / 100.0,  # P(path max >= p)
            } for p in levels]
        horizons[str(h)] = out

    ladder = {h: [{**row, "price": float(row["price"]) * ratio} for row in rows]
              for h, rows in (base.get("limit_ladder") or {}).items()}
    r = base["rating"]
    return {
        "asof": now_utc.isoformat(timespec="seconds"),
        "asof_ny": ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "session_open": bool(is_open),
        "etf": base.get("etf"),
        "spot": float(spot),
        "base_spot": base_spot,
        "base_run_at": base.get("run_at"),
        "base_spot_status": base.get("spot_status"),
        "change_vs_base": ratio - 1.0,
        "rating": {"label": r["label"], "conviction": r["conviction"], "score": r["score"],
                   "score_se": r.get("score_se"), "horizon_days": r["horizon_days"]},
        "horizons": horizons,
        "limit_ladder": ladder,
    }


# ---------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------
def load_base(root: str) -> Optional[Dict]:
    path = os.path.join(root, "output", "forecast.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("no usable base forecast at %s: %s", path, exc)
        return None


def write_live(root: str, live: Dict) -> str:
    path = os.path.join(root, "output", "live.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(live, fh, indent=1, allow_nan=False)
    os.replace(tmp, path)   # readers never see a half-written file
    return path


def append_spot_log(root: str, live: Dict) -> str:
    path = os.path.join(root, "logs", "spot_log.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SPOT_LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({
            "ts_utc": live["asof"], "ny_time": live["asof_ny"], "spot": f"{live['spot']:.4f}",
            "session": "open" if live["session_open"] else "closed",
            "base_run_at": live["base_run_at"], "base_spot": f"{live['base_spot']:.4f}",
            "rating": live["rating"]["label"], "score": f"{live['rating']['score']:.4f}",
        })
    return path


def refresh_once(root: str, fetch: Callable[[str], Optional[float]] = fetch_spot,
                 now: Optional[datetime] = None, log_row: bool = True) -> Optional[Dict]:
    """One tick: quote -> live.json (+ a spot_log row). Returns the live dict, or None if skipped."""
    base = load_base(root)
    if base is None:
        return None
    spot = fetch(base.get("etf") or "SPXL")
    if spot is None:
        log.warning("no quote for %s this minute", base.get("etf"))
        return None
    live = reprice(base, spot, now=now)
    write_live(root, live)
    if log_row:
        append_spot_log(root, live)
    return live


def run_loop(root: str, interval: float = 60.0, stop: Optional[threading.Event] = None,
             fetch: Callable[[str], Optional[float]] = fetch_spot, idle_interval: float = 300.0) -> None:
    """Tick every ``interval`` seconds while the regular session is open; idle otherwise."""
    stop = stop or threading.Event()
    log.info("live loop: every %.0fs during the session", interval)
    while not stop.is_set():
        _, is_open = session_state(ny_now())
        started = time.monotonic()
        if is_open:
            try:
                refresh_once(root, fetch=fetch)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive whatever Yahoo returns
                log.warning("live refresh failed: %s", exc)
            wait = interval
        else:
            wait = idle_interval
        stop.wait(max(0.0, wait - (time.monotonic() - started)))


def start_background(root: str, interval: float = 60.0) -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=run_loop, args=(root, interval, stop), name="spxlcast-live", daemon=True).start()
    return stop
