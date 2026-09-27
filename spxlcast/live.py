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
backtests). Outside the session it fetches nothing: it marks live.json closed after the bell and
restates it once on each new full run (at that run's own spot, or at the last quote of the session
when that is later), so the page always matches the latest report. ``spot_since`` records when the
quote last moved, so the health check can tell a frozen feed from a live one.

    python -m spxlcast serve --root /data --live          # status page + minute loop
"""
from __future__ import annotations

import csv
import io
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


def _utc(stamp) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _run_id(stamp) -> Optional[str]:
    """A run's time the way the track record and the archive name it: '2026-09-22T21:40:21Z' (floored)."""
    t = _utc(stamp)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else stamp


def reprice(base: Dict, spot: float, now: Optional[datetime] = None,
            session_open: Optional[bool] = None) -> Dict:
    """Restate a stored forecast (``forecast_to_dict`` output) at a new spot price, quoted at ``now``
    (default: the clock). ``session_open`` defaults to the session state at ``now``."""
    base_spot = float(base["spot"])
    if not (np.isfinite(spot) and spot > 0 and np.isfinite(base_spot) and base_spot > 0):
        raise ValueError("spot and base spot must be positive finite prices")
    ratio = spot / base_spot
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ny = pd.Timestamp(now_utc).tz_convert(NY_TZ)
    is_open = session_state(ny)[1] if session_open is None else session_open

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
            # every path starts at the spot; clamping keeps that point mass exactly at the quote, which
            # base_spot * ratio can miss by one ulp (a level at the quote must read 100% touched)
            lo = [min(v * ratio, spot) for v in grid["path_min"]]
            hi = [max(v * ratio, spot) for v in grid["path_max"]]
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
        "base_run_at": _run_id(base.get("run_at")),     # joins forecast_log.run_at as a string
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
    tmp = f"{path}.{os.urandom(6).hex()}.tmp"    # never shared: two loops overlap during a deploy
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            json.dump(live, fh, indent=1, allow_nan=False)
        for attempt in range(5):
            try:
                os.replace(tmp, path)   # readers never see a half-written file
                break
            except PermissionError:     # Windows: a reader holds live.json open for a moment
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def append_spot_log(root: str, live: Dict) -> str:
    path = os.path.join(root, "logs", "spot_log.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = io.StringIO()
    csv.DictWriter(row, fieldnames=SPOT_LOG_COLUMNS).writerow({
        "ts_utc": live["asof"], "ny_time": live["asof_ny"], "spot": f"{live['spot']:.4f}",
        "session": "open" if live["session_open"] else "closed",
        "base_run_at": live["base_run_at"], "base_spot": f"{live['base_spot']:.4f}",
        "rating": live["rating"]["label"], "score": f"{live['rating']['score']:.4f}",
    })
    with open(path, "ab") as fh:        # creates it when absent
        pass
    prefix = b""
    if os.path.getsize(path) == 0:
        # The header is written at offset 0, not appended: every loop that finds the file absent or
        # emptied writes the same bytes to the same place, so loops starting together leave one header.
        with open(path, "r+b") as fh:
            fh.write((",".join(SPOT_LOG_COLUMNS) + "\r\n").encode("utf-8"))
    else:
        with open(path, "rb") as fh:    # a torn last row gets its line end, so the new row is not glued on
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        prefix = b"" if last == b"\n" else b"\n" if last == b"\r" else b"\r\n"
    with open(path, "ab") as fh:
        fh.write(prefix + row.getvalue().encode("utf-8"))
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
        cur = _load_live(root)
        if cur is not None and _run_id(cur.get("base_run_at")) == _run_id(base.get("run_at")):
            return None
        # a full run landed during a quote outage: show it at its own price, not the last run's rating
        live = reprice(base, float(base["spot"]), now=_utc(base.get("run_at")))
        write_live(root, live)
        return live
    live = reprice(base, spot, now=now)
    # asof is only when this tick wrote: a feed that keeps answering with an old price shows in spot_since
    cur = _load_live(root)
    same = cur is not None and cur.get("session_open") is True and cur.get("spot") == live["spot"]
    since = (cur.get("spot_since") or cur.get("asof")) if same else None
    live["spot_since"] = since if isinstance(since, str) else live["asof"]
    try:
        write_live(root, live)
    finally:                            # the price record must not depend on the page file
        if log_row:
            append_spot_log(root, live)
    return live


def _load_live(root: str) -> Optional[Dict]:
    try:
        with open(os.path.join(root, "output", "live.json"), encoding="utf-8") as fh:
            cur = json.load(fh)
    except (OSError, ValueError, RecursionError):
        return None
    return cur if isinstance(cur, dict) else None


def settle_closed(root: str) -> Optional[Dict]:
    """Outside the session: mark live.json closed at its last quote, or restate it at the spot and
    time of a newer full run (at the loop's last quote when that is later than the run, as for a run
    that started before the bell and finished after it). Fetches nothing and logs no spot_log row.
    None when nothing changed."""
    if not os.path.exists(os.path.join(root, "output", "forecast.json")):
        return None                     # no run yet: nothing to show, and nothing to warn about every minute
    base = load_base(root)
    if base is None:
        return None
    cur = _load_live(root)
    # both sides normalised: a live.json written before run ids were normalised still names this run
    last = cur if cur is not None and _run_id(cur.get("base_run_at")) == _run_id(base.get("run_at")) else None
    if last is not None and last.get("session_open") is False:
        return None
    run_at, cur_at = _utc(base.get("run_at")), _utc(cur.get("asof")) if cur is not None else None
    if last is None and run_at and cur_at and cur_at > run_at:
        last = cur                      # the previous run's last quote is later than this run's own price
    try:
        spot, at = float(last["spot"]), _utc(last["asof"])
    except (KeyError, TypeError, ValueError):   # no quote on this run yet: its own spot, as of the run
        spot, at = float(base["spot"]), run_at
    live = reprice(base, spot, now=at, session_open=False)
    write_live(root, live)
    return live


def _seconds_to_open(now: pd.Timestamp) -> float:
    """Seconds from ``now`` (New York) to the next 9:30 ET, on any day."""
    day = now.date() if (now.hour, now.minute) < (9, 30) else (now + pd.Timedelta(days=1)).date()
    return (pd.Timestamp(f"{day} 09:30", tz=NY_TZ) - now).total_seconds()


def run_loop(root: str, interval: float = 60.0, stop: Optional[threading.Event] = None,
             fetch: Callable[[str], Optional[float]] = fetch_spot, idle_interval: float = 60.0) -> None:
    """Tick every ``interval`` seconds while the regular session is open; otherwise settle live.json
    every ``idle_interval`` seconds and wake at the open."""
    stop = stop or threading.Event()
    log.info("live loop: every %.0fs during the session", interval)
    while not stop.is_set():
        now = ny_now()
        _, is_open = session_state(now)
        started = time.monotonic()
        try:
            if is_open:
                refresh_once(root, fetch=fetch)
            else:
                settle_closed(root)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive whatever Yahoo returns
            log.warning("live refresh failed: %s", exc)
        wait = interval if is_open else min(idle_interval, _seconds_to_open(now) + 1.0)
        stop.wait(max(0.0, wait - (time.monotonic() - started)))


def start_background(root: str, interval: float = 60.0) -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=run_loop, args=(root, interval, stop), name="spxlcast-live", daemon=True).start()
    return stop
