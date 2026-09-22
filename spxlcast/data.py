"""Data access: Yahoo Finance (prices, fund info, holdings, news) and optional FRED macro series.

Everything is cached on disk (see cache.py). Every fetch is defensive: a missing ticker or a
timed-out FRED call degrades to ``None`` and a note in ``MarketSnapshot.notes`` instead of an error.
Failed or empty fetches are never cached.
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .cache import Cache
from .config import FRED_SERIES, MARKET_TICKERS, Config
from .env import fred_api_key

log = logging.getLogger(__name__)
NY_TZ = "America/New_York"


def _match_tz(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Make a timestamp comparable with ``index`` (tz-naive vs tz-aware inputs must not crash)."""
    ts = pd.Timestamp(ts)
    tz = getattr(index, "tz", None)
    if tz is None:
        return ts.tz_convert(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


@dataclass
class NewsItem:
    ticker: str
    title: str
    summary: str
    published: datetime  # UTC, tz-aware
    provider: str
    url: str


@dataclass
class MarketSnapshot:
    asof: datetime                                   # run time (UTC)
    prices: Dict[str, pd.DataFrame] = field(default_factory=dict)  # ticker -> OHLCV, auto-adjusted
    infos: Dict[str, dict] = field(default_factory=dict)
    holdings: Optional[pd.DataFrame] = None          # columns: symbol, name, weight
    equity_stats: Dict[str, float] = field(default_factory=dict)  # yield-style ratios from SPY fund data
    fred: Dict[str, pd.Series] = field(default_factory=dict)
    news: List[NewsItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    calendar: Optional[pd.DatetimeIndex] = None      # trading sessions of the index ETF
    intraday: bool = False                           # last price bar is a live, partial session
    fetched_at: Dict[str, datetime] = field(default_factory=dict)   # dataset -> UTC fetch time
    max_stale_sessions: int = 5

    # ---- convenience accessors -------------------------------------------------------
    def close(self, ticker: str) -> Optional[pd.Series]:
        df = self.prices.get(ticker)
        if df is None or df.empty or "Close" not in df:
            return None
        s = df["Close"].dropna()
        return s if len(s) else None

    def last(self, ticker: str) -> Optional[float]:
        s = self.close(ticker)
        return float(s.iloc[-1]) if s is not None else None

    def last_date(self, ticker: str) -> Optional[pd.Timestamp]:
        s = self.close(ticker)
        return s.index[-1] if s is not None else None

    def _sessions_between(self, earlier: pd.Timestamp, later: pd.Timestamp) -> int:
        cal = self.calendar
        if cal is None:
            return 0
        earlier, later = _match_tz(earlier, cal), _match_tz(later, cal)
        return int(cal.searchsorted(later) - cal.searchsorted(earlier))

    def fresh_last(self, ticker: str) -> Optional[float]:
        """Last close, or None if the series is stale versus the trading calendar."""
        s = self.close(ticker)
        if s is None:
            return None
        if self.calendar is not None and self._sessions_between(s.index[-1], self.calendar[-1]) > self.max_stale_sessions:
            return None
        return float(s.iloc[-1])

    def _reference(self, ticker: str, days: int) -> Optional[Tuple[float, float]]:
        """(last, value ``days`` sessions before the series' own latest observation), counted on the
        trading calendar. None when the latest observation is stale versus the calendar, or when the
        series has no observation close enough to the reference session (a hole in the history)."""
        s = self.close(ticker)
        if s is None or days < 1:
            return None
        cal = self.calendar if self.calendar is not None else s.index
        last_date = _match_tz(s.index[-1], cal)
        if self.calendar is not None and self._sessions_between(last_date, cal[-1]) > self.max_stale_sessions:
            return None
        pos = int(cal.searchsorted(last_date, side="right")) - 1   # calendar slot of the latest observation
        if pos - days < 0:
            return None
        ref_date = _match_tz(cal[pos - days], s.index)
        obs = s[s.index <= ref_date]
        if obs.empty or _match_tz(obs.index[-1], cal) == last_date:
            return None
        tolerance = min(3, days - 1)                                # exact match for 1D, a few sessions for longer windows
        if self._sessions_between(obs.index[-1], _match_tz(ref_date, cal)) > tolerance:
            return None
        return float(s.iloc[-1]), float(obs.iloc[-1])

    def change(self, ticker: str, days: int) -> Optional[float]:
        """Fractional change in the close over the last ``days`` trading sessions."""
        ref = self._reference(ticker, days)
        return None if ref is None or ref[1] == 0 else ref[0] / ref[1] - 1.0

    def diff(self, ticker: str, days: int) -> Optional[float]:
        """Absolute change over ``days`` trading sessions (for yields quoted in %)."""
        ref = self._reference(ticker, days)
        return None if ref is None else ref[0] - ref[1]

    def fred_last(self, series_id: str) -> Optional[float]:
        s = self.fred.get(series_id)
        if s is None or s.empty:
            return None
        return float(s.dropna().iloc[-1])

    def info(self, ticker: str) -> dict:
        return self.infos.get(ticker) or {}


# ---------------------------------------------------------------------------------------
# Trading-session helpers
# ---------------------------------------------------------------------------------------
def ny_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def session_state(now: Optional[pd.Timestamp] = None) -> Tuple[str, bool]:
    """(New York date, is the regular session open right now)."""
    now = now if now is not None else ny_now()
    t = now.time()
    is_open = now.weekday() < 5 and (t.hour, t.minute) >= (9, 30) and t.hour < 16
    return str(now.date()), is_open


# ---------------------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------------------
def _tidy_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(NY_TZ).tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"]) if "Close" in df else df


def fetch_prices(tickers: List[str], period: str, cache: Cache, ttl_hours: float,
                 required: Tuple[str, ...] = ()) -> Tuple[Dict[str, pd.DataFrame], Optional[float]]:
    """Download histories. The cache key carries the NY session date and open/closed state so a
    price fetched during the session is never served as the close after the bell. Returns the
    data and the Unix time it was fetched (cache mtime)."""
    ny_date, is_open = session_state()
    key = f"prices:{period}:{','.join(sorted(tickers))}:{ny_date}:{'open' if is_open else 'closed'}"

    def _download() -> Optional[Dict[str, pd.DataFrame]]:
        out: Dict[str, pd.DataFrame] = {}
        try:
            raw = yf.download(
                tickers, period=period, group_by="ticker", auto_adjust=True,
                threads=True, progress=False, actions=False,
            )
        except Exception as exc:  # pragma: no cover - network
            log.debug("yf.download failed: %s", exc)
            raw = None
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                present = set(raw.columns.get_level_values(0))
                for t in tickers:
                    if t in present:
                        sub = raw[t].dropna(how="all")
                        if not sub.empty:
                            out[t] = _tidy_history(sub)
            else:  # single ticker, flat columns
                out[tickers[0]] = _tidy_history(raw.dropna(how="all"))
        # Fill any gaps one at a time (yf.download occasionally drops a symbol).
        for t in tickers:
            if t not in out:
                try:
                    h = yf.Ticker(t).history(period=period, auto_adjust=True, actions=False)
                    if h is not None and not h.empty:
                        out[t] = _tidy_history(h)
                except Exception as exc:  # pragma: no cover - network
                    log.debug("history failed for %s: %s", t, exc)
        if any(t not in out for t in required):
            return None   # do not cache a download that lacks the essentials
        return out

    data = cache.get_or_fetch(key, ttl_hours, _download) or {}
    # Yahoo intermittently returns only the latest bar for some indices (^VIX3M, ^VIX6M). Keep a
    # per-ticker archive that accumulates every row ever seen so a partial response never erases
    # history; the archive is refreshed with new rows and served back merged.
    for t, df in list(data.items()):
        akey = f"archive:{t}:{period}"
        old = cache.get(akey, ttl_hours=10 ** 6)
        merged = df if old is None else pd.concat([old, df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        if old is None or len(merged) != len(old) or not merged.index[-1] == old.index[-1]:
            cache.put(akey, merged)
        data[t] = merged
    return data, cache.mtime(key)


def fetch_info(ticker: str, cache: Cache, ttl_hours: float) -> dict:
    def _fetch() -> dict:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:  # pragma: no cover - network
            log.debug("info failed for %s: %s", ticker, exc)
            info = {}
        return dict(info)

    return cache.get_or_fetch(f"info:{ticker}", ttl_hours, _fetch) or {}


def fetch_infos(tickers: List[str], cache: Cache, ttl_hours: float, workers: int = 4) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_info, t, cache, ttl_hours): t for t in tickers}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result() or {}
    return out


def fetch_holdings(fund: str, cache: Cache, ttl_hours: float):
    """Top holdings (symbol, name, weight) and yield-style valuation ratios for an ETF."""

    def _fetch():
        holdings = None
        stats: Dict[str, float] = {}
        try:
            fd = yf.Ticker(fund).funds_data
            th = fd.top_holdings
            if th is not None and not th.empty:
                holdings = pd.DataFrame({
                    "symbol": th.index.astype(str),
                    "name": th["Name"].astype(str).values,
                    "weight": pd.to_numeric(th["Holding Percent"], errors="coerce").values,
                }).dropna(subset=["weight"]).reset_index(drop=True)
            eh = fd.equity_holdings
            if eh is not None and not eh.empty and fund in eh.columns:
                col = eh[fund]
                mapping = {
                    "Price/Earnings": "earnings_to_price",
                    "Price/Book": "book_to_price",
                    "Price/Sales": "sales_to_price",
                    "Price/Cashflow": "cashflow_to_price",
                    "3 Year Earnings Growth": "three_year_earnings_growth",
                }
                for row, name in mapping.items():
                    if row in col.index:
                        val = pd.to_numeric(col.loc[row], errors="coerce")
                        if pd.notna(val):
                            stats[name] = float(val)
        except Exception as exc:  # pragma: no cover - network
            log.debug("funds_data failed for %s: %s", fund, exc)
        if holdings is None and not stats:
            return None
        return {"holdings": holdings, "stats": stats}

    res = cache.get_or_fetch(f"holdings:{fund}", ttl_hours, _fetch) or {}
    return res.get("holdings"), res.get("stats", {})


def _parse_news(ticker: str, raw: list) -> List[NewsItem]:
    items: List[NewsItem] = []
    for entry in raw or []:
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
        title = (content.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or content.get("description") or "").strip()
        published = None
        stamp = content.get("pubDate") or content.get("displayTime")
        if isinstance(stamp, str) and stamp:
            try:
                published = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                published = None
        if published is None and entry.get("providerPublishTime"):
            try:
                epoch = float(entry["providerPublishTime"])
                if 0 < epoch < 4e9:   # seconds, not milliseconds
                    published = datetime.fromtimestamp(epoch, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError):
                published = None
        if published is None:
            log.debug("dropping undated news item: %s", title[:60])
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        provider = content.get("provider")
        if isinstance(provider, dict):
            provider = provider.get("displayName") or ""
        provider = provider or entry.get("publisher") or ""
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            block = content.get(key)
            if isinstance(block, dict) and block.get("url"):
                url = block["url"]
                break
        url = url or entry.get("link") or ""
        items.append(NewsItem(ticker=ticker, title=title, summary=summary, published=published,
                              provider=str(provider), url=str(url)))
    return items


def fetch_news(tickers: List[str], cache: Cache, ttl_hours: float, workers: int = 4) -> List[NewsItem]:
    def _one(t: str) -> List[NewsItem]:
        def _fetch():
            try:
                return _parse_news(t, yf.Ticker(t).news)
            except Exception as exc:  # pragma: no cover - network
                log.debug("news failed for %s: %s", t, exc)
                return []
        return cache.get_or_fetch(f"news:{t}", ttl_hours, _fetch) or []

    out: List[NewsItem] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for items in pool.map(_one, tickers):
            out.extend(items)
    return out


# ---------------------------------------------------------------------------------------
# FRED (optional)
# ---------------------------------------------------------------------------------------
def _fred_api(series_id: str, timeout: float, api_key: str, years: int = 6) -> Optional[pd.Series]:
    """Official FRED API (needs a free key). Missing observations are reported as '.'."""
    start = (datetime.now(timezone.utc) - pd.Timedelta(days=365 * years)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start},
        timeout=timeout, headers={"User-Agent": "Mozilla/5.0 spxlcast"},
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if not obs:
        return None
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    return s if len(s) else None


def _fred_csv(series_id: str, timeout: float) -> Optional[pd.Series]:
    """Key-less chart endpoint; blocked on some networks, kept as the fallback."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 spxlcast"})
    resp.raise_for_status()
    if "text/csv" not in resp.headers.get("Content-Type", "") and not resp.text.lower().startswith(("date", "observation")):
        return None
    df = pd.read_csv(io.StringIO(resp.text))
    if df.shape[1] != 2:
        return None
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    return s if len(s) else None


def _fred_one(series_id: str, timeout: float, api_key: Optional[str] = None) -> Optional[pd.Series]:
    if api_key:
        return _fred_api(series_id, timeout, api_key)
    return _fred_csv(series_id, timeout)


def fetch_fred(series_ids: List[str], cache: Cache, ttl_hours: float, timeout: float,
               api_key: Optional[str] = None) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    pending = []
    for sid in series_ids:
        hit = cache.get(f"fred:{sid}", ttl_hours)
        if hit is not None:
            out[sid] = hit
        else:
            pending.append(sid)
    if not pending:
        return out
    with ThreadPoolExecutor(max_workers=min(8, len(pending))) as pool:
        futures = {pool.submit(_fred_one, sid, timeout, api_key): sid for sid in pending}
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                s = fut.result()
            except Exception as exc:
                log.debug("FRED %s failed: %s", sid, exc)
                s = None
            if s is not None:
                out[sid] = s
                cache.put(f"fred:{sid}", s)
    return out


# ---------------------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------------------
def _utc(ts: Optional[float]) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)


def load_market(cfg: Config) -> MarketSnapshot:
    cache = Cache(cfg.cache_dir, read_enabled=not cfg.refresh)
    cache.prune(max_age_hours=72)
    snap = MarketSnapshot(asof=datetime.now(timezone.utc), max_stale_sessions=cfg.max_stale_sessions)

    snap.holdings, snap.equity_stats = fetch_holdings(cfg.index_etf, cache, cfg.info_ttl_hours)
    top_symbols: List[str] = []
    if snap.holdings is not None and not snap.holdings.empty:
        top_symbols = [s for s in snap.holdings["symbol"].tolist()[: cfg.news_holdings_top_n]]

    tickers = list(dict.fromkeys(list(MARKET_TICKERS.keys()) + top_symbols))
    ny_date, is_open = session_state()
    price_ttl = cfg.price_ttl_hours_open if is_open else cfg.price_ttl_hours
    snap.prices, price_time = fetch_prices(tickers, cfg.history_period, cache, price_ttl,
                                           required=(cfg.etf, cfg.index_etf))
    snap.fetched_at["prices"] = _utc(price_time)
    missing = [t for t in tickers if t not in snap.prices]
    if missing:
        snap.notes.append("No price history for: " + ", ".join(missing))

    # Trading calendar from the index ETF; drop CBOE/CBOT index rows on NYSE holidays.
    idx_close = snap.close(cfg.index_etf)
    if idx_close is not None:
        snap.calendar = idx_close.index
        for t in list(snap.prices):
            if t.startswith("^"):
                df = snap.prices[t]
                snap.prices[t] = df[df.index.isin(snap.calendar)]
    # Is the latest bar a live, partial session?
    last = snap.last_date(cfg.etf)
    snap.intraday = bool(last is not None and str(last.date()) == ny_date and is_open)

    info_tickers = [cfg.etf, cfg.index_etf] + top_symbols
    snap.infos = fetch_infos(info_tickers, cache, cfg.info_ttl_hours)

    if cfg.use_fred:
        key = fred_api_key()
        snap.fred = fetch_fred(list(FRED_SERIES.keys()), cache, cfg.info_ttl_hours, cfg.fred_timeout, api_key=key)
        got = sorted(snap.fred.keys())
        if not got:
            snap.notes.append(("FRED unreachable" if key else "FRED unreachable and no FRED_API_KEY set")
                              + ": using Yahoo yields and default inflation; credit/labour/CPI signals skipped")
        elif len(got) < len(FRED_SERIES):
            snap.notes.append("FRED partial: missing " + ", ".join(s for s in FRED_SERIES if s not in got))

    if cfg.use_news:
        news_tickers = list(dict.fromkeys(list(cfg.news_tickers) + top_symbols))
        snap.news = fetch_news(news_tickers, cache, cfg.news_ttl_hours)
        snap.fetched_at["news"] = _utc(min((cache.mtime(f"news:{t}") or 0) for t in news_tickers) or None)
        if not snap.news:
            snap.notes.append("No news returned by Yahoo; sentiment set to neutral")

    return snap
