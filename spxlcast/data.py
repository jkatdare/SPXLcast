"""Data access: Yahoo Finance (prices, fund info, holdings, news) and optional FRED macro series.

Everything is cached on disk (see cache.py). Every fetch is defensive: a missing ticker or a
timed-out FRED call degrades to ``None`` and a note in ``MarketSnapshot.notes`` instead of an error.
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .cache import Cache
from .config import FRED_SERIES, MARKET_TICKERS, Config

log = logging.getLogger(__name__)


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
    asof: datetime
    prices: Dict[str, pd.DataFrame] = field(default_factory=dict)  # ticker -> OHLCV, auto-adjusted
    infos: Dict[str, dict] = field(default_factory=dict)
    holdings: Optional[pd.DataFrame] = None  # columns: symbol, name, weight
    equity_stats: Dict[str, float] = field(default_factory=dict)  # yield-style ratios from SPY fund data
    fred: Dict[str, pd.Series] = field(default_factory=dict)
    news: List[NewsItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

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

    def change(self, ticker: str, days: int) -> Optional[float]:
        """Fractional change in the close over the last ``days`` trading days."""
        s = self.close(ticker)
        if s is None or len(s) <= days:
            return None
        return float(s.iloc[-1] / s.iloc[-1 - days] - 1.0)

    def diff(self, ticker: str, days: int) -> Optional[float]:
        """Absolute change over ``days`` trading days (for yields quoted in %)."""
        s = self.close(ticker)
        if s is None or len(s) <= days:
            return None
        return float(s.iloc[-1] - s.iloc[-1 - days])

    def fred_last(self, series_id: str) -> Optional[float]:
        s = self.fred.get(series_id)
        if s is None or s.empty:
            return None
        return float(s.dropna().iloc[-1])

    def info(self, ticker: str) -> dict:
        return self.infos.get(ticker) or {}


# ---------------------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------------------
def _tidy_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"]) if "Close" in df else df


def fetch_prices(tickers: List[str], period: str, cache: Cache, ttl_hours: float) -> Dict[str, pd.DataFrame]:
    key = f"prices:{period}:{','.join(sorted(tickers))}"

    def _download() -> Dict[str, pd.DataFrame]:
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
        return out

    return cache.get_or_fetch(key, ttl_hours, _download) or {}


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
        return {"holdings": holdings, "stats": stats}

    res = cache.get_or_fetch(f"holdings:{fund}", ttl_hours, _fetch) or {}
    return res.get("holdings"), res.get("stats", {})


def _parse_news(ticker: str, raw: list) -> List[NewsItem]:
    items: List[NewsItem] = []
    now = datetime.now(timezone.utc)
    for entry in raw or []:
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
        title = (content.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or content.get("description") or "").strip()
        published = now
        stamp = content.get("pubDate") or content.get("displayTime")
        if isinstance(stamp, str) and stamp:
            try:
                published = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        elif entry.get("providerPublishTime"):
            published = datetime.fromtimestamp(float(entry["providerPublishTime"]), tz=timezone.utc)
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
def _fred_one(series_id: str, timeout: float) -> Optional[pd.Series]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 spxlcast"})
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    return s if len(s) else None


def fetch_fred(series_ids: List[str], cache: Cache, ttl_hours: float, timeout: float) -> Dict[str, pd.Series]:
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
        futures = {pool.submit(_fred_one, sid, timeout): sid for sid in pending}
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
def load_market(cfg: Config) -> MarketSnapshot:
    cache = Cache(cfg.cache_dir, enabled=not cfg.refresh)
    snap = MarketSnapshot(asof=datetime.now(timezone.utc))

    snap.holdings, snap.equity_stats = fetch_holdings(cfg.index_etf, cache, cfg.info_ttl_hours)
    top_symbols: List[str] = []
    if snap.holdings is not None and not snap.holdings.empty:
        top_symbols = [s for s in snap.holdings["symbol"].tolist()[: cfg.news_holdings_top_n]]

    tickers = list(dict.fromkeys(list(MARKET_TICKERS.keys()) + top_symbols))
    snap.prices = fetch_prices(tickers, cfg.history_period, cache, cfg.price_ttl_hours)
    missing = [t for t in tickers if t not in snap.prices]
    if missing:
        snap.notes.append("No price history for: " + ", ".join(missing))

    info_tickers = [cfg.etf, cfg.index_etf] + top_symbols
    snap.infos = fetch_infos(info_tickers, cache, cfg.info_ttl_hours)

    if cfg.use_fred:
        snap.fred = fetch_fred(list(FRED_SERIES.keys()), cache, cfg.info_ttl_hours, cfg.fred_timeout)
        got = sorted(snap.fred.keys())
        if not got:
            snap.notes.append("FRED unreachable: using Yahoo yields and default inflation; "
                              "credit/labour/CPI signals skipped")
        elif len(got) < len(FRED_SERIES):
            snap.notes.append("FRED partial: missing " + ", ".join(s for s in FRED_SERIES if s not in got))

    if cfg.use_news:
        news_tickers = list(dict.fromkeys(list(cfg.news_tickers) + top_symbols))
        snap.news = fetch_news(news_tickers, cache, cfg.news_ttl_hours)
        if not snap.news:
            snap.notes.append("No news returned by Yahoo; sentiment set to neutral")

    return snap
