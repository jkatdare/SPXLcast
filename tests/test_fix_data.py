"""Regression tests for the data-group fixes of the break-it campaign (all offline)."""
import logging
import os
import time
import traceback
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
import requests

import spxlcast.data as data
from spxlcast.cache import Cache
from spxlcast.config import Config, nyse_session
from spxlcast.data import MarketSnapshot, session_state
from spxlcast.etf import Calibration, build_etf_params, calibrate, expense_ratio_from_info

NY = "America/New_York"


def _bars(closes, dates) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6},
                        index=pd.DatetimeIndex(dates))


def _serve(monkeypatch, frames, clock):
    """Yahoo returns ``frames`` and New York time is ``clock``."""
    monkeypatch.setattr(data, "ny_now", lambda: pd.Timestamp(clock, tz=NY))
    monkeypatch.setattr(data.yf, "download", lambda tickers, **kw: pd.concat({t: frames[t] for t in tickers}, axis=1))


def _fetch(cache, tickers):
    return data.fetch_prices(tickers, "5y", cache, ttl_hours=-1.0)[0]   # never reuse the download cache


def _archive(cache, t):
    return Cache(cache.directory).get(f"archive:{t}:5y", 1e6)


# ---- R2-01: the fresh download wins every overlapping date -------------------------------------
@pytest.mark.parametrize("n_old,n_new", [(60, 40), (257, 255), (500, 499), (1263, 1255), (300, 1255)])
def test_download_wins_every_overlapping_date(tmp_path, monkeypatch, n_old, n_new):
    cache = Cache(str(tmp_path))
    days = pd.bdate_range(end="2026-09-23", periods=max(n_old, n_new))
    px = 100 + np.arange(len(days), dtype=float)
    cache.put("archive:SPY:5y", _bars(px[-n_old:] * 1.0025, days[-n_old:]))   # archived before an ex-date
    new = _bars(px[-n_new:], days[-n_new:])
    _serve(monkeypatch, {"SPY": new}, "2026-09-23 17:40")
    served = _fetch(cache, ["SPY"])["SPY"]
    assert (served.loc[new.index, "Close"] == new["Close"]).all()
    # same length and last date as before, but new values: the archive is re-saved
    assert _archive(cache, "SPY").equals(served)


def test_archived_rows_older_than_the_download_follow_a_split(tmp_path, monkeypatch):
    cache = Cache(str(tmp_path))
    days = pd.bdate_range(end="2026-09-23", periods=300)
    px = 150 * np.cumprod(1 + 0.01 * np.random.default_rng(1).standard_normal(300))
    cache.put("archive:SPXL:5y", _bars(px[:-1], days[:-1]))                  # built before a 3-for-1 split
    _serve(monkeypatch, {"SPXL": _bars(px[-250:] / 3, days[-250:])}, "2026-09-23 17:40")
    served = _fetch(cache, ["SPXL"])["SPXL"]
    assert len(served) == 300 and np.allclose(served["Close"], px / 3, rtol=1e-12)
    assert served["Close"].pct_change().abs().max() < 0.1                     # no fake -67% day at the seam


def test_a_partial_or_mixed_overlap_does_not_rescale_older_rows(tmp_path, monkeypatch):
    days = pd.bdate_range(end="2026-09-23", periods=100)
    px = 20 + np.arange(100.0) / 10
    # a single corrected print for the last archived date (a latest-bar-only response)
    cache = Cache(str(tmp_path / "a"))
    cache.put("archive:^VIX3M:5y", _bars(px, days))
    _serve(monkeypatch, {"^VIX3M": _bars([px[-1] * 1.05], days[-1:])}, "2026-09-24 08:40")
    served = _fetch(cache, ["^VIX3M"])["^VIX3M"]
    assert np.array_equal(served["Close"].iloc[:-1], px[:-1]) and served["Close"].iloc[-1] == px[-1] * 1.05
    # an archive that already mixes two bases gives no single ratio to apply
    cache = Cache(str(tmp_path / "b"))
    mixed = px.copy()
    mixed[40::2] *= 1.0025
    cache.put("archive:SPY:5y", _bars(mixed, days))
    _serve(monkeypatch, {"SPY": _bars(px[40:], days[40:])}, "2026-09-23 17:40")
    served = _fetch(cache, ["SPY"])["SPY"]
    assert np.array_equal(served["Close"], px)


def test_partial_bar_is_served_but_never_archived(tmp_path, monkeypatch):
    cache = Cache(str(tmp_path))
    days = pd.bdate_range(end="2026-09-24", periods=30)
    final = _bars(100 + np.arange(30.0), days)
    partial = final.copy()
    partial.iloc[-1, :4] *= 0.97                                              # the 09:40 print
    _serve(monkeypatch, {"SPXL": partial}, "2026-09-24 09:40")
    served = _fetch(cache, ["SPXL"])["SPXL"]
    assert served["Close"].iloc[-1] == partial["Close"].iloc[-1]
    assert _archive(cache, "SPXL").index[-1] == days[-2]
    # the next morning Yahoo has the final close for the 24th, and that is what is served
    nxt = pd.concat([final, _bars([131.0], [pd.Timestamp("2026-09-25")])])
    _serve(monkeypatch, {"SPXL": nxt}, "2026-09-25 09:40")
    served = _fetch(cache, ["SPXL"])["SPXL"]
    assert served.loc["2026-09-24", "Close"] == final["Close"].iloc[-1]
    assert _archive(cache, "SPXL").index[-1] == days[-1]
    _serve(monkeypatch, {"SPXL": nxt}, "2026-09-25 16:40")                   # after the close it is kept
    _fetch(cache, ["SPXL"])
    assert _archive(cache, "SPXL").index[-1] == pd.Timestamp("2026-09-25")


# ---- R2-06: --refresh bypasses the download cache, not the archive -------------------------------
def test_refresh_keeps_the_archive(tmp_path, monkeypatch):
    days = pd.bdate_range(end="2026-09-23", periods=200)
    _serve(monkeypatch, {"^VIX3M": _bars(20 + np.sin(np.arange(200.0)), days)}, "2026-09-23 17:40")
    _fetch(Cache(str(tmp_path)), ["^VIX3M"])
    _serve(monkeypatch, {"^VIX3M": _bars([21.0], [pd.Timestamp("2026-09-24")])}, "2026-09-24 17:40")
    served = _fetch(Cache(str(tmp_path), read_enabled=False), ["^VIX3M"])["^VIX3M"]
    assert len(served) == 201
    assert len(_archive(Cache(str(tmp_path)), "^VIX3M")) == 201


# ---- R1-40: zero and non-finite closes --------------------------------------------------------
def test_bad_closes_are_dropped_for_funds_but_not_for_yields(tmp_path, monkeypatch):
    cache = Cache(str(tmp_path))
    days = pd.bdate_range(end="2026-09-23", periods=40)
    spxl = _bars(np.linspace(100, 120, 40), days)
    spxl.loc[days[10], "Close"] = 0.0
    spxl.loc[days[12], "Close"] = np.inf
    irx = _bars(np.full(40, 4.0), days)
    irx.loc[days[5], "Close"] = 0.0                                           # a bill yield can print zero
    cache.put("archive:SPXL:5y", _bars([0.0], [days[0] - pd.Timedelta(days=1)]))   # a zero Yahoo later dropped
    _serve(monkeypatch, {"SPXL": spxl, "^IRX": irx}, "2026-09-23 17:40")
    out = _fetch(cache, ["SPXL", "^IRX"])
    assert len(out["SPXL"]) == 38 and (out["SPXL"]["Close"] > 0).all() and np.isfinite(out["SPXL"]["Close"]).all()
    assert len(out["^IRX"]) == 40
    assert _archive(cache, "SPXL").equals(out["SPXL"])


def _pair(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.normal(0.0004, 0.011, n)
    etf = 3.0 * idx - 0.10 / 252 + rng.normal(0, 0.0008, n)
    days = pd.bdate_range(end="2026-09-23", periods=n)
    return pd.Series(50 * np.cumprod(1 + etf), index=days), pd.Series(400 * np.cumprod(1 + idx), index=days)


def _snap(etf_px, idx_px):
    snap = MarketSnapshot(asof=datetime(2026, 9, 23, 21, 40, tzinfo=timezone.utc))
    snap.prices = {"SPXL": etf_px.to_frame("Close"), "SPY": idx_px.to_frame("Close")}
    return snap


def test_zero_close_does_not_break_the_calibration():
    etf_px, idx_px = _pair()
    etf_px.iloc[-100] = 0.0
    idx_px.iloc[-200] = np.inf
    cal = calibrate(etf_px, idx_px, 504)
    assert np.isfinite(cal.beta) and abs(cal.beta - 3.0) < 0.03 and cal.r2 > 0.99


def test_implausible_calibration_keeps_the_stated_leverage():
    etf_px, idx_px = _pair()
    clean = build_etf_params(_snap(etf_px, idx_px), Config(), 0.04)
    assert clean.calibration is not None and clean.leverage == 3.0 and 0 < clean.tracking_sd_daily < 0.0012
    # half the index still on the pre-dividend basis, half of SPXL on the pre-split scale
    mixed_idx = idx_px.copy()
    mixed_idx.iloc[::2] *= 1.0033
    mixed_etf = etf_px.copy()
    mixed_etf.iloc[::2] *= 3.0
    for e, i in ((etf_px, mixed_idx), (mixed_etf, idx_px)):
        p = build_etf_params(_snap(e, i), Config(), 0.04)
        assert p.leverage == 3.0 and p.tracking_sd_daily == 0.0 and p.calibration is None
        assert any(n.startswith("calibration rejected") for n in p.notes)
        assert np.isfinite(p.financing_rate)


def test_non_finite_calibration_keeps_the_stated_leverage(monkeypatch):
    nan = float("nan")
    monkeypatch.setattr("spxlcast.etf.calibrate", lambda *a, **k: Calibration(nan, nan, nan, nan, 504, 0, nan))
    p = build_etf_params(_snap(*_pair()), Config(), 0.04)
    assert p.leverage == 3.0 and p.tracking_sd_daily == 0.0 and p.calibration is None
    assert np.isfinite(p.annual_cost)


# ---- R1-05: NYSE holidays and early closes -----------------------------------------------------
@pytest.mark.parametrize("stamp,is_open", [
    ("2026-09-24 09:29", False), ("2026-09-24 09:30", True), ("2026-09-24 15:59", True),
    ("2026-09-24 16:00", False), ("2026-09-26 11:00", False),
    ("2026-11-26 11:00", False),                                   # Thanksgiving
    ("2026-11-27 12:59", True), ("2026-11-27 13:00", False),       # 13:00 early close
    ("2026-12-24 12:30", True), ("2026-12-24 14:00", False),
    ("2026-12-25 11:00", False), ("2027-01-01 11:00", False),
    ("2026-07-03 11:00", False), ("2026-07-02 15:30", True),       # 4 July on a Saturday: Friday off, no early close
    ("2025-07-03 13:30", False), ("2026-04-03 11:00", False),      # early close; Good Friday
    ("2027-06-18 11:00", False),                                   # Juneteenth observed on the Friday
    ("2021-12-31 11:00", True),                                    # a Saturday New Year's Day is not moved
    ("2025-01-09 11:00", False),                                   # national day of mourning
])
def test_session_state_follows_the_nyse_calendar(stamp, is_open):
    assert session_state(pd.Timestamp(stamp, tz=NY)) == (stamp[:10], is_open)


def test_nyse_calendar_matches_published_schedules():
    def closed(year):
        return [str(d.date()) for d in pd.date_range(f"{year}-01-01", f"{year}-12-31")
                if d.weekday() < 5 and nyse_session(d.date()) is None]

    def early(year):
        return [str(d.date()) for d in pd.date_range(f"{year}-01-01", f"{year}-12-31")
                if (nyse_session(d.date()) or (None, None))[1] == datetime(2000, 1, 1, 13).time()]

    assert closed(2026) == ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
                            "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
    assert early(2026) == ["2026-11-27", "2026-12-24"]
    assert early(2025) == ["2025-07-03", "2025-11-28", "2025-12-24"]
    sessions = {y: sum(nyse_session(d.date()) is not None for d in pd.date_range(f"{y}-01-01", f"{y}-12-31"))
                for y in (2023, 2024, 2025)}
    assert sessions == {2023: 250, 2024: 252, 2025: 250}
    assert nyse_session(date(2026, 9, 24)) == (datetime(2000, 1, 1, 9, 30).time(), datetime(2000, 1, 1, 16).time())


# ---- R1-19: the FRED key never reaches a log or a traceback ------------------------------------
SECRET = "k" * 32
FRED_URL = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={SECRET}&file_type=json"


class _Resp:
    status_code = 400

    def raise_for_status(self):
        raise requests.HTTPError(f"400 Client Error: Bad Request for url: {FRED_URL}", response=self)


def _refused(*a, **kw):
    raise requests.ConnectionError(f"HTTPSConnectionPool(host='api.stlouisfed.org', port=443): "
                                   f"Max retries exceeded with url: {FRED_URL}")


@pytest.mark.parametrize("get,detail", [(_refused, "ConnectionError"), (lambda *a, **kw: _Resp(), "HTTP 400")])
def test_fred_errors_never_carry_the_key(tmp_path, monkeypatch, caplog, get, detail):
    monkeypatch.setattr(data.requests, "get", get)
    with pytest.raises(RuntimeError) as ei:
        data._fred_api("DGS10", 1.0, SECRET)
    text = "".join(traceback.format_exception(ei.type, ei.value, ei.tb))
    assert SECRET not in text and detail in text
    caplog.set_level(logging.DEBUG)
    assert data.fetch_fred(["DGS10"], Cache(str(tmp_path)), 1.0, 1.0, api_key=SECRET) == {}
    assert "DGS10" in caplog.text and SECRET not in caplog.text


def test_urllib3_request_lines_are_not_logged_under_verbose(caplog):
    caplog.set_level(logging.DEBUG)                                 # what -v does to the root logger
    logging.getLogger("urllib3.connectionpool").debug('https://api.stlouisfed.org:443 "GET /?api_key=%s"', SECRET)
    assert SECRET not in caplog.text


# ---- R2-05: hourly runs read fresh headlines ------------------------------------------------------
def test_hourly_runs_refetch_the_news(tmp_path, monkeypatch):
    calls = []

    class Ticker:
        def __init__(self, t):
            self.t = t

        @property
        def news(self):
            calls.append(self.t)
            return [{"content": {"title": "Stocks rise", "pubDate": "2026-09-24T14:00:00Z"}}]

    monkeypatch.setattr(data.yf, "Ticker", Ticker)
    cache, ttl = Cache(str(tmp_path)), Config().news_ttl_hours
    data.fetch_news(["SPY"], cache, ttl)
    stamp = time.time() - 55 * 60                                     # the previous hourly run's fetch
    os.utime(cache._path("news:SPY"), (stamp, stamp))
    data.fetch_news(["SPY"], cache, ttl)
    assert calls == ["SPY", "SPY"]


# ---- R1-42: non-numeric Yahoo info values ---------------------------------------------------------
def test_non_numeric_info_values_are_dropped(tmp_path, monkeypatch):
    class Ticker:
        def __init__(self, t):
            self.info = {"trailingPE": "Infinity", "forwardPE": "21.5", "yield": float("nan"),
                         "dividendYield": 0.012, "netExpenseRatio": "0.87", "expenseRatio": "n/a", "longName": "Fund"}

    monkeypatch.setattr(data.yf, "Ticker", Ticker)
    cache = Cache(str(tmp_path))
    want = {"forwardPE": 21.5, "dividendYield": 0.012, "netExpenseRatio": 0.87, "longName": "Fund"}
    assert data.fetch_info("SPY", cache, 1.0) == want
    assert data.fetch_info("SPY", cache, 1.0) == want                 # and again from the cache


def test_expense_ratio_skips_non_numeric_values():
    assert expense_ratio_from_info({"netExpenseRatio": "Infinity", "annualReportExpenseRatio": "0.0095"}, 0.0091) \
        == (0.0095, "Yahoo info.annualReportExpenseRatio")
    assert expense_ratio_from_info({"netExpenseRatio": "n/a", "expenseRatio": None}, 0.0091) == (0.0091, "default")
