"""Regression tests for the backtest-group fixes of the break-it campaign (all offline, synthetic data)."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_drift as bd  # noqa: E402
import backtest_rating as br  # noqa: E402
from spxlcast.config import Config  # noqa: E402
from spxlcast.evaluation import effective_n, spearman  # noqa: E402
from spxlcast.fundamentals import cpi_yoy_from, sahm_gap_from  # noqa: E402


# ---- synthetic backtest inputs ------------------------------------------------------------
def _monthly(start, end, values):
    idx = pd.date_range(start, end, freq="MS")
    return pd.Series(values(np.arange(len(idx))), index=idx, dtype=float)


def _shiller(end="2026-06"):
    """Quarter-end E and D, linearly interpolated in between, as in Shiller's file."""
    idx = pd.date_range("2000-01", end, freq="MS")
    qe = idx.month.isin([3, 6, 9, 12])
    e = pd.Series(np.where(qe, 100.0 + 10.0 * np.arange(len(idx)) % 7, np.nan), index=idx).interpolate()
    return pd.DataFrame({"E": e.bfill(), "D": e.bfill() / 3.0})


def _inputs(start="2019-01-02", end="2023-06-30", shiller_end="2023-02", seed=0):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    r = 0.0004 + 0.01 * rng.standard_normal(len(idx))
    tr = 1000.0 * np.cumprod(1.0 + r)
    closes = pd.DataFrame({"^SP500TR": tr, "^GSPC": tr / 2.0, "^VIX": 18.0 + rng.random(len(idx)),
                           "^VIX3M": 19.0 + rng.random(len(idx)), "^VIX6M": 20.0 + rng.random(len(idx)),
                           "SPXL": 50.0 * np.cumprod(1.0 + 3.0 * r - 0.0002)}, index=idx)
    daily = pd.bdate_range("2000-01-03", end)
    fred = {"DGS3MO": pd.Series(2.0, index=daily), "DGS10": pd.Series(3.0, index=daily),
            "T10YIE": pd.Series(2.2, index=daily),
            "BAMLH0A0HYM2": pd.Series(4.0, index=pd.bdate_range("2022-06-01", end)),
            "CPIAUCSL": _monthly("2000-01", "2023-05", lambda i: 170.0 * 1.0025 ** i),
            "UNRATE": _monthly("2000-01", "2023-05", lambda i: 4.0 + 0.0 * i)}
    return {"closes": closes, "fred": fred, "shiller": _shiller(shiller_end), "fetched": "test"}


def _run(tmp_path, start, paths=40):
    pd.to_pickle(_inputs(), tmp_path / "rating_inputs.pkl")
    br.run(argparse.Namespace(paths=paths, start=start, term_structure="flat", refresh=False, out=str(tmp_path)))
    return (tmp_path / "rating_backtest.txt").read_text(encoding="utf-8")


# ---- R1-38: earnings only once the quarter is reported ------------------------------------
def test_reported_quarter_is_a_quarter_end_at_least_two_months_old():
    for origin in pd.date_range("2008-01-31", "2009-12-31", freq="ME"):
        q = br.reported_quarter(origin)
        age = (origin.year - q.year) * 12 + origin.month - q.month
        assert q.month in (3, 6, 9, 12) and 2 <= age <= 4, (origin, q)
    assert br.reported_quarter(pd.Timestamp("2009-01-30")) == pd.Timestamp("2008-09-01")   # Q4 not yet out
    assert br.reported_quarter(pd.Timestamp("2009-02-27")) == pd.Timestamp("2008-12-01")


def test_inputs_at_reads_quarter_end_earnings_not_the_interpolated_month():
    inputs = _inputs()
    cfg = Config()
    d = br.build_daily(inputs, cfg)
    pos = br.month_end_positions(d.index, "2021-01")[0]                 # 2021-01-29: last quarter out is Sep 2020
    x = br.inputs_at(d, pos, inputs, cfg, "flat", {})
    sh = inputs["shiller"]
    assert x["fund"].earnings_yield == pytest.approx(sh.at[pd.Timestamp("2020-09-01"), "E"] / d["gspc"].iloc[pos])
    assert sh.at[pd.Timestamp("2020-10-01"), "E"] != sh.at[pd.Timestamp("2020-09-01"), "E"]   # the leak it avoids


# ---- R1-34: CPI YoY and the Sahm gap by calendar month, as the live model -----------------
def test_inputs_at_computes_cpi_and_sahm_by_date_across_a_missing_month():
    inputs = _inputs(end="2026-09-25", shiller_end="2026-06")
    cpi = _monthly("2000-01", "2026-08", lambda i: 170.0 * 1.0028 ** i)
    un = _monthly("2000-01", "2026-08", lambda i: 4.0 + 0.02 * np.maximum(i - 300, 0))
    inputs["fred"]["CPIAUCSL"] = cpi.drop(pd.Timestamp("2025-10-01"))       # BLS published no October 2025
    inputs["fred"]["UNRATE"] = un.drop(pd.Timestamp("2025-10-01"))
    cfg = Config()
    d = br.build_daily(inputs, cfg)
    pos = [p for p in br.month_end_positions(d.index, "2026-08") if d.index[p].month == 8][0]
    macro = br.inputs_at(d, pos, inputs, cfg, "flat", {})["macro"]
    released = br.released(inputs["fred"]["CPIAUCSL"], d.index[pos])
    assert macro.cpi_yoy == pytest.approx(cpi_yoy_from(released))
    assert macro.cpi_yoy == pytest.approx(1.0028 ** 12 - 1.0)                # 12 months, not 12 rows
    assert macro.unemployment_sahm_gap == pytest.approx(
        sahm_gap_from(br.released(inputs["fred"]["UNRATE"], d.index[pos])))


# ---- R1-31: closed-form intervals that cover ----------------------------------------------
def test_diff_and_rank_correlation_intervals_cover_on_overlapping_windows():
    rng = np.random.default_rng(3)
    n, w, h = 300, 6, 126
    pos = pd.Series(np.arange(n) * 21)
    hits = {"buy": [], "sell": [], "rank": []}
    for _ in range(150):
        y = pd.Series(np.convolve(rng.standard_t(4, n + w) * 0.06, np.ones(w), "valid")[1:n + 1])
        buy = np.empty(n, bool)
        buy[0] = True
        flips = rng.random(n) < 0.04
        for i in range(1, n):
            buy[i] = (not buy[i - 1]) if flips[i] else buy[i - 1]
        sell = np.zeros(n, bool)
        for c in rng.choice(n - 10, 4, replace=False):
            sell[c:c + rng.integers(1, 7)] = True
        score = pd.Series(rng.standard_normal(n)).ewm(alpha=0.1).mean().values
        for key, sel in (("buy", buy & ~sell), ("sell", sell)):
            lo, hi = br._diff_ci(y, pd.Series(sel), pos, h)
            if np.isfinite(lo):
                hits[key].append(lo <= 0 <= hi)
        lo, hi = br._corr_ci(spearman(score, y.values), effective_n(pos, h))
        hits["rank"].append(lo <= 0 <= hi)
    for key, v in hits.items():
        assert np.mean(v) >= 0.85, (key, np.mean(v))


def test_diff_ci_is_na_for_a_lone_month_or_run():
    y = pd.Series(np.linspace(-0.2, 0.3, 60))
    pos = pd.Series(np.arange(60) * 21)
    lone = pd.Series(np.arange(60) == 30)
    run = pd.Series((np.arange(60) >= 30) & (np.arange(60) < 33))       # three overlapping 6-month windows
    for sel in (lone, run):
        assert all(math.isnan(v) for v in br._diff_ci(y, sel, pos, 126))
    assert all(math.isnan(v) for v in br._corr_ci(0.1, 4.0))


# ---- R1-36: ties at the score cap, late starts, no origins ---------------------------------
def test_rating_section_survives_a_quarter_of_scores_tied_at_the_cap():
    n = 120
    rng = np.random.default_rng(1)
    score = np.where(np.arange(n) % 4 == 0, 1.0, rng.uniform(-0.5, 0.99, n))
    o = pd.DataFrame({"date": pd.date_range("2011-01-31", periods=n, freq="ME"), "pos": np.arange(n) * 21,
                      "score": score, "score_const7": score, "label": np.where(score >= 0.3, "BUY", "HOLD")})
    for hh in (21, 63, 126):
        o[f"excess_{hh}"] = rng.standard_normal(n) * 0.1
        o[f"ret_{hh}"] = o[f"excess_{hh}"] + 0.01
        o[f"sp_{hh}"] = o[f"excess_{hh}"] / 3
    text = "\n".join(br.rating_section(o))
    assert "tied at a score of +1.00" in text
    assert "Q5 (highest)" in text


def test_run_reports_gaps_and_handles_late_or_empty_starts(tmp_path, capsys):
    text = _run(tmp_path, "2019-01")
    err = capsys.readouterr().err
    assert "credit-stress penalty cannot fire" in text and "credit-stress penalty cannot fire" in err
    assert "Skipped" in text and "Shiller earnings end 2023-02" in text     # origins past the Shiller file
    assert "excess minus the other months" in text and "Calibration" in text

    late = tmp_path / "late"
    late.mkdir()
    text = _run(late, "2023-03")
    assert "No month-end has a full 6-month outcome yet" in text

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no complete month-end"):
        _run(empty, "2024-01")


# ---- R1-37: Shiller link discovery and download validation ---------------------------------
class _Resp:
    def __init__(self, text="", status=200, content=None):
        self.text, self.status_code = text, status
        self.content = text.encode() if content is None else content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.mark.parametrize("page, expected", [
    ('<a href="/s/ie_data.xls?ver=2">data</a>', "https://shillerdata.com/s/ie_data.xls?ver=2"),
    ('<a href=https://x.wsimg.com/d/ie_data.xls?ver=1>data</a>', "https://x.wsimg.com/d/ie_data.xls?ver=1"),
    ('<p>was http://www.econ.yale.edu/~shiller/data/ie_data.xls, now</p><a href="//x.wsimg.com/ie_data.xls">d</a>',
     "https://x.wsimg.com/ie_data.xls"),
    ('{"u":"https:\\/\\/x.wsimg.com\\/IE_DATA.XLS?a=1&amp;b=2"}', "https://x.wsimg.com/IE_DATA.XLS?a=1&b=2"),
    ("(download: https://x.wsimg.com/ie_data.xlsx?ver=4)", "https://x.wsimg.com/ie_data.xlsx?ver=4"),
])
def test_shiller_url_finds_the_link_in_page_variants(monkeypatch, page, expected):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(page))
    assert bd.shiller_url() == expected


def test_shiller_url_warns_when_it_falls_back_to_the_stale_yale_file(monkeypatch, capsys):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp("Just a moment...", 403))
    assert bd.shiller_url() == bd.URL
    assert "stopped updating in 2023" in capsys.readouterr().err


def test_load_shiller_never_caches_a_non_excel_download(monkeypatch, tmp_path):
    def fake_get(url, *a, **k):
        return _Resp('<a href="https://x.wsimg.com/ie_data.xls">d</a>') if url == bd.PAGE else _Resp("<html>cookies</html>")
    monkeypatch.setattr(requests, "get", fake_get)
    path = tmp_path / "ie_data.xls"
    with pytest.raises(RuntimeError, match="did not return an Excel file"):
        bd.load_shiller(path)
    assert not path.exists()

    path.write_bytes(b"<html>left by an older version</html>")
    with pytest.raises(RuntimeError, match="delete it"):
        bd.load_shiller(path)
