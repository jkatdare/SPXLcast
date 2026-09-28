"""The assessment (leverage cost, drawdown risk, 3-month range) replaces the retired buy / hold / sell
verdict everywhere a user sees output: report, quiet line, JSON, live block, chart, log and scorer."""
import copy
import csv
import dataclasses
import io
import json
import math
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from scipy.stats import norm

import spxlcast.assess as A
import spxlcast.cli as cli
import spxlcast.pipeline as pipeline
from spxlcast import tracklog
from spxlcast.config import Config
from spxlcast.data import MarketSnapshot
from spxlcast.live import refresh_once, reprice
from spxlcast.report import quiet_summary, render_all, render_score
from spxlcast.serve import render_live

VERDICT = re.compile(r"\b(BUY|HOLD|SELL)\b|conviction|score [+-]")
FRI_1559 = datetime(2026, 9, 25, 19, 59, 30, tzinfo=timezone.utc)


def _market(n=300):
    cal = pd.bdate_range(end="2026-09-25", periods=n)
    rng = np.random.default_rng(1)
    r = 0.0003 + 0.011 * rng.standard_normal(n)
    frame = lambda v: pd.DataFrame({"Close": np.asarray(v, dtype=float)}, index=cal)  # noqa: E731
    snap = MarketSnapshot(asof=datetime(2026, 9, 25, 21, 40, tzinfo=timezone.utc))
    snap.prices = {"SPY": frame(450 * np.cumprod(1 + r)), "SPXL": frame(100 * np.cumprod(1 + 3 * r - 0.0001)),
                   "^IRX": frame(np.full(n, 4.0)), "^TNX": frame(np.full(n, 4.2)), "^VIX": frame(np.full(n, 17.0)),
                   "^VIX3M": frame(np.full(n, 19.0)), "^VIX6M": frame(np.full(n, 22.0))}
    snap.calendar = cal
    snap.infos = {"SPXL": {"netExpenseRatio": 0.87}, "SPY": {"trailingPE": 27.0, "yield": 0.012}}
    return snap


def _cfg(**kw):
    return Config(**{"n_paths": 2000, "horizons": (5, 21, 63, 126), "use_news": False, "use_fred": False, **kw})


def _run(cfg):
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "load_market", lambda c: _market())
    try:
        return pipeline.run_forecast(cfg)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def fc():
    return _run(_cfg())


def _report(fc, **kw) -> str:
    console = Console(record=True, width=400)
    render_all(fc, console, **kw)
    return console.export_text()


# ---- the forecast, its JSON and the report ---------------------------------------------------
def test_the_forecast_carries_the_assessment(fc):
    a = fc.assessment
    assert a.leverage.hurdle == pytest.approx(fc.etf.breakeven_index_return(fc.vol.one_year_vol))
    assert a.leverage.fees == fc.etf.expense_ratio and a.leverage.financing == fc.etf.financing_rate
    assert a.leverage.expected_index_return == fc.expected.final
    assert a.drawdown.p_dip20_3m == fc.sim.summary(63)["p_drawdown_20"]
    q = fc.sim.quantiles(63, (5, 50, 95))
    assert a.range_3m == (q[5.0], q[50.0], q[95.0])
    assert a.leverage.level in A.LEVERAGE_LEVELS and a.drawdown.level in A.DRAWDOWN_LEVELS
    doc = json.loads(json.dumps(pipeline.forecast_to_dict(fc), allow_nan=False))
    assert doc["assessment"]["leverage"]["level"] == a.leverage.level
    assert doc["assessment"]["range_3m"]["p95"] == pytest.approx(q[95.0])
    assert doc["rating"]["label"] == fc.rating.label          # kept in the JSON for continuity


def test_report_shows_the_assessment_and_no_verdict(fc):
    text = _report(fc)
    assert "Assessment" in text and "Rating" not in text and not VERDICT.search(text)
    a = fc.assessment
    assert re.search(rf"Leverage cost +{a.leverage.level}\b", text) and re.search(rf"Drawdown risk +{a.drawdown.level}\b", text)
    flat = " ".join(text.replace("│", " ").split())                   # the panel wraps its sentences
    assert f"has to return {a.leverage.hurdle:.1%} a year on average, dividends included, for SPXL just to break " \
           f"even over the long run" in flat
    assert f"Compared with 3 times the S&P 500's return, SPXL gives up about " \
           f"{a.leverage.financing + a.leverage.fees + a.leverage.drag:.1%} a year" in flat
    assert f"{a.drawdown.p_dip20_3m:.0%} chance SPXL closes at or below {fc.spot * 0.8:,.2f}, a fall of 20% or more" in flat
    assert f"ends between {a.range_3m[0]:,.2f} and {a.range_3m[2]:,.2f} in 90% of the model's simulations" in flat
    hist = A.load_reference()["p_dip20_3m"]["by_level"][a.drawdown.level]
    assert f"In the backtest's {hist['n']} months at this level from 1990 to 2026" in flat
    assert f"such a fall followed {hist['real']:.0%} of the time" in flat and "3x fund rebuilt" in flat
    assert re.search(r"That is higher than in \d+% of the months from 1990 to 2026 \(low = the cheapest third", flat)
    assert "This is not a buy or sell signal" in flat and "not a forecast for the coming months" in flat
    assert "Volatility drag to 6M" in flat and "Volatility decay" not in flat
    # the other sections are unchanged
    for title in ("price distribution", "Buy-limit ladder", "S&P 500 expected total return", "Volatility"):
        assert title in text
    line = quiet_summary(fc)
    assert line.startswith(f"leverage cost {a.leverage.level}") and f"drawdown risk {a.drawdown.level}" in line
    assert "break even over the long run" in line and "chance of a fall of 20% or more within 3 months" in line
    assert f"3-month range {a.range_3m[0]:,.2f} to {a.range_3m[2]:,.2f} (90% of outcomes), typical" in line
    assert not VERDICT.search(line) and "rating" not in line


def test_the_percentile_shown_stays_inside_the_level_shown(fc):
    """Rounding a percentile at a cutoff never prints a share that belongs to the next third."""
    a = fc.assessment
    for pctile, level, shown in ((66.66, "normal", "66%"), (33.34, "normal", "34%"), (33.2, "low", "33%"),
                                 (66.7, "elevated", "67%")):
        other = copy.copy(fc)
        other.assessment = dataclasses.replace(a, drawdown=dataclasses.replace(a.drawdown, percentile=pctile, level=level))
        flat = " ".join(_report(other, sections={"assessment"}).replace("│", " ").split())
        assert f"higher than in {shown} of the months from 1990 to 2026 (low = the calmest third" in flat
    other.assessment = dataclasses.replace(a, period=None)       # a reference without its period
    flat = " ".join(_report(other, sections={"assessment"}).replace("│", " ").split())
    assert "of the months in the backtest" in flat and "1990-2026 backtest" in flat


def test_no_reference_and_horizons_without_3_months_still_render(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "REFERENCE_PATH", tmp_path / "missing.json")
    fc = _run(_cfg(n_paths=300, horizons=(5, 21), rating_horizon=21))
    assert 63 in fc.sim.horizons                                  # the assessment's horizon is always simulated
    a = fc.assessment
    assert a.leverage.level == a.drawdown.level == A.NA and a.range_3m is not None and len(a.notes) == 1
    assert math.isfinite(a.drawdown.p_dip20_3m)
    text = _report(fc)
    assert all(re.search(rf"{name} +n/a", text) for name in ("Leverage cost", "Drawdown risk"))
    assert "higher than in" not in text and "missing or unreadable" in text and "3-month range  In 3 months" in text
    assert quiet_summary(fc).startswith("leverage cost n/a") and "3-month range" in quiet_summary(fc)
    assert json.dumps(pipeline.forecast_to_dict(fc), allow_nan=False)


def test_cli_quiet_line_json_and_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "load_market", lambda c: _market())
    out, log = tmp_path / "forecast.json", tmp_path / "log.csv"
    assert cli.main(["forecast", "--quiet", "--paths", "300", "--horizons", "5", "63", "--no-news", "--no-fred",
                     "--json", str(out), "--log-file", str(log), "--price", "90"]) == 0
    printed = capsys.readouterr().out
    doc = json.loads(out.read_text(encoding="utf-8"))
    a = doc["assessment"]
    assert f"leverage cost {a['leverage']['level']}" in printed and f"drawdown risk {a['drawdown']['level']}" in printed
    assert "price 90.00:" in printed and not VERDICT.search(printed) and "rating" not in printed
    price_line = next(ln for ln in printed.splitlines() if "price 90.00:" in ln)
    assert "percentile of the price in 6M" in price_line
    assert ("chance it dips to it" in price_line) != ("chance it rises to it" in price_line)   # one direction only
    row = next(csv.DictReader(open(log, encoding="utf-8")))
    assert float(row["hurdle"]) == pytest.approx(a["leverage"]["hurdle"], abs=1e-5)
    assert row["leverage_cost"] == a["leverage"]["level"] and row["drawdown_risk"] == a["drawdown"]["level"]
    assert row["rating"] == doc["rating"]["label"] and row["h63_p_dd20"]    # the old columns stay


def test_fan_chart_title_has_no_rating(fc, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from spxlcast.plots import save_fan_chart
    titles, orig = [], Axes.set_title
    monkeypatch.setattr(Axes, "set_title", lambda self, label, *a, **k: (titles.append(label), orig(self, label, *a, **k))[1])
    save_fan_chart(fc, io.BytesIO(), fmt="png")
    assert len(titles) == 2 and not any("rating" in t or VERDICT.search(t) for t in titles)


# ---- live.json and the page ------------------------------------------------------------------
def test_live_restates_the_assessment_at_the_quote(fc, tmp_path):
    base = json.loads(json.dumps(pipeline.forecast_to_dict(fc, [90.0])))
    live = reprice(base, fc.spot * 1.1, now=FRI_1559)
    a, b = live["assessment"], base["assessment"]
    assert a["leverage"] == b["leverage"] and a["drawdown"] == b["drawdown"]     # about returns: unchanged
    for k in ("p5", "median", "p95"):
        assert a["range_3m"][k] == pytest.approx(b["range_3m"][k] * 1.1)       # prices: scale with the quote
    assert live["horizons"]["63"]["quantile_prices"]["95.0"] == pytest.approx(a["range_3m"]["p95"])
    path = tmp_path / "live.json"
    path.write_text(json.dumps(live, allow_nan=False), encoding="utf-8")
    block = render_live(str(path), now=FRI_1559)
    assert f'leverage cost <b class="lvl">{a["leverage"]["level"]}</b>' in block
    assert f'drawdown risk <b class="lvl">{a["drawdown"]["level"]}</b>' in block
    assert f'{a["drawdown"]["p_dip20_3m"]:.0%} chance of a fall of 20% or more within 3 months' in block
    assert f'needs {a["leverage"]["hurdle"]:.1%}/yr for SPXL to break even over the long run' in block
    assert "the months from 1990 to 2026" in block and "<th>Low end (5% below)</th>" in block
    assert "rating" not in block and not VERDICT.search(block) and "not a buy or sell signal" in block


def test_older_files_without_an_assessment_fall_back(fc, tmp_path):
    base = json.loads(json.dumps(pipeline.forecast_to_dict(fc, [90.0])))
    del base["assessment"]                                       # a forecast.json from before this change
    live = reprice(base, fc.spot, now=FRI_1559)
    assert live["assessment"] is None
    path = tmp_path / "live.json"
    for doc in (live, {k: v for k, v in live.items() if k not in ("assessment", "rating")},
                {**live, "assessment": "text"}, {**live, "assessment": {"leverage": [], "drawdown": {}}},
                {**live, "assessment": {"leverage": {"level": "high", "hurdle": "x"}, "drawdown": {"level": "low"}}}):
        path.write_text(json.dumps(doc), encoding="utf-8")
        block = render_live(str(path), now=FRI_1559)
        assert 'class="live"' in block and "since the full run" in block and "rating" not in block
    assert 'leverage cost <b class="lvl">high</b> &middot; drawdown risk <b class="lvl">low</b>' in block
    # the minute loop still records the rating in spot_log.csv for continuity
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "forecast.json").write_text(json.dumps(base), encoding="utf-8")
    refresh_once(str(tmp_path), fetch=lambda t: fc.spot, now=FRI_1559)
    row = next(csv.DictReader(open(tmp_path / "logs" / "spot_log.csv", encoding="utf-8")))
    assert row["rating"] == fc.rating.label


# ---- the track record: drawdown risk by level ------------------------------------------------
class _Sim:
    """Lognormal terminal prices with a zero median log return; the dip chance is fixed."""

    def __init__(self, spot, horizons=(21, 63), p_dd20=0.05):
        self.spot, self.horizons, self.p_dd20 = spot, list(horizons), p_dd20
        self.rf_growth = {h: 1.0 + 0.04 * h / 252 for h in self.horizons}

    def quantiles(self, h, levels):
        return {float(q): self.spot * math.exp(0.02 * math.sqrt(h) * norm.ppf(q / 100.0)) for q in levels}

    def summary(self, h):
        return {"p_positive": 0.5, "p_beat_rf": 0.48, "p_drawdown_20": self.p_dd20, "p_up_20": 0.05}


def _row_fc(spot, day, level=None):
    a = None if level is None else SimpleNamespace(leverage=SimpleNamespace(hurdle=0.07, level="normal"),
                                                   drawdown=SimpleNamespace(level=level))
    return SimpleNamespace(
        snap=SimpleNamespace(asof=datetime(day.year, day.month, day.day, 21, 40, tzinfo=timezone.utc)),
        spot_date=str(day.date()), spot_status="close", spot=float(spot),
        rating=SimpleNamespace(label="HOLD", score=0.0, score_se=0.01, conviction="Low"),
        expected=SimpleNamespace(final=0.06), macro=SimpleNamespace(rf_3m=0.04, vix=17.0),
        etf=SimpleNamespace(annual_cost=0.10), sentiment=None, sim=_Sim(spot),
        vol=SimpleNamespace(total_vol=lambda h: 0.16), assessment=a,
        data_quality=lambda: {"flags": [], "fred_series": 9, "news_fetched": None})


def _dips(closes, positions, h=63):
    px = closes.to_numpy()
    return [float(px[p + 1: p + h + 1].min() <= 0.8 * px[p]) for p in positions]


def test_scorer_reports_drawdown_risk_by_level(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=600)
    px = np.full(600, 100.0)
    px[100:111] = 70.0                                        # two 30% dips
    px[400:406] = 75.0
    closes = pd.Series(px, index=dates)
    path = tmp_path / "log.csv"
    shown = list(range(0, 250, 10))                           # rows that logged the level the page showed
    older = list(range(250, 510, 10))                         # rows logged before the column: placed by h63_p_dd20
    for p in shown + older:
        tracklog.append_log(_row_fc(px[p], dates[p], "elevated" if p in shown else None), str(path))
    rep = tracklog.score_log(str(path), closes=closes, cfg=Config())
    ref = A.load_reference()["p_dip20_3m"]
    assert set(rep.by_drawdown_risk) == {"elevated", "low"}   # 0.05 is below the reference's lower cutoff
    for label, pos in (("elevated", shown), ("low", older)):
        r = rep.by_drawdown_risk[label]
        assert r["n"] == len(pos) and r["pred"] == pytest.approx(0.05)
        assert r["real"] == pytest.approx(np.mean(_dips(closes, pos)))
        assert r["n_eff"] == pytest.approx((len(pos) - 1) * 10 / 63 + 1)
        assert all(np.isfinite(r["real_ci"])) and r["real_ci"][0] <= r["real"] <= r["real_ci"][1]
        assert r["backtest_pred"] == pytest.approx(ref["by_level"][label]["pred"])
    assert 0 < rep.by_drawdown_risk["elevated"]["real"] < 1 and 0 < rep.by_drawdown_risk["low"]["real"] < 1

    console = Console(record=True, width=160)
    render_score(rep, console, str(path))
    text = console.export_text()
    assert "Drawdown risk check" in text and "rating" not in text
    lines = [ln for ln in text.splitlines() if ln.strip().startswith(("elevated", "low"))]
    assert [ln.split()[0] for ln in lines] == ["low", "elevated"]          # in level order
    assert f"{ref['by_level']['low']['pred']:.0%} / {ref['by_level']['low']['real']:.0%}" in lines[0]

    # an old log without the new columns: every row placed by its logged dip chance
    df = pd.read_csv(path).drop(columns=["hurdle", "leverage_cost", "drawdown_risk"])
    df.to_csv(path, index=False)
    rep = tracklog.score_log(str(path), closes=closes, cfg=Config())
    assert set(rep.by_drawdown_risk) == {"low"} and rep.by_drawdown_risk["low"]["n"] == len(shown + older)
    # and without a reference: only rows that logged their level can be placed, with no backtest column
    monkeypatch.setattr(tracklog, "load_reference", lambda: None)
    tracklog.append_log(_row_fc(px[520], dates[520], "normal"), str(path))
    rep = tracklog.score_log(str(path), closes=closes, cfg=Config())
    assert set(rep.by_drawdown_risk) == {"normal"} and math.isnan(rep.by_drawdown_risk["normal"]["backtest_pred"])
    console = Console(record=True, width=160)
    render_score(rep, console, str(path))
    assert "n/a / n/a" in console.export_text()
