"""Regression tests for the hand-offs between the fix groups (all offline)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console

import spxlcast.cli as cli
import spxlcast.data as data
import spxlcast.pipeline as pipeline
from spxlcast.config import Config
from spxlcast.data import MarketSnapshot
from spxlcast.report import render_drivers


def _market(n=300):
    cal = pd.bdate_range(end="2026-09-25", periods=n)
    rng = np.random.default_rng(1)
    r = 0.0003 + 0.011 * rng.standard_normal(n)
    frame = lambda v, idx=cal: pd.DataFrame({"Close": np.asarray(v, dtype=float)}, index=idx)  # noqa: E731
    snap = MarketSnapshot(asof=datetime(2026, 9, 25, 21, 40, tzinfo=timezone.utc))
    snap.prices = {"SPY": frame(450 * np.cumprod(1 + r)), "SPXL": frame(100 * np.cumprod(1 + 3 * r - 0.0001)),
                   "^IRX": frame(np.full(n, 4.0)), "^TNX": frame(np.full(n, 4.2)), "^VIX": frame(np.full(n, 17.0)),
                   "^VIX3M": frame(np.full(n, 19.0)), "^VIX6M": frame(np.full(n, 22.0))}
    snap.calendar = cal
    snap.infos = {"SPXL": {"netExpenseRatio": 0.87}, "SPY": {"trailingPE": 27.0, "yield": 0.012}}
    return snap


def test_report_break_even_is_the_one_year_figure_of_the_rating(monkeypatch):
    # R1-44: the report row and the rating's context line both use the 1-year vol, whatever the horizons
    monkeypatch.setattr(pipeline, "load_market", lambda cfg: _market())
    fc = pipeline.run_forecast(Config(n_paths=400, horizons=(5,), rating_horizon=5, use_news=False, use_fred=False))
    want = fc.etf.breakeven_index_return(fc.vol.one_year_vol)
    assert abs(want - fc.etf.breakeven_index_return(fc.vol.total_vol(5))) > 0.002     # the old row differed
    console = Console(record=True, width=200)
    render_drivers(fc, console)
    row = next(line for line in console.export_text().splitlines() if "Break-even index return" in line)
    assert f"{want:.1%}" in row
    assert f"{want:+.1%}/yr" in " ".join(fc.rating.reasons)


def test_score_fails_the_step_when_there_are_no_prices(monkeypatch, tmp_path):
    # R1-47: infra/daily.sh keeps the last good score.txt (and alerts) only when `score` exits non-zero
    log = str(tmp_path / "log.csv")
    monkeypatch.setattr(pipeline, "load_market", lambda cfg: _market())
    assert cli.main(["log", "--paths", "200", "--horizons", "5", "--rating-horizon", "5", "--no-news", "--no-fred",
                     "--log-file", log]) == 0
    monkeypatch.setattr(data, "fetch_prices", lambda *a, **k: ({}, None))
    assert cli.main(["score", "--log-file", log]) == 1
    closes = _market().close("SPXL")
    monkeypatch.setattr(data, "fetch_prices", lambda *a, **k: ({"SPXL": pd.DataFrame({"Close": closes})}, None))
    assert cli.main(["score", "--log-file", log]) == 0          # nothing to score yet is not a failure
