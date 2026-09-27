"""Regression tests for the model-group fixes of the break-it campaign (all offline)."""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import spxlcast.cli as cli
import spxlcast.pipeline as pipeline
from spxlcast.config import FRED_SERIES, Config
from spxlcast.data import MarketSnapshot
from spxlcast.fundamentals import (MacroState, build_fundamentals, build_macro, cpi_yoy_from,
                                   expected_index_return, sahm_gap_from, vol_term_structure)
from spxlcast.montecarlo import simulate

ASOF = datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc)
LAST = pd.Timestamp("2026-09-25")


# ---- a small synthetic market ------------------------------------------------------------
def _frame(values, idx):
    return pd.DataFrame({"Close": np.asarray(values, dtype=float)}, index=idx)


def _market(n=300, fred=True, spxl_lag=0, vol_lag=0):
    cal = pd.bdate_range(end=LAST, periods=n)
    rng = np.random.default_rng(0)
    r = 0.0003 + 0.011 * rng.standard_normal(n)
    r[0] = 0.0
    spy = 450.0 * np.cumprod(1 + r)
    spxl = 100.0 * np.cumprod(1 + 3 * r - 0.0001 + 0.0005 * rng.standard_normal(n))
    snap = MarketSnapshot(asof=ASOF, max_stale_sessions=5)
    snap.prices = {"SPY": _frame(spy, cal), "SPXL": _frame(spxl[:n - spxl_lag], cal[:n - spxl_lag]),
                   "^IRX": _frame(np.full(n, 4.0), cal), "^TNX": _frame(np.full(n, 4.2), cal),
                   "^VIX": _frame(np.full(n, 17.0), cal)}
    for t, v in (("^VIX3M", 19.0), ("^VIX6M", 20.5)):
        snap.prices[t] = _frame(np.full(n - vol_lag, v), cal[:n - vol_lag])
    snap.calendar = cal
    snap.fetched_at = {"prices": ASOF}
    snap.infos = {"SPXL": {"netExpenseRatio": 0.87}, "SPY": {"trailingPE": 27.0, "yield": 0.012}}
    if fred:
        daily = pd.bdate_range(end=LAST - pd.Timedelta(days=1), periods=400)
        levels = {"DGS3MO": 4.1, "DGS2": 3.9, "DGS10": 4.2, "DFII10": 1.9, "T10YIE": 2.3, "SOFR": 4.05,
                  "BAMLH0A0HYM2": 3.1}
        snap.fred = {k: pd.Series(v, index=daily) for k, v in levels.items()}
        months = pd.date_range(end="2026-08-01", periods=60, freq="MS")   # the August print, 55 days old
        snap.fred["UNRATE"] = pd.Series(4.2, index=months)
        snap.fred["CPIAUCSL"] = pd.Series(300 * 1.0025 ** np.arange(60), index=months)
        assert set(snap.fred) == set(FRED_SERIES)
    return snap


def _patch_market(monkeypatch, snap):
    monkeypatch.setattr(pipeline, "load_market", lambda cfg: snap)


def _cfg(**kw):
    base = dict(n_paths=400, horizons=(5, 21), rating_horizon=21, use_news=False)
    base.update(kw)
    return Config(**base)


# ---- R1-34: CPI YoY and the Sahm gap by calendar month ------------------------------------------
def _cpi(monthly_growth=0.0028, end="2026-08-01", n=40):
    months = pd.date_range(end=end, periods=n, freq="MS")
    return pd.Series(300 * (1 + monthly_growth) ** np.arange(n), index=months)


def test_cpi_yoy_is_twelve_months_back_by_date_across_a_missing_month():
    full = _cpi()
    gappy = full.drop(pd.Timestamp("2025-10-01"))          # BLS published no October 2025 CPI
    true = full.iloc[-1] / full.loc["2025-08-01"] - 1
    assert abs(cpi_yoy_from(full) - true) < 1e-12
    assert abs(cpi_yoy_from(gappy) - true) < 1e-12
    assert abs(gappy.iloc[-1] / gappy.iloc[-13] - 1 - true) > 2e-3   # the old row arithmetic spans 13 months


def test_cpi_yoy_base_month_hole():
    full = _cpi(end="2026-10-01")
    one = full.drop(pd.Timestamp("2025-10-01"))            # the base month itself is a one-month hole
    assert abs(cpi_yoy_from(one) - cpi_yoy_from(full)) < 1e-5  # filled with the mean of its neighbours
    two = full.drop([pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01")])
    assert cpi_yoy_from(two) is None                       # a longer hole is never bridged
    assert cpi_yoy_from(full.iloc[-12:]) is None
    assert cpi_yoy_from(None) is None and cpi_yoy_from(pd.Series(dtype=float)) is None


def test_build_macro_cpi_gap_fires_no_spurious_hot_inflation_penalty():
    cpi = _cpi(monthly_growth=0.0028).drop(pd.Timestamp("2025-10-01"))   # 3.41% a year, below the 3.5% level
    macro = build_macro(MarketSnapshot(asof=ASOF, fred={"CPIAUCSL": cpi}), Config())
    assert abs(macro.cpi_yoy - (1.0028 ** 12 - 1)) < 1e-12
    assert macro.sources["cpi_yoy"] == "FRED:CPIAUCSL"
    fund = build_fundamentals(MarketSnapshot(asof=ASOF), Config())
    assert "hot_inflation" not in expected_index_return(fund, macro, Config()).adjustments
    # no observation a year back: reported unavailable, not computed across the hole
    holed = _cpi().drop([pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")])
    macro2 = build_macro(MarketSnapshot(asof=ASOF, fred={"CPIAUCSL": holed}), Config())
    assert macro2.cpi_yoy is None and "skipped" in macro2.sources["cpi_yoy"]


def test_sahm_gap_uses_calendar_months():
    months = pd.date_range("2024-06-01", "2026-08-01", freq="MS")
    un = pd.Series(np.linspace(4.0, 4.9, len(months)), index=months)   # linear, so the hole is filled exactly
    gappy = un.drop(pd.Timestamp("2025-10-01"))
    assert abs(sahm_gap_from(gappy) - sahm_gap_from(un)) < 1e-12
    roll = un.rolling(3).mean()
    assert abs(sahm_gap_from(un) - (roll.iloc[-1] - roll.iloc[-13:-1].min())) < 1e-12
    groll = gappy.rolling(3).mean()
    assert abs((groll.iloc[-1] - groll.iloc[-13:-1].min()) - sahm_gap_from(un)) > 1e-3   # rows != months
    assert sahm_gap_from(un.iloc[-14:]) is None
    macro = build_macro(MarketSnapshot(asof=ASOF, fred={"UNRATE": gappy}), Config())
    assert abs(macro.unemployment_sahm_gap - sahm_gap_from(un)) < 1e-12 and macro.unemployment == 0.049


# ---- R1-42: non-numeric Yahoo info values ----------------------------------------------------
def test_non_numeric_info_values_fall_back():
    snap = MarketSnapshot(asof=ASOF, infos={"SPY": {"trailingPE": "Infinity", "yield": "n/a",
                                                    "dividendYield": float("inf")}})
    snap.equity_stats = {"earnings_to_price": 0.04}
    f = build_fundamentals(snap, Config())
    assert abs(f.earnings_yield - 0.04) < 1e-12 and "funds_data" in f.sources["trailing_pe"]
    assert f.dividend_yield == 0.013 and "default" in f.sources["dividend_yield"]
    snap.infos = {"SPY": {"trailingPE": "25", "yield": "0.012"}}      # numeric strings are numbers
    f2 = build_fundamentals(snap, Config())
    assert f2.trailing_pe == 25.0 and f2.dividend_yield == 0.012


# ---- R1-44: the 1-year break-even does not depend on the horizons --------------------------------
def test_one_year_vol_is_kept_when_the_curve_is_cut_short():
    macro = MacroState(rf_3m=0.04, y2=None, y5=None, y10=0.045, y30=None, curve_10y_3m=0.005, breakeven_10y=None,
                       real_10y=None, sofr=None, hy_oas=None, unemployment=None, unemployment_sahm_gap=None,
                       cpi_yoy=None, vix=25.0, vix3m=26.0, vix6m=27.0, vvix=None, skew=None, dxy=None, oil=None,
                       gold=None)
    snap = MarketSnapshot(asof=ASOF)
    full = vol_term_structure(macro, snap, Config(), 252)
    for T in (5, 126, 252, 504):
        assert abs(vol_term_structure(macro, snap, Config(), T).one_year_vol - full.total_vol(252)) < 1e-12
    assert vol_term_structure(macro, snap, Config(override_vol=0.2), 21).one_year_vol == 0.2


def test_rating_context_breakeven_is_independent_of_horizons(monkeypatch):
    _patch_market(monkeypatch, _market())
    needs = []
    for horizons, rh in (((5, 21, 252), 21), ((5,), 5)):
        fc = pipeline.run_forecast(_cfg(horizons=horizons, rating_horizon=rh))
        needs.append(next(x for x in fc.rating.reasons if "needs about" in x).split("needs about")[1])
    assert needs[0] == needs[1]


# ---- R1-08 / R1-07: ladder and price lookups at the spot --------------------------------------
@pytest.fixture(scope="module")
def short_forecast():
    snap = _market()
    orig = pipeline.load_market
    pipeline.load_market = lambda cfg: snap
    try:
        return pipeline.run_forecast(_cfg(n_paths=20000, horizons=(5, 10, 21, 126), rating_horizon=126))
    finally:
        pipeline.load_market = orig


def test_limit_ladder_p_fill_matches_the_simulation(short_forecast):
    fc = short_forecast
    for h in fc.sim.horizons:
        mins = fc.sim.path_min[h]
        rows = fc.limit_ladder(h)
        prices = [r["price"] for r in rows]
        assert len(prices) == len(set(prices))                # no two rungs at one price
        for r in rows:
            assert abs(float(np.mean(mins <= r["price"])) - r["p_fill"]) < 0.01
            assert r["price"] <= fc.spot
    rows5 = fc.limit_ladder(5)
    assert np.mean(fc.sim.path_min[5] >= fc.spot) > 0.1       # the atom at the spot exists at 1W
    assert rows5[0] == {**rows5[0], "p_fill": 1.0, "price": fc.spot, "vs_spot": 0.0}
    assert [r["p_fill"] for r in fc.limit_ladder(126)] == [0.9, 0.75, 0.5, 0.25, 0.1]


def test_price_lookup_at_the_spot_is_touched(short_forecast):
    fc = short_forecast
    for r in fc.price_lookup(fc.spot):
        assert r["p_touch_below"] == 1.0 and r["p_touch_above"] == 1.0


# ---- R1-41 / R2-04: data-quality flags for degraded inputs ------------------------------------
def _flags(monkeypatch, snap, **kw):
    _patch_market(monkeypatch, snap)
    return pipeline.run_forecast(_cfg(n_paths=200, **kw)).data_quality()["flags"]


def test_clean_market_has_no_flags(monkeypatch):
    assert _flags(monkeypatch, _market()) == []


def test_stale_inputs_are_flagged(monkeypatch):
    snap = _market()
    snap.fred["DGS3MO"] = snap.fred["DGS3MO"].iloc[:-30]           # stopped updating six weeks ago
    assert _flags(monkeypatch, snap) == ["fred:stale"]
    snap = _market()
    snap.fred["CPIAUCSL"] = snap.fred["CPIAUCSL"].iloc[:-3]        # latest print dated May: 147 days old
    assert _flags(monkeypatch, snap) == ["fred:stale"]
    assert "spot:stale" in _flags(monkeypatch, _market(spxl_lag=3))
    assert "spot:stale" not in _flags(monkeypatch, _market(spxl_lag=1))
    assert _flags(monkeypatch, _market(n=50)) == ["etf:uncalibrated"]


def test_vol_pillars_from_an_older_session_are_flagged(monkeypatch):
    # 09:40 ET: today's VIX is live but Yahoo has no bar yet for today's VIX3M / VIX6M
    assert _flags(monkeypatch, _market(vol_lag=1)) == ["vix3m:stale", "vix6m:stale"]


# ---- R1-43: CLI input validation ---------------------------------------------------------------
@pytest.mark.parametrize("argv", [
    ["forecast", "--price", "nan"], ["forecast", "--price", "inf"], ["forecast", "--price", "0"],
    ["price", "nan"], ["forecast", "--seed", "-1"], ["forecast", "--index-drift", "-0.5"],
    ["forecast", "--index-drift", "nan"], ["forecast", "--vol", "inf"], ["forecast", "--pe", "nan"],
    ["forecast", "--skew", "inf"],
])
def test_cli_rejects_non_finite_and_out_of_range_inputs(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_cli_accepts_the_edges():
    args = cli.build_parser().parse_args(["price", "190.5", "--seed", "0", "--index-drift", "-0.49"])
    assert args.prices == [190.5] and args.seed == 0 and args.index_drift == -0.49


def test_lowest_accepted_drift_simulates_finite_paths():
    # the per-path drift draw is clipped to +/-50 points, so any drift above -50% stays above -100%
    sim = simulate(spot=100.0, mu_annual=np.full(5, -0.49), sigma_annual=np.full(5, 0.2), leverage=3.0,
                   daily_cost=0.0, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[5], n_paths=2000,
                   seed=1, drift_sd_annual=5.0)
    assert np.all(np.isfinite(sim.terminal[5]))


# ---- R1-39 / R1-09: outputs survive a rendering failure and are replaced atomically -------------
def _run_cli(monkeypatch, tmp_path, extra=()):
    _patch_market(monkeypatch, _market())
    argv = ["forecast", "--paths", "200", "--horizons", "5", "21", "--rating-horizon", "21", "--no-news",
            "--price", "90", "--log-file", str(tmp_path / "log.csv"), "--json", str(tmp_path / "out" / "forecast.json"),
            "--plot", str(tmp_path / "out" / "fan.png"), *extra]
    return cli.main(argv)


def test_render_failure_still_writes_the_outputs(monkeypatch, tmp_path, capsys):
    from rich.errors import MarkupError

    def boom(*a, **k):
        raise MarkupError("closing tag '[/]' at position 42 has nothing to close")
    monkeypatch.setattr(cli, "render_all", boom)
    assert _run_cli(monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert "could not be rendered" in out and "[/]" in out
    doc = json.loads((tmp_path / "out" / "forecast.json").read_text(encoding="utf-8"))
    assert doc["etf"] == "SPXL" and "90.0" in doc["price_lookup"]
    assert (tmp_path / "out" / "fan.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(pd.read_csv(tmp_path / "log.csv")) == 1
    assert sorted(os.listdir(tmp_path / "out")) == ["fan.png", "forecast.json"]   # no temp files left


def test_forecast_json_is_never_truncated_while_the_new_one_is_built(monkeypatch, tmp_path):
    path = tmp_path / "out" / "forecast.json"
    path.parent.mkdir()
    path.write_text('{"old": true}', encoding="utf-8")
    seen = []
    real = cli.forecast_to_dict

    def spy(fc, prices=None):
        seen.append(path.read_text(encoding="utf-8"))           # what the live loop would read right now
        return real(fc, prices)
    monkeypatch.setattr(cli, "forecast_to_dict", spy)
    assert _run_cli(monkeypatch, tmp_path) == 0
    assert seen == ['{"old": true}']
    assert json.loads(path.read_text(encoding="utf-8"))["etf"] == "SPXL"


def test_write_atomic_keeps_the_old_file_when_the_writer_fails(tmp_path):
    path = tmp_path / "fan.png"
    path.write_bytes(b"old")

    def fail(fh):
        fh.write(b"partial")
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError):
        cli._write_atomic(str(path), fail)
    assert path.read_bytes() == b"old" and os.listdir(tmp_path) == ["fan.png"]


def test_write_atomic_retries_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    path = tmp_path / "forecast.json"
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("in use")   # Windows, while the live loop reads the file
        real(src, dst)
    monkeypatch.setattr(cli.os, "replace", flaky)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cli._write_atomic(str(path), lambda fh: fh.write(b"{}"))
    assert path.read_bytes() == b"{}" and len(calls) == 3 and os.listdir(tmp_path) == ["forecast.json"]
