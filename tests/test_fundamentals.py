import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from spxlcast.config import Config
from spxlcast.data import MarketSnapshot
from spxlcast.etf import ETFParams, calibrate, expense_ratio_from_info
from spxlcast.fundamentals import (IndexFundamentals, MacroState, build_fundamentals, build_macro,
                                   discount_to_bey, expected_index_return, vol_term_structure)


def _macro(**kw):
    base = dict(rf_3m=0.04, y2=0.042, y5=0.045, y10=0.045, y30=0.048, curve_10y_3m=0.005, breakeven_10y=0.023,
                real_10y=0.022, sofr=None, hy_oas=None, unemployment=None, unemployment_sahm_gap=None, cpi_yoy=None,
                vix=18.0, vix3m=20.0, vix6m=21.0, vvix=None, skew=None, dxy=None, oil=None, gold=None)
    base.update(kw)
    return MacroState(**base)


def _fund(ey=0.045, dy=0.013, g=0.055):
    return IndexFundamentals(index_level=5000, trailing_pe=1 / ey, earnings_yield=ey, dividend_yield=dy,
                             eps_growth=g, book_to_price=None, sales_to_price=None)


def _snap(prices=None, fred=None, infos=None):
    snap = MarketSnapshot(asof=datetime.now(timezone.utc))
    snap.prices = prices or {}
    snap.fred = fred or {}
    snap.infos = infos or {}
    if "SPY" in snap.prices:
        snap.calendar = snap.prices["SPY"].index
    return snap


def _px(values, start="2026-01-02"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"Close": values}, index=idx)


def test_expected_return_blend_without_adjustments():
    cfg = Config(use_macro_adjustments=False)
    e = expected_index_return(_fund(), _macro(), cfg)
    assert e.earnings_yield_model == 0.045 + 0.023
    assert e.dividend_growth_model == 0.013 + 0.055
    assert e.final == e.base == 0.5 * (0.068 + 0.068)
    assert e.adjustments == {}


def test_valuation_term_is_countercyclical_and_absolute():
    cfg = Config()
    neutral = expected_index_return(_fund(ey=cfg.neutral_earnings_yield), _macro(), cfg)
    assert abs(neutral.adjustments["valuation"]) < 1e-12
    cheap = expected_index_return(_fund(ey=0.09), _macro(), cfg)
    mild = expected_index_return(_fund(ey=0.07), _macro(), cfg)
    dear = expected_index_return(_fund(ey=0.035), _macro(), cfg)
    assert cheap.adjustments["valuation"] == cfg.valuation_adj_cap        # capped
    assert abs(mild.adjustments["valuation"] - 0.5 * (0.07 - 0.05)) < 1e-12
    assert dear.adjustments["valuation"] < 0
    # rates do not enter the valuation term (they are charged through the ETF financing instead)
    high_real = expected_index_return(_fund(), _macro(real_10y=0.05, y10=0.075), cfg)
    assert high_real.adjustments["valuation"] == expected_index_return(_fund(), _macro(), cfg).adjustments["valuation"]


def test_regime_penalties_are_small_and_capped():
    cfg = Config()
    neutral = expected_index_return(_fund(), _macro(), cfg)
    inverted = expected_index_return(_fund(), _macro(curve_10y_3m=-0.012), cfg)
    stressed = expected_index_return(_fund(), _macro(hy_oas=0.07), cfg)
    hot = expected_index_return(_fund(), _macro(cpi_yoy=0.06), cfg)
    weak_labour = expected_index_return(_fund(), _macro(unemployment_sahm_gap=0.6), cfg)
    assert inverted.adjustments["inverted_yield_curve"] == -cfg.inverted_curve_penalty  # fully inverted
    assert stressed.adjustments["credit_stress"] == -cfg.hy_stress_cap
    assert hot.adjustments["hot_inflation"] == -cfg.inflation_adj_cap
    assert weak_labour.adjustments["labour_deterioration"] == -cfg.sahm_penalty
    for e in (inverted, stressed, hot, weak_labour):
        assert e.final < neutral.final
    everything = expected_index_return(_fund(), _macro(curve_10y_3m=-0.02, hy_oas=0.09, cpi_yoy=0.08,
                                                       unemployment_sahm_gap=1.0), cfg)
    regime_total = sum(v for k, v in everything.adjustments.items() if k != "valuation")
    assert abs(regime_total + cfg.regime_adj_cap) < 1e-12
    # no curve information => no inversion penalty
    unknown = expected_index_return(_fund(), _macro(curve_10y_3m=None), cfg)
    assert "inverted_yield_curve" not in unknown.adjustments


def test_override_drift():
    cfg = Config(override_index_drift=0.11)
    assert expected_index_return(_fund(), _macro(), cfg).final == 0.11


def test_vol_term_structure_matches_pillars_with_tenor_haircuts():
    cfg = Config()
    vts = vol_term_structure(_macro(), _snap(), cfg, 252)
    assert len(vts.daily) == 252
    assert np.all(vts.daily >= cfg.vol_floor)
    h1, h3, h6 = cfg.vrp_vol_points
    assert vts.pillars[21] == (18.0 - h1) / 100.0
    assert vts.pillars[63] == (20.0 - h3) / 100.0
    assert vts.pillars[126] == (21.0 - h6) / 100.0
    for h, s in vts.pillars.items():   # total vol to each pillar reproduces the haircut implied vol
        assert abs(vts.total_vol(h) - s) < 1e-9


def test_vol_override_and_scalar_haircut():
    vts = vol_term_structure(_macro(), _snap(), Config(override_vol=0.2), 63)
    assert np.allclose(vts.daily, 0.2)
    vts2 = vol_term_structure(_macro(), _snap(), Config(vrp_vol_points=2.0), 126)
    assert vts2.pillars[126] == (21.0 - 2.0) / 100.0


def test_discount_to_bond_equivalent_yield():
    assert abs(discount_to_bey(0.0396) - 0.04056) < 1e-4
    assert discount_to_bey(0.0) == 0.0


def test_build_macro_short_rate_fallbacks():
    n = 30
    spy = _px(np.linspace(500, 510, n))
    irx = _px(np.full(n, 3.96))
    tnx = _px(np.full(n, 5.0))
    macro = build_macro(_snap({"SPY": spy, "^IRX": irx, "^TNX": tnx}), Config())
    assert abs(macro.rf_3m - discount_to_bey(0.0396)) < 1e-12
    assert "bond-equivalent" in macro.sources["rf_3m"]
    assert macro.curve_10y_3m is not None and abs(macro.curve_10y_3m - (0.05 - macro.rf_3m)) < 1e-12
    # no short rate at all: the 2y future stands in and the curve signal is switched off
    macro2 = build_macro(_snap({"SPY": spy, "^TNX": tnx, "2YY=F": _px(np.full(n, 4.4))}), Config())
    assert abs(macro2.rf_3m - 0.044) < 1e-12 and macro2.curve_10y_3m is None
    macro3 = build_macro(_snap({"SPY": spy}), Config())
    assert macro3.rf_3m == 0.04 and macro3.curve_10y_3m is None and macro3.y10 == 0.04


def test_build_macro_ignores_stale_series():
    n = 40
    spy = _px(np.linspace(500, 510, n))
    vix = _px(np.full(20, 18.0))      # ends 20 sessions before SPY
    macro = build_macro(_snap({"SPY": spy, "^VIX": vix}), Config())
    assert macro.vix is None


def test_sahm_gap_uses_three_month_averages():
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    un = pd.Series([4.0] * 20 + [4.4, 4.6, 4.7, 4.8], index=months)
    macro = build_macro(_snap(fred={"UNRATE": un}), Config())
    roll = un.rolling(3).mean()
    assert abs(macro.unemployment_sahm_gap - (roll.iloc[-1] - roll.iloc[-13:-1].min())) < 1e-12
    assert macro.unemployment_sahm_gap > 0.5


def test_calibration_is_robust_to_dislocation_days():
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.Series(rng.normal(0.0004, 0.011, n))
    etf = 3.0 * idx - 0.10 / 252 + rng.normal(0, 0.0008, n)
    etf.iloc[100] += 0.03      # close-vs-NAV dislocation days
    etf.iloc[101] -= 0.03
    dates = pd.bdate_range("2023-01-02", periods=n)
    idx_px = pd.Series(100 * np.cumprod(1 + idx.values), index=dates)
    etf_px = pd.Series(50 * np.cumprod(1 + etf.values), index=dates)
    cal = calibrate(etf_px, idx_px, lookback=504)
    assert abs(cal.beta - 3.0) < 0.03
    assert cal.n_outliers >= 2
    assert cal.resid_sd_daily < 0.0012          # outliers do not inflate the tracking noise
    assert cal.r2 > 0.99
    assert abs(cal.empirical_drag_annual - 0.10) < 0.05


def test_expense_ratio_parsing():
    assert expense_ratio_from_info({"netExpenseRatio": 0.84}, 0.0091) == (0.0084, "Yahoo info.netExpenseRatio")
    assert expense_ratio_from_info({"netExpenseRatio": 0.0945}, 0.0091) == (0.000945, "Yahoo info.netExpenseRatio")
    assert expense_ratio_from_info({"annualReportExpenseRatio": 0.0091}, 0.01) == (0.0091, "Yahoo info.annualReportExpenseRatio")
    assert expense_ratio_from_info({}, 0.0091) == (0.0091, "default")


def test_etf_params_drag_and_breakeven():
    p = ETFParams(leverage=3.0, expense_ratio=0.009, financing_rate=0.088, tracking_sd_daily=0.0, calibration=None)
    assert abs(p.theoretical_drag(0.16) - 3 * 0.0256) < 1e-12
    # median break-even: exp(cost/L + 0.5*L*sigma^2) - 1
    assert abs(p.breakeven_index_return(0.16) - (math.exp(0.097 / 3 + 1.5 * 0.0256) - 1)) < 1e-12


def test_build_fundamentals_from_info():
    snap = _snap(infos={"SPY": {"trailingPE": 25.0, "yield": 0.012}})
    snap.equity_stats = {"earnings_to_price": 0.041, "book_to_price": 0.19}
    f = build_fundamentals(snap, Config(), inflation=0.02)
    assert f.trailing_pe == 25.0 and abs(f.earnings_yield - 0.04) < 1e-12 and f.dividend_yield == 0.012
    assert abs(f.eps_growth - (Config().long_run_real_eps_growth + 0.02)) < 1e-12
    # fallbacks when info is empty
    snap.infos = {"SPY": {}}
    f2 = build_fundamentals(snap, Config())
    assert abs(f2.earnings_yield - 0.041) < 1e-12
    assert f2.dividend_yield == 0.013
    f3 = build_fundamentals(snap, Config(override_eps_growth=0.07))
    assert f3.eps_growth == 0.07
