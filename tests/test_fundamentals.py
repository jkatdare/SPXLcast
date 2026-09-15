from datetime import datetime, timezone

import numpy as np
import pandas as pd

from spxlcast.config import Config
from spxlcast.data import MarketSnapshot
from spxlcast.etf import ETFParams, calibrate, expense_ratio_from_info
from spxlcast.fundamentals import (IndexFundamentals, MacroState, build_fundamentals, expected_index_return,
                                   vol_term_structure)


def _macro(**kw):
    base = dict(rf_3m=0.04, y2=0.042, y5=0.045, y10=0.045, y30=0.048, curve_10y_3m=0.005, breakeven_10y=0.023,
                real_10y=0.022, sofr=None, hy_oas=None, unemployment=None, unemployment_sahm_gap=None, cpi_yoy=None,
                vix=18.0, vix3m=20.0, vix6m=21.0, vvix=None, skew=None, dxy=None, oil=None, gold=None)
    base.update(kw)
    return MacroState(**base)


def _fund(ey=0.045, dy=0.013, g=0.055):
    return IndexFundamentals(index_level=5000, trailing_pe=1 / ey, earnings_yield=ey, dividend_yield=dy,
                             eps_growth=g, book_to_price=None, sales_to_price=None)


def test_expected_return_blend_without_adjustments():
    cfg = Config(use_macro_adjustments=False)
    e = expected_index_return(_fund(), _macro(), cfg)
    assert e.earnings_yield_model == 0.045 + 0.023
    assert e.dividend_growth_model == 0.013 + 0.055
    assert e.final == e.base == 0.5 * (0.068 + 0.068)
    assert e.adjustments == {}


def test_macro_adjustments_move_drift_in_the_right_direction():
    cfg = Config()
    neutral = expected_index_return(_fund(), _macro(), cfg)
    inverted = expected_index_return(_fund(), _macro(curve_10y_3m=-0.012), cfg)
    stressed = expected_index_return(_fund(), _macro(hy_oas=0.07), cfg)
    hot = expected_index_return(_fund(), _macro(cpi_yoy=0.06), cfg)
    weak_labour = expected_index_return(_fund(), _macro(unemployment_sahm_gap=0.6), cfg)
    assert inverted.final < neutral.final
    assert inverted.adjustments["inverted_yield_curve"] == -cfg.inverted_curve_penalty  # fully inverted
    assert stressed.final < neutral.final
    assert hot.final < neutral.final
    assert weak_labour.final < neutral.final
    cheap = expected_index_return(_fund(ey=0.07), _macro(), cfg)
    assert cheap.adjustments["equity_risk_premium"] > neutral.adjustments["equity_risk_premium"]


def test_override_drift():
    cfg = Config(override_index_drift=0.11)
    assert expected_index_return(_fund(), _macro(), cfg).final == 0.11


def test_vol_term_structure_matches_pillars():
    cfg = Config()
    snap = MarketSnapshot(asof=datetime.now(timezone.utc))
    vts = vol_term_structure(_macro(), snap, cfg, 252)
    assert len(vts.daily) == 252
    assert np.all(vts.daily >= cfg.vol_floor)
    # total vol to each pillar reproduces the (haircut) implied vol
    for h, s in vts.pillars.items():
        assert abs(vts.total_vol(h) - s) < 1e-9
    assert vts.pillars[21] == (18.0 - cfg.vrp_vol_points) / 100.0


def test_vol_override():
    vts = vol_term_structure(_macro(), MarketSnapshot(asof=datetime.now(timezone.utc)), Config(override_vol=0.2), 63)
    assert np.allclose(vts.daily, 0.2)


def test_calibration_recovers_leverage_and_costs():
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.Series(rng.normal(0.0004, 0.011, n))
    etf = 3.0 * idx - 0.10 / 252 + rng.normal(0, 0.0008, n)
    dates = pd.bdate_range("2023-01-02", periods=n)
    idx_px = pd.Series(100 * np.cumprod(1 + idx.values), index=dates)
    etf_px = pd.Series(50 * np.cumprod(1 + etf.values), index=dates)
    cal = calibrate(etf_px, idx_px, lookback=504)
    assert abs(cal.beta - 3.0) < 0.05
    assert cal.r2 > 0.99
    assert abs(cal.empirical_drag_annual - 0.10) < 0.05


def test_expense_ratio_parsing():
    assert expense_ratio_from_info({"netExpenseRatio": 0.84}, 0.0091) == (0.0084, "Yahoo info.netExpenseRatio")
    assert expense_ratio_from_info({"annualReportExpenseRatio": 0.0091}, 0.01) == (0.0091, "Yahoo info.annualReportExpenseRatio")
    assert expense_ratio_from_info({}, 0.0091) == (0.0091, "default")


def test_etf_params_drag_and_breakeven():
    p = ETFParams(leverage=3.0, expense_ratio=0.009, financing_rate=0.088, tracking_sd_daily=0.0, calibration=None)
    assert abs(p.theoretical_drag(0.16) - 3 * 0.0256) < 1e-12
    assert p.breakeven_index_return(0.16) > 0.05


def test_build_fundamentals_from_info():
    snap = MarketSnapshot(asof=datetime.now(timezone.utc))
    snap.infos = {"SPY": {"trailingPE": 25.0, "yield": 0.012}}
    snap.equity_stats = {"earnings_to_price": 0.041, "book_to_price": 0.19}
    f = build_fundamentals(snap, Config())
    assert f.trailing_pe == 25.0 and abs(f.earnings_yield - 0.04) < 1e-12 and f.dividend_yield == 0.012
    # fallbacks when info is empty
    snap.infos = {"SPY": {}}
    f2 = build_fundamentals(snap, Config())
    assert abs(f2.earnings_yield - 0.041) < 1e-12
    assert f2.dividend_yield == 0.013
