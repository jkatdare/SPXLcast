"""Leveraged-ETF mechanics for SPXL and its calibration against history.

Daily:  r_etf = L * r_index - (expense + financing) / 252 + tracking noise
where financing ~= (L - 1) * (short rate + swap spread): the fund gets L x exposure through swaps
and pays roughly the short rate plus a spread on the borrowed (L - 1) x notional.
The volatility drag of daily rebalancing (~ L(L-1)/2 * sigma^2 per year) is NOT a parameter:
it emerges automatically from compounding the daily leveraged returns in the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketSnapshot


@dataclass
class Calibration:
    beta: float                 # realised daily leverage vs the index proxy
    intercept_daily: float      # realised daily alpha (captures costs + noise)
    resid_sd_daily: float       # tracking noise
    r2: float
    n: int
    empirical_drag_annual: float  # -intercept * 252 (very noisy; diagnostic)


@dataclass
class ETFParams:
    leverage: float
    expense_ratio: float
    financing_rate: float       # annualised, already multiplied by (L-1)
    tracking_sd_daily: float
    calibration: Optional[Calibration]
    notes: List[str] = field(default_factory=list)

    @property
    def annual_cost(self) -> float:
        return self.expense_ratio + self.financing_rate

    @property
    def daily_cost(self) -> float:
        return self.annual_cost / 252.0

    def theoretical_drag(self, sigma_annual: float) -> float:
        """Approximate annual volatility decay of the leveraged fund at index vol ``sigma``."""
        return 0.5 * self.leverage * (self.leverage - 1.0) * sigma_annual ** 2

    def breakeven_index_return(self, sigma_annual: float) -> float:
        """Index total return needed for the fund to roughly break even over a year."""
        return (self.annual_cost + self.theoretical_drag(sigma_annual)) / self.leverage


def calibrate(etf_close: pd.Series, index_close: pd.Series, lookback: int) -> Optional[Calibration]:
    r_etf = etf_close.pct_change()
    r_idx = index_close.pct_change()
    df = pd.concat([r_etf.rename("etf"), r_idx.rename("idx")], axis=1).dropna().iloc[-lookback:]
    if len(df) < 60:
        return None
    x = df["idx"].values
    y = df["etf"].values
    xm, ym = x.mean(), y.mean()
    beta = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    intercept = float(ym - beta * xm)
    resid = y - (intercept + beta * x)
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return Calibration(beta=beta, intercept_daily=intercept, resid_sd_daily=float(resid.std(ddof=2)),
                       r2=r2, n=int(len(df)), empirical_drag_annual=-intercept * 252.0)


def expense_ratio_from_info(info: dict, default: float) -> tuple[float, str]:
    for key in ("netExpenseRatio", "annualReportExpenseRatio", "expenseRatio"):
        v = info.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        # Yahoo reports 0.84 for 0.84% in newer fields and 0.0084 in older ones.
        return (v / 100.0 if v > 0.2 else v), f"Yahoo info.{key}"
    return default, "default"


def build_etf_params(snap: MarketSnapshot, cfg: Config, rf_short: float) -> ETFParams:
    notes: List[str] = []
    expense, src = expense_ratio_from_info(snap.info(cfg.etf), cfg.expense_ratio_default)
    notes.append(f"expense ratio {expense:.2%} ({src})")

    cal = None
    etf_close, idx_close = snap.close(cfg.etf), snap.close(cfg.index_etf)
    if etf_close is not None and idx_close is not None:
        cal = calibrate(etf_close, idx_close, cfg.calibration_lookback_days)

    leverage = cfg.leverage_target
    tracking = 0.0
    if cal is not None:
        # Trust the stated leverage unless the realised beta is clearly different.
        if abs(cal.beta - cfg.leverage_target) < 0.25:
            leverage = cfg.leverage_target
        else:
            leverage = cal.beta
            notes.append(f"realised beta {cal.beta:.2f} differs from target; using realised")
        tracking = cal.resid_sd_daily
        notes.append(f"calibrated on {cal.n} days: beta {cal.beta:.3f}, R2 {cal.r2:.3f}, "
                     f"tracking noise {cal.resid_sd_daily:.3%}/day, empirical drag {cal.empirical_drag_annual:+.1%}/yr (noisy)")
    else:
        notes.append("calibration unavailable: using stated leverage and zero tracking noise")

    financing = (leverage - 1.0) * (rf_short + cfg.swap_spread)
    notes.append(f"financing {(leverage - 1.0):.0f}x ({rf_short:.2%} + {cfg.swap_spread:.2%} spread) = {financing:.2%}/yr")
    return ETFParams(leverage=leverage, expense_ratio=expense, financing_rate=financing,
                     tracking_sd_daily=tracking, calibration=cal, notes=notes)
