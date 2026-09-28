"""Leveraged-ETF mechanics for SPXL and its calibration against history.

Daily:  r_etf = L * r_index - (expense + financing) / 252 + tracking noise
where financing ~= (L - 1) * (short rate + swap spread): the fund gets L x exposure through
total-return swaps and pays roughly the short rate plus a spread on the borrowed (L - 1) x notional
while its cash collateral earns the bill rate. Replicating SPXL from SPY total return over 1-5 years
confirms this structure; the spread is an all-in residual calibrated so the replication matches.
The volatility drag of daily rebalancing (~ L(L-1)/2 * sigma^2 per year) is NOT a parameter:
it emerges automatically from compounding the daily leveraged returns in the simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketSnapshot


@dataclass
class Calibration:
    beta: float                 # realised daily leverage vs the index proxy (robust fit)
    intercept_daily: float      # realised daily alpha (captures costs + noise)
    resid_sd_daily: float       # tracking noise on ordinary days
    r2: float
    n: int
    n_outliers: int             # dislocation days excluded from the fit (close-vs-NAV noise)
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
        """Arithmetic annual index total return at which the fund's expected log growth is zero: its
        long-run break-even.

        The fund's log return is L*ln(1+r) - cost - 0.5*L^2*sigma^2 per year with r the arithmetic
        index return, so it is zero on average when ln(1+r) = cost/L + 0.5*L*sigma^2; at a constant vol
        the median is then flat too, while with the simulation's bursts of volatility the 1-year median
        sits higher and only approaches flat over longer holds. (Using the
        geometric drag L(L-1)/2*sigma^2 alone understates this by 0.5*sigma^2, the index's own
        Ito term.)
        """
        return math.exp(self.annual_cost / self.leverage + 0.5 * self.leverage * sigma_annual ** 2) - 1.0


def _ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    beta = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    return beta, float(ym - beta * xm)


def calibrate(etf_close: pd.Series, index_close: pd.Series, lookback: int,
              outlier_mads: float = 5.0) -> Optional[Calibration]:
    """Robust regression of daily ETF returns on daily index returns.

    A plain OLS is dominated by a handful of days where the ETF's closing print sits away from its
    NAV (e.g. April 2025); those days drag beta down and inflate the tracking noise several-fold.
    Residuals beyond ``outlier_mads`` robust standard deviations are excluded and the fit repeated.
    A zero or non-finite close is a missing day, not a -100% day followed by an infinite one.
    """
    r_etf = etf_close.where(np.isfinite(etf_close) & (etf_close > 0)).pct_change(fill_method=None)
    r_idx = index_close.where(np.isfinite(index_close) & (index_close > 0)).pct_change(fill_method=None)
    df = pd.concat([r_etf.rename("etf"), r_idx.rename("idx")], axis=1).dropna().iloc[-lookback:]
    if len(df) < 60:
        return None
    x = df["idx"].values
    y = df["etf"].values
    beta, intercept = _ols(x, y)
    resid = y - (intercept + beta * x)
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    mask = np.abs(resid) <= outlier_mads * mad if mad > 0 else np.ones(len(resid), dtype=bool)
    n_out = int((~mask).sum())
    if n_out and mask.sum() >= 60:
        beta, intercept = _ols(x[mask], y[mask])
        resid = y - (intercept + beta * x)
    elif n_out:                      # too few inliers to refit: report the plain fit consistently
        mask = np.ones(len(resid), dtype=bool)
        n_out = 0
    inl = resid[mask]
    ss_tot = float(((y[mask] - y[mask].mean()) ** 2).sum())
    r2 = 1.0 - float((inl ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return Calibration(beta=beta, intercept_daily=intercept, resid_sd_daily=float(inl.std(ddof=2)),
                       r2=r2, n=int(len(df)), n_outliers=n_out, empirical_drag_annual=-intercept * 252.0)


def expense_ratio_from_info(info: dict, default: float) -> tuple[float, str]:
    """Yahoo's ``netExpenseRatio`` is always in percent (0.84 -> 0.84%); the legacy keys were
    reported as decimals by older yfinance versions, so those keep a magnitude heuristic."""
    for key in ("netExpenseRatio", "annualReportExpenseRatio", "expenseRatio"):
        v = info.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        if key == "netExpenseRatio" or v > 0.2:
            v = v / 100.0
        return v, f"Yahoo info.{key}"
    return default, "default"


def build_etf_params(snap: MarketSnapshot, cfg: Config, rf_short: float) -> ETFParams:
    notes: List[str] = []
    expense, src = expense_ratio_from_info(snap.info(cfg.etf), cfg.expense_ratio_default)
    notes.append(f"expense ratio {expense:.2%} ({src})")

    cal = None
    etf_close, idx_close = snap.close(cfg.etf), snap.close(cfg.index_etf)
    if etf_close is not None and idx_close is not None:
        cal = calibrate(etf_close, idx_close, cfg.calibration_lookback_days, cfg.calibration_outlier_mads)

    leverage = cfg.leverage_target
    tracking = 0.0
    if cal is not None and not (np.isfinite(cal.beta) and np.isfinite(cal.resid_sd_daily)
                                and cal.r2 >= cfg.calibration_min_r2):
        # A 3x fund tracks its index almost exactly, so a poor fit means bad or mixed-basis prices: its
        # beta and noise must not drive the simulation (a run without a calibration is flagged).
        notes.append(f"calibration rejected (beta {cal.beta:.3f}, R2 {cal.r2:.3f} < {cfg.calibration_min_r2:.2f}, "
                     f"tracking noise {cal.resid_sd_daily:.3%}/day on {cal.n} days): check the price history; "
                     f"using stated leverage and zero tracking noise")
        cal = None
    elif cal is not None:
        # Trust the stated leverage unless the realised beta is clearly different.
        if abs(cal.beta - cfg.leverage_target) < 0.25:
            leverage = cfg.leverage_target
        else:
            leverage = cal.beta
            notes.append(f"realised beta {cal.beta:.2f} differs from target; using realised")
        tracking = cal.resid_sd_daily
        notes.append(f"calibrated on {cal.n} days ({cal.n_outliers} dislocation days excluded): beta {cal.beta:.3f}, "
                     f"R2 {cal.r2:.3f}, tracking noise {cal.resid_sd_daily:.3%}/day, "
                     f"empirical drag {cal.empirical_drag_annual:+.1%}/yr (noisy)")
    else:
        notes.append("calibration unavailable: using stated leverage and zero tracking noise")

    financing = (leverage - 1.0) * (rf_short + cfg.swap_spread)
    notes.append(f"financing {(leverage - 1.0):.0f}x ({rf_short:.2%} + {cfg.swap_spread:.2%} all-in spread) = {financing:.2%}/yr")
    return ETFParams(leverage=leverage, expense_ratio=expense, financing_rate=financing,
                     tracking_sd_daily=tracking, calibration=cal, notes=notes)
