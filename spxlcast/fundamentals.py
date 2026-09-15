"""Fundamental and macro drivers of the S&P 500, and the expected-return / volatility inputs.

No technical analysis is used anywhere: the drift comes from valuation (earnings yield, dividend
yield, earnings growth) and macro conditions (real rates, yield curve, credit, inflation, labour),
and the volatility comes from the options market (VIX term structure), not from price patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketSnapshot


def _pct(v: Optional[float]) -> Optional[float]:
    """Yahoo/FRED quote yields in percent; convert to decimal."""
    return None if v is None or not np.isfinite(v) else float(v) / 100.0


@dataclass
class MacroState:
    rf_3m: float                       # short risk-free rate (decimal)
    y2: Optional[float]
    y5: Optional[float]
    y10: float
    y30: Optional[float]
    curve_10y_3m: float
    breakeven_10y: Optional[float]
    real_10y: Optional[float]
    sofr: Optional[float]
    hy_oas: Optional[float]
    unemployment: Optional[float]
    unemployment_sahm_gap: Optional[float]   # 3m avg minus 12m min (percentage points)
    cpi_yoy: Optional[float]
    vix: Optional[float]                # in vol points (17.5 == 17.5%)
    vix3m: Optional[float]
    vix6m: Optional[float]
    vvix: Optional[float]
    skew: Optional[float]
    dxy: Optional[float]
    oil: Optional[float]
    gold: Optional[float]
    sources: Dict[str, str] = field(default_factory=dict)


@dataclass
class IndexFundamentals:
    index_level: Optional[float]
    trailing_pe: Optional[float]
    earnings_yield: float
    dividend_yield: float
    eps_growth: float
    book_to_price: Optional[float]
    sales_to_price: Optional[float]
    sources: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExpectedReturn:
    """Annualised nominal total-return expectation for the S&P 500 and how it was built."""
    earnings_yield_model: float
    dividend_growth_model: float
    base: float
    adjustments: Dict[str, float]
    final: float
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------
def build_macro(snap: MarketSnapshot, cfg: Config) -> MacroState:
    src: Dict[str, str] = {}

    def pick(fred_id: Optional[str], yahoo: Optional[str], label: str) -> Optional[float]:
        if fred_id:
            v = snap.fred_last(fred_id)
            if v is not None:
                src[label] = f"FRED:{fred_id}"
                return _pct(v)
        if yahoo:
            v = snap.last(yahoo)
            if v is not None:
                src[label] = f"Yahoo:{yahoo}"
                return _pct(v)
        return None

    rf_3m = pick("DGS3MO", "^IRX", "rf_3m")
    if rf_3m is None:
        rf_3m = 0.04
        src["rf_3m"] = "default"
    y10 = pick("DGS10", "^TNX", "y10")
    if y10 is None:
        y10 = rf_3m
        src["y10"] = "default(=rf_3m)"
    y2 = pick("DGS2", "2YY=F", "y2")
    y5 = pick(None, "^FVX", "y5")
    y30 = pick(None, "^TYX", "y30")
    breakeven = pick("T10YIE", None, "breakeven_10y")
    real_10y = pick("DFII10", None, "real_10y")
    if real_10y is None:
        real_10y = y10 - (breakeven if breakeven is not None else cfg.expected_inflation_default)
        src["real_10y"] = "derived: y10 - breakeven (or default inflation)"
    sofr = pick("SOFR", None, "sofr")
    hy_oas = pick("BAMLH0A0HYM2", None, "hy_oas")

    unemployment = None
    sahm_gap = None
    un = snap.fred.get("UNRATE")
    if un is not None and len(un.dropna()) >= 12:
        un = un.dropna()
        unemployment = float(un.iloc[-1]) / 100.0
        sahm_gap = float(un.iloc[-3:].mean() - un.iloc[-12:].min())
        src["unemployment"] = "FRED:UNRATE"

    cpi_yoy = None
    cpi = snap.fred.get("CPIAUCSL")
    if cpi is not None and len(cpi.dropna()) >= 13:
        cpi = cpi.dropna()
        cpi_yoy = float(cpi.iloc[-1] / cpi.iloc[-13] - 1.0)
        src["cpi_yoy"] = "FRED:CPIAUCSL"

    return MacroState(
        rf_3m=rf_3m, y2=y2, y5=y5, y10=y10, y30=y30, curve_10y_3m=y10 - rf_3m,
        breakeven_10y=breakeven, real_10y=real_10y, sofr=sofr, hy_oas=hy_oas,
        unemployment=unemployment, unemployment_sahm_gap=sahm_gap, cpi_yoy=cpi_yoy,
        vix=snap.last("^VIX"), vix3m=snap.last("^VIX3M"), vix6m=snap.last("^VIX6M"),
        vvix=snap.last("^VVIX"), skew=snap.last("^SKEW"),
        dxy=snap.last("DX-Y.NYB"), oil=snap.last("CL=F"), gold=snap.last("GC=F"),
        sources=src,
    )


def build_fundamentals(snap: MarketSnapshot, cfg: Config) -> IndexFundamentals:
    src: Dict[str, str] = {}
    info = snap.info(cfg.index_etf)
    stats = snap.equity_stats or {}

    trailing_pe = cfg.override_trailing_pe
    if trailing_pe is not None:
        src["trailing_pe"] = "override"
    else:
        pe = info.get("trailingPE")
        if pe and np.isfinite(pe) and pe > 0:
            trailing_pe = float(pe)
            src["trailing_pe"] = f"Yahoo:{cfg.index_etf}.info.trailingPE"
        elif stats.get("earnings_to_price"):
            trailing_pe = 1.0 / stats["earnings_to_price"]
            src["trailing_pe"] = f"Yahoo:{cfg.index_etf}.funds_data (1/E-P)"

    if trailing_pe is not None and trailing_pe > 0:
        earnings_yield = 1.0 / trailing_pe
    elif stats.get("earnings_to_price"):
        earnings_yield = float(stats["earnings_to_price"])
        src["earnings_yield"] = "Yahoo funds_data"
    else:
        earnings_yield = 0.045
        src["earnings_yield"] = "default 4.5%"

    dividend_yield = cfg.override_dividend_yield
    if dividend_yield is not None:
        src["dividend_yield"] = "override"
    else:
        dy = info.get("yield")
        if dy is not None and np.isfinite(dy) and 0 < dy < 0.2:
            dividend_yield = float(dy)
            src["dividend_yield"] = f"Yahoo:{cfg.index_etf}.info.yield"
        else:
            dy = info.get("dividendYield")
            if dy is not None and np.isfinite(dy) and dy > 0:
                dividend_yield = float(dy) / 100.0 if dy > 0.2 else float(dy)
                src["dividend_yield"] = f"Yahoo:{cfg.index_etf}.info.dividendYield"
            else:
                dividend_yield = 0.013
                src["dividend_yield"] = "default 1.3%"

    eps_growth = cfg.long_run_eps_growth
    src["eps_growth"] = f"config long_run_eps_growth={eps_growth:.3f}"

    return IndexFundamentals(
        index_level=snap.last(cfg.index), trailing_pe=trailing_pe, earnings_yield=earnings_yield,
        dividend_yield=dividend_yield, eps_growth=eps_growth,
        book_to_price=stats.get("book_to_price"), sales_to_price=stats.get("sales_to_price"),
        sources=src,
    )


# ---------------------------------------------------------------------------------------
def expected_index_return(fund: IndexFundamentals, macro: MacroState, cfg: Config) -> ExpectedReturn:
    """Blend two classic long-run models, then apply transparent macro adjustments.

    * Earnings-yield model:   E[r] = earnings yield + expected inflation
      (real return ~ E/P; the breakeven adds the nominal component)
    * Dividend-growth model:  E[r] = dividend yield + long-run nominal EPS growth
    """
    notes: List[str] = []
    infl = macro.breakeven_10y if macro.breakeven_10y is not None else cfg.expected_inflation_default
    m_ey = fund.earnings_yield + infl
    m_dg = fund.dividend_yield + fund.eps_growth
    base = 0.5 * m_ey + 0.5 * m_dg
    adj: Dict[str, float] = {}

    if cfg.override_index_drift is not None:
        final = cfg.override_index_drift
        notes.append("index drift overridden from the command line")
        return ExpectedReturn(m_ey, m_dg, base, adj, final, notes)

    if cfg.use_macro_adjustments:
        # 1) Equity risk premium versus real yields: cheap vs. bonds => higher drift, and vice versa.
        if macro.real_10y is not None:
            erp = fund.earnings_yield - macro.real_10y
            a = float(np.clip((erp - cfg.neutral_erp) * cfg.erp_sensitivity, -cfg.erp_adj_cap, cfg.erp_adj_cap))
            adj["equity_risk_premium"] = a
            notes.append(f"ERP (E/P minus real 10y) = {erp:+.2%} vs neutral {cfg.neutral_erp:.2%}")
        # 2) Yield-curve inversion is the classic recession signal.
        if macro.curve_10y_3m < 0:
            depth = min(1.0, -macro.curve_10y_3m / 0.01)
            adj["inverted_yield_curve"] = -cfg.inverted_curve_penalty * depth
            notes.append(f"10y-3m curve inverted by {macro.curve_10y_3m:.2%}")
        # 3) Credit stress.
        if macro.hy_oas is not None and macro.hy_oas > cfg.hy_stress_level:
            adj["credit_stress"] = -float(min(macro.hy_oas - cfg.hy_stress_level, cfg.hy_stress_cap))
            notes.append(f"HY OAS {macro.hy_oas:.2%} above stress level {cfg.hy_stress_level:.2%}")
        # 4) Hot inflation raises the odds of tighter policy.
        if macro.cpi_yoy is not None and macro.cpi_yoy > cfg.inflation_hot_level:
            adj["hot_inflation"] = -float(min((macro.cpi_yoy - cfg.inflation_hot_level) * 0.5, cfg.inflation_adj_cap))
            notes.append(f"CPI YoY {macro.cpi_yoy:.2%} above {cfg.inflation_hot_level:.2%}")
        # 5) Sahm-rule style labour deterioration.
        if macro.unemployment_sahm_gap is not None and macro.unemployment_sahm_gap >= 0.5:
            adj["labour_deterioration"] = -cfg.sahm_penalty
            notes.append(f"Unemployment 3m-avg exceeds 12m-min by {macro.unemployment_sahm_gap:.2f}pp")

    final = float(np.clip(base + sum(adj.values()), cfg.drift_floor, cfg.drift_cap))
    return ExpectedReturn(m_ey, m_dg, base, adj, final, notes)


# ---------------------------------------------------------------------------------------
def realized_vol(close: pd.Series, days: int) -> Optional[float]:
    r = np.log(close).diff().dropna()
    if len(r) < max(10, days // 2):
        return None
    return float(r.iloc[-days:].std(ddof=1) * np.sqrt(252.0))


@dataclass
class VolTermStructure:
    """Annualised vol for each future trading day, built from the VIX curve."""
    daily: np.ndarray               # shape (T,), annualised vol applying on day t
    pillars: Dict[int, float]       # horizon days -> implied vol (after the VRP haircut)
    notes: List[str] = field(default_factory=list)

    def total_vol(self, horizon: int) -> float:
        """Annualised vol that reproduces the total variance to ``horizon`` days."""
        h = min(horizon, len(self.daily))
        var_daily_sum = float(np.sum((self.daily[:h] ** 2) / 252.0))
        return float(np.sqrt(var_daily_sum * 252.0 / h))


def vol_term_structure(macro: MacroState, snap: MarketSnapshot, cfg: Config, horizon: int) -> VolTermStructure:
    notes: List[str] = []
    if cfg.override_vol is not None:
        daily = np.full(horizon, float(cfg.override_vol))
        return VolTermStructure(daily, {horizon: float(cfg.override_vol)}, ["flat vol override"])

    haircut = cfg.vrp_vol_points / 100.0
    close = snap.close(cfg.index_etf)
    rv = realized_vol(close, 63) if close is not None else None

    def implied(v: Optional[float], fallback: float) -> float:
        if v is None or not np.isfinite(v) or v <= 0:
            return fallback
        return max(cfg.vol_floor, v / 100.0 - haircut)

    fallback = rv if rv is not None else cfg.long_run_vol
    if macro.vix is None:
        notes.append("VIX unavailable: using realised vol as the 1-month pillar")
    s1 = implied(macro.vix, fallback)
    s3 = implied(macro.vix3m, s1)
    s6 = implied(macro.vix6m, s3)
    s12 = float(np.sqrt(0.5 * s6 ** 2 + 0.5 * cfg.long_run_vol ** 2))  # mean-revert toward long run
    pillars = {21: s1, 63: s3, 126: s6, 252: s12}

    # Forward variances between pillars so that total variance to each pillar matches.
    n = max(horizon, 252)
    daily = np.empty(n)
    prev_h, prev_var = 0, 0.0
    floor_var = cfg.vol_floor ** 2
    for h, s in pillars.items():
        total_var = s ** 2 * h
        fwd_var = max(floor_var, (total_var - prev_var) / (h - prev_h))
        daily[prev_h:h] = np.sqrt(fwd_var)
        prev_var = prev_var + fwd_var * (h - prev_h)
        prev_h = h
    daily[252:] = daily[251]
    notes.append(f"pillars after {cfg.vrp_vol_points:.1f}pt VRP haircut: "
                 + ", ".join(f"{h}d {s:.1%}" for h, s in pillars.items()))
    if rv is not None:
        notes.append(f"realised 3m vol of {cfg.index_etf}: {rv:.1%} (diagnostic only)")
    return VolTermStructure(daily[:horizon], pillars, notes)


# ---------------------------------------------------------------------------------------
@dataclass
class Sensitivity:
    driver: str
    unit: str            # what a "+1" shock of the driver means: "+1%", "+1pp" or "+1pt"
    beta: float          # SPXL fractional move per unit shock (0.03 == +3%)
    r2: float
    n: int


def influencer_sensitivities(snap: MarketSnapshot, cfg: Config, lookback: int = 252) -> List[Sensitivity]:
    """OLS beta of daily SPXL returns on daily moves of each influencer (descriptive statistics).

    Betas are normalised so that ``beta`` is the fractional SPXL move for a one-unit shock:
    +1% return of a price driver, +1 percentage point of a yield, +1 point of the VIX.
    """
    etf = snap.close(cfg.etf)
    if etf is None:
        return []
    y = etf.pct_change().dropna().iloc[-lookback:]
    drivers = {
        "SPY total return": ("SPY", "ret"),
        "10y Treasury yield": ("^TNX", "yield"),
        "3m T-bill yield": ("^IRX", "yield"),
        "VIX": ("^VIX", "points"),
        "US Dollar Index": ("DX-Y.NYB", "ret"),
        "High-yield bonds (HYG)": ("HYG", "ret"),
        "Long Treasuries (TLT)": ("TLT", "ret"),
        "WTI crude": ("CL=F", "ret"),
        "Gold": ("GC=F", "ret"),
    }
    out: List[Sensitivity] = []
    for label, (ticker, kind) in drivers.items():
        s = snap.close(ticker)
        if s is None:
            continue
        if kind == "ret":
            x, unit = s.pct_change() * 100.0, "+1%"          # shock unit: 1% return
        elif kind == "yield":
            x, unit = s.diff(), "+1pp"                       # Yahoo quotes yields in %, so diff is in pp
        else:
            x, unit = s.diff(), "+1pt"
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if len(df) < 60 or df["x"].std() == 0:
            continue
        beta = float(np.cov(df["x"], df["y"], ddof=1)[0, 1] / df["x"].var(ddof=1))
        r2 = float(np.corrcoef(df["x"], df["y"])[0, 1] ** 2)
        out.append(Sensitivity(label, unit, beta, r2, int(len(df))))
    return out
