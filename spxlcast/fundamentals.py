"""Fundamental and macro drivers of the S&P 500, and the expected-return / volatility inputs.

No technical analysis is used anywhere: the drift comes from valuation (earnings yield, dividend
yield, earnings growth) and macro conditions (yield curve, credit, inflation, labour), and the
volatility comes from the options market (VIX term structure), not from price patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketSnapshot
from .data import _finite as _num      # a Yahoo info value as a finite float, or None


def _pct(v: Optional[float]) -> Optional[float]:
    """Yahoo/FRED quote yields in percent; convert to decimal."""
    return None if v is None or not np.isfinite(v) else float(v) / 100.0


def _calendar_months(series: Optional[pd.Series]) -> Optional[pd.Series]:
    """A monthly FRED series on a gap-free calendar of months, so positions are months, not rows.

    A single missing month (BLS published no October 2025 CPI or unemployment) is filled with the
    mean of its two neighbours; a longer hole stays NaN, so nothing is computed across it."""
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.reindex(pd.period_range(s.index[0], s.index[-1], freq="M"))
    hole = s.isna() & s.shift(1).notna() & s.shift(-1).notna()
    return s.where(~hole, (s.shift(1) + s.shift(-1)) / 2.0)


def cpi_yoy_from(series: Optional[pd.Series]) -> Optional[float]:
    """Year-over-year change of a monthly index (CPIAUCSL) from its latest month to the same month a
    year earlier, by date. None when that month is missing (beyond a one-month hole) or too early."""
    s = _calendar_months(series)
    if s is None or len(s) < 13:
        return None
    base = float(s.iloc[-13])
    if not np.isfinite(base) or base <= 0:
        return None
    return float(s.iloc[-1] / base - 1.0)


def sahm_gap_from(series: Optional[pd.Series]) -> Optional[float]:
    """Sahm gap of a monthly unemployment rate (in its own units, pp for UNRATE): the latest 3-month
    average minus the minimum of the 12 three-month averages before it, on calendar months. Windows
    that touch a hole longer than one month are skipped; None without the latest window."""
    s = _calendar_months(series)
    if s is None or len(s) < 15:
        return None
    roll = s.rolling(3).mean()
    prior = roll.iloc[-13:-1].dropna()
    if not np.isfinite(roll.iloc[-1]) or prior.empty:
        return None
    return float(roll.iloc[-1] - prior.min())


def discount_to_bey(d: float, days: int = 91) -> float:
    """Bank-discount T-bill rate (what ^IRX quotes) to bond-equivalent yield (what FRED DGS3MO quotes)."""
    return 365.0 * d / (360.0 - days * d)


@dataclass
class MacroState:
    rf_3m: float                       # short risk-free rate (decimal, bond-equivalent)
    y2: Optional[float]
    y5: Optional[float]
    y10: float
    y30: Optional[float]
    curve_10y_3m: Optional[float]      # None when the short rate had to be defaulted
    breakeven_10y: Optional[float]
    real_10y: Optional[float]
    sofr: Optional[float]
    hy_oas: Optional[float]
    unemployment: Optional[float]
    unemployment_sahm_gap: Optional[float]   # current 3m avg minus min of the prior 12 3m avgs (pp)
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
    eps_growth: float                  # nominal long-run per-share growth
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
def _vol_nowcast(snap: MarketSnapshot, ticker: str, lookback: int = 252) -> Optional[float]:
    """``ticker``'s last close, brought up to the VIX's newer session when it has no bar for it yet
    (at 09:40 ET Yahoo has today's VIX but not today's VIX3M/VIX6M), so all vol pillars are of one
    instant: scaled by the VIX's move since then to the power of the beta of its daily log changes
    on the VIX's over the last ``lookback`` sessions."""
    v = snap.fresh_last(ticker)
    s, vix = snap.close(ticker), snap.close("^VIX")
    if v is None or vix is None or snap.fresh_last("^VIX") is None or s.index[-1] >= vix.index[-1]:
        return v
    both = pd.concat({"vix": vix, "v": s}, axis=1, join="inner").iloc[-lookback - 1:]
    if not len(both) or both.index[-1] != s.index[-1] or (both <= 0).any().any():
        return v
    r = np.log(both).diff().dropna()
    if len(r) < 60 or not r.iloc[:, 0].var() > 0:
        return v
    beta = float(np.clip(r.cov().iloc[0, 1] / r.iloc[:, 0].var(), 0.0, 1.0))
    return v * (float(vix.iloc[-1]) / float(both.iloc[-1, 0])) ** beta


def build_macro(snap: MarketSnapshot, cfg: Config) -> MacroState:
    src: Dict[str, str] = {}

    def pick(fred_id: Optional[str], yahoo: Optional[str], label: str) -> Optional[float]:
        if fred_id:
            v = snap.fred_last(fred_id)
            if v is not None:
                src[label] = f"FRED:{fred_id}"
                return _pct(v)
        if yahoo:
            v = snap.fresh_last(yahoo)
            if v is not None:
                src[label] = f"Yahoo:{yahoo}"
                return _pct(v)
        return None

    y2 = pick("DGS2", "2YY=F", "y2")
    if src.get("y2") == "Yahoo:2YY=F":
        src["y2"] = "Yahoo:2YY=F (yield future, ~10-25bp below the 2y CMT)"
    y5 = pick(None, "^FVX", "y5")
    y10 = pick("DGS10", "^TNX", "y10")
    y30 = pick(None, "^TYX", "y30")

    rf_3m = _pct(snap.fred_last("DGS3MO"))
    curve_ok = True
    if rf_3m is not None:
        src["rf_3m"] = "FRED:DGS3MO"
    else:
        irx = snap.fresh_last("^IRX")
        if irx is not None and np.isfinite(irx) and 0 <= irx < 30:
            rf_3m = discount_to_bey(irx / 100.0)
            src["rf_3m"] = "Yahoo:^IRX (discount rate converted to bond-equivalent yield)"
        elif y2 is not None:
            rf_3m, src["rf_3m"] = y2, "proxy: 2y yield (3m bill unavailable)"
            curve_ok = False
        elif y10 is not None:
            rf_3m, src["rf_3m"] = y10, "proxy: 10y yield (3m bill unavailable)"
            curve_ok = False
        else:
            rf_3m, src["rf_3m"] = 0.04, "default 4%"
            curve_ok = False
    if y10 is None:
        y10, src["y10"] = rf_3m, "default (= short rate)"
        curve_ok = False
    curve = (y10 - rf_3m) if curve_ok else None
    if curve is None:
        src["curve_10y_3m"] = "unavailable (short rate defaulted); inversion signal skipped"

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
    if un is not None and len(un.dropna()) >= 15:
        unemployment = float(un.dropna().iloc[-1]) / 100.0
        sahm_gap = sahm_gap_from(un)
        src["unemployment"] = "FRED:UNRATE"

    cpi_yoy = None
    cpi = snap.fred.get("CPIAUCSL")
    if cpi is not None and len(cpi.dropna()) >= 13:
        cpi_yoy = cpi_yoy_from(cpi)
        src["cpi_yoy"] = ("FRED:CPIAUCSL" if cpi_yoy is not None
                          else "FRED:CPIAUCSL has no month a year before the latest; YoY skipped")

    vix3m, vix6m = _vol_nowcast(snap, "^VIX3M"), _vol_nowcast(snap, "^VIX6M")
    if vix3m != snap.fresh_last("^VIX3M") or vix6m != snap.fresh_last("^VIX6M"):
        src["vix_term"] = "Yahoo; 3M/6M moved with the VIX since their last close (no bar yet today)"

    return MacroState(
        rf_3m=rf_3m, y2=y2, y5=y5, y10=y10, y30=y30, curve_10y_3m=curve,
        breakeven_10y=breakeven, real_10y=real_10y, sofr=sofr, hy_oas=hy_oas,
        unemployment=unemployment, unemployment_sahm_gap=sahm_gap, cpi_yoy=cpi_yoy,
        vix=snap.fresh_last("^VIX"), vix3m=vix3m, vix6m=vix6m,
        vvix=snap.fresh_last("^VVIX"), skew=snap.fresh_last("^SKEW"),
        dxy=snap.fresh_last("DX-Y.NYB"), oil=snap.fresh_last("CL=F"), gold=snap.fresh_last("GC=F"),
        sources=src,
    )


def expected_inflation(macro: MacroState, cfg: Config) -> float:
    return macro.breakeven_10y if macro.breakeven_10y is not None else cfg.expected_inflation_default


def build_fundamentals(snap: MarketSnapshot, cfg: Config, inflation: Optional[float] = None) -> IndexFundamentals:
    src: Dict[str, str] = {}
    info = snap.info(cfg.index_etf)
    stats = snap.equity_stats or {}
    inflation = cfg.expected_inflation_default if inflation is None else inflation

    trailing_pe = cfg.override_trailing_pe
    if trailing_pe is not None:
        src["trailing_pe"] = "override"
    else:
        pe = _num(info.get("trailingPE"))
        if pe is not None and pe > 0:
            trailing_pe = pe
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
        dy = _num(info.get("yield"))
        if dy is not None and 0 < dy < 0.2:
            dividend_yield = dy
            src["dividend_yield"] = f"Yahoo:{cfg.index_etf}.info.yield"
        else:
            dy = _num(info.get("dividendYield"))
            if dy is not None and dy > 0:
                dividend_yield = dy / 100.0 if dy > 0.2 else dy
                src["dividend_yield"] = f"Yahoo:{cfg.index_etf}.info.dividendYield"
            else:
                dividend_yield = 0.013
                src["dividend_yield"] = "default 1.3%"

    if cfg.override_eps_growth is not None:
        eps_growth = cfg.override_eps_growth
        src["eps_growth"] = "override"
    else:
        eps_growth = cfg.long_run_real_eps_growth + inflation
        src["eps_growth"] = f"real {cfg.long_run_real_eps_growth:.1%} + inflation {inflation:.2%}"

    return IndexFundamentals(
        index_level=snap.last(cfg.index), trailing_pe=trailing_pe, earnings_yield=earnings_yield,
        dividend_yield=dividend_yield, eps_growth=eps_growth,
        book_to_price=stats.get("book_to_price"), sales_to_price=stats.get("sales_to_price"),
        sources=src,
    )


# ---------------------------------------------------------------------------------------
def expected_index_return(fund: IndexFundamentals, macro: MacroState, cfg: Config) -> ExpectedReturn:
    """Blend two classic long-run models, then apply transparent, capped adjustments.

    * Earnings-yield model:   E[r] = earnings yield + expected inflation
      (real return ~ E/P; the breakeven adds the nominal component)
    * Dividend-growth model:  E[r] = dividend yield + long-run nominal EPS growth
    * Valuation term (countercyclical): E/P above its neutral level adds drift, below subtracts.
      It is deliberately absolute (not relative to bond yields) so interest rates are charged to
      SPXL only once, through the fund's financing cost.
    * Regime penalties (inverted curve, credit stress, hot inflation, weak labour) are small and
      capped in total: at a 6-month horizon they describe risk rather than forecast returns.
    """
    notes: List[str] = []
    infl = expected_inflation(macro, cfg)
    m_ey = fund.earnings_yield + infl
    m_dg = fund.dividend_yield + fund.eps_growth
    base = 0.5 * m_ey + 0.5 * m_dg
    adj: Dict[str, float] = {}

    if cfg.override_index_drift is not None:
        final = cfg.override_index_drift
        notes.append("index drift overridden from the command line")
        return ExpectedReturn(m_ey, m_dg, base, adj, final, notes)

    if cfg.use_macro_adjustments:
        if cfg.valuation_sensitivity > 0 and fund.earnings_yield > 0:
            a = float(np.clip((fund.earnings_yield - cfg.neutral_earnings_yield) * cfg.valuation_sensitivity,
                              -cfg.valuation_adj_cap, cfg.valuation_adj_cap))
            adj["valuation"] = a
            notes.append(f"E/P {fund.earnings_yield:.2%} vs neutral {cfg.neutral_earnings_yield:.2%} "
                         f"(P/E {1 / fund.earnings_yield:.1f} vs {1 / cfg.neutral_earnings_yield:.0f})")

        regime: Dict[str, float] = {}
        if macro.curve_10y_3m is not None and macro.curve_10y_3m < 0:
            depth = min(1.0, -macro.curve_10y_3m / 0.01)
            regime["inverted_yield_curve"] = -cfg.inverted_curve_penalty * depth
            notes.append(f"10y-3m curve inverted by {macro.curve_10y_3m:.2%}")
        if macro.hy_oas is not None and macro.hy_oas > cfg.hy_stress_level:
            regime["credit_stress"] = -float(min(macro.hy_oas - cfg.hy_stress_level, cfg.hy_stress_cap))
            notes.append(f"HY OAS {macro.hy_oas:.2%} above stress level {cfg.hy_stress_level:.2%}")
        if macro.cpi_yoy is not None and macro.cpi_yoy > cfg.inflation_hot_level:
            regime["hot_inflation"] = -float(min((macro.cpi_yoy - cfg.inflation_hot_level) * 0.5, cfg.inflation_adj_cap))
            notes.append(f"CPI YoY {macro.cpi_yoy:.2%} above {cfg.inflation_hot_level:.2%}")
        if macro.unemployment_sahm_gap is not None and macro.unemployment_sahm_gap >= 0.5:
            regime["labour_deterioration"] = -cfg.sahm_penalty
            notes.append(f"Unemployment 3m-avg exceeds the 12m-min by {macro.unemployment_sahm_gap:.2f}pp (Sahm)")
        total = sum(regime.values())
        if total < -cfg.regime_adj_cap:
            scale = cfg.regime_adj_cap / -total
            regime = {k: v * scale for k, v in regime.items()}
            notes.append(f"regime penalties scaled to the {cfg.regime_adj_cap:.1%} cap")
        adj.update(regime)

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
    one_year_vol: Optional[float] = None   # total vol to 252 days, also when ``daily`` stops before that

    def total_vol(self, horizon: int) -> float:
        """Annualised vol that reproduces the total variance to ``horizon`` days."""
        h = min(horizon, len(self.daily))
        var_daily_sum = float(np.sum((self.daily[:h] ** 2) / 252.0))
        return float(np.sqrt(var_daily_sum * 252.0 / h))


def _haircuts(cfg: Config):
    v = cfg.vrp_vol_points
    if v is None:
        return (0.0, 0.0, 0.0)
    if np.ndim(v) == 0:
        return (float(v),) * 3
    v = tuple(float(x) for x in v) or (0.0,)
    return (v + (v[-1],) * 3)[:3]


def vol_term_structure(macro: MacroState, snap: MarketSnapshot, cfg: Config, horizon: int) -> VolTermStructure:
    notes: List[str] = []
    if cfg.override_vol is not None:
        daily = np.full(horizon, float(cfg.override_vol))
        return VolTermStructure(daily, {horizon: float(cfg.override_vol)}, ["flat vol override"],
                                float(cfg.override_vol))

    h1, h3, h6 = _haircuts(cfg)
    close = snap.close(cfg.index_etf)
    rv = realized_vol(close, 63) if close is not None else None

    def implied(v: Optional[float], haircut_pts: float, fallback: float) -> float:
        if v is None or not np.isfinite(v) or v <= 0:
            return fallback
        return max(cfg.vol_floor, v / 100.0 - haircut_pts / 100.0)

    fallback = rv if rv is not None else cfg.long_run_vol
    if macro.vix is None:
        notes.append("VIX unavailable or stale: using realised vol as the 1-month pillar")
    s1 = implied(macro.vix, h1, fallback)
    s3 = implied(macro.vix3m, h3, s1)
    s6 = implied(macro.vix6m, h6, s3)
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
    notes.append(f"pillars after VRP haircuts of {h1:.0f}/{h3:.0f}/{h6:.0f} pts: "
                 + ", ".join(f"{h}d {s:.1%}" for h, s in pillars.items()))
    if rv is not None:
        notes.append(f"realised 3m vol of {cfg.index_etf}: {rv:.1%} (diagnostic only)")
    return VolTermStructure(daily[:horizon], pillars, notes, float(np.sqrt(np.mean(daily[:252] ** 2))))


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
