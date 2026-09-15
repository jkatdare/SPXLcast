"""Configuration for SPXLcast.

Every modelling assumption lives here so it can be inspected and overridden from the CLI.
Rates, yields and returns are expressed as decimals (0.05 == 5%) unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class Config:
    # --- Universe -----------------------------------------------------------------
    etf: str = "SPXL"                 # the leveraged fund we forecast
    index_etf: str = "SPY"            # total-return proxy for the S&P 500 (dividends included)
    index: str = "^GSPC"              # S&P 500 price index (for display)
    leverage_target: float = 3.0      # the fund's stated daily leverage

    # --- Horizons (trading days) ----------------------------------------------------
    horizons: Tuple[int, ...] = (21, 63, 126, 252)
    rating_horizon: int = 126         # the rating is judged on this horizon

    # --- Monte Carlo ----------------------------------------------------------------
    n_paths: int = 20_000
    seed: Optional[int] = 42
    t_dof: float = 4.0                # Student-t degrees of freedom for daily index shocks (fat tails)
    max_daily_move: float = 0.20      # index circuit-breaker: a single day cannot move more than +/-20%

    # --- Expected-return model for the S&P 500 (annualised, nominal, total return) -------
    long_run_eps_growth: float = 0.055      # nominal earnings growth used in the dividend-growth model
    expected_inflation_default: float = 0.025  # used when a market breakeven is unavailable
    neutral_erp: float = 0.035              # "normal" earnings-yield minus real-10y spread
    erp_sensitivity: float = 0.5            # drift adj per 1.00 of ERP deviation (capped below)
    erp_adj_cap: float = 0.015
    inverted_curve_penalty: float = 0.02    # full penalty when 10y-3m <= -1.0%
    hy_stress_level: float = 0.05           # HY OAS above this is treated as credit stress
    hy_stress_cap: float = 0.03
    inflation_hot_level: float = 0.035      # CPI YoY above this is a headwind
    inflation_adj_cap: float = 0.015
    sahm_penalty: float = 0.02              # unemployment 3m-avg >= 12m-min + 0.5 => recession signal
    drift_floor: float = -0.10
    drift_cap: float = 0.20
    use_macro_adjustments: bool = True

    # --- Volatility -------------------------------------------------------------------
    vrp_vol_points: float = 2.0       # implied vol usually exceeds realised vol by ~2 points
    vol_floor: float = 0.08
    long_run_vol: float = 0.16        # S&P 500 long-run annualised vol, used beyond the VIX curve

    # --- Leveraged ETF costs -----------------------------------------------------------
    expense_ratio_default: float = 0.0091   # used if Yahoo does not report one
    swap_spread: float = 0.004              # financing spread over the short rate on swap notional
    calibration_lookback_days: int = 504    # ~2 years for the beta / tracking calibration

    # --- News sentiment --------------------------------------------------------------
    use_news: bool = True
    news_tickers: Tuple[str, ...] = ("SPXL", "SPY", "^GSPC")
    news_holdings_top_n: int = 10           # also read news for the top-N S&P 500 constituents
    news_half_life_days: float = 2.0
    news_max_age_days: float = 7.0
    sentiment_drift_scale: float = 0.10     # annualised drift shift when the sentiment score is +/-1
    sentiment_days: int = 21                # ... applied to the first N trading days only

    # --- Rating ----------------------------------------------------------------------
    buy_score: float = 0.30
    sell_score: float = -0.30
    edge_scale: float = 0.15                # annualised excess return that maps to score = 1
    prob_scale: float = 0.10                # P(beat T-bill) - 0.5 that maps to score = 1

    # --- Data ------------------------------------------------------------------------
    history_period: str = "5y"
    cache_dir: str = ".cache"
    price_ttl_hours: float = 6.0
    info_ttl_hours: float = 12.0
    news_ttl_hours: float = 1.0
    use_fred: bool = True
    fred_timeout: float = 8.0
    refresh: bool = False               # ignore the cache

    # --- Overrides (None => derived from data) ---------------------------------------
    override_index_drift: Optional[float] = None   # annual expected S&P 500 total return
    override_vol: Optional[float] = None           # flat annualised vol
    override_dividend_yield: Optional[float] = None
    override_trailing_pe: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Tickers used as "influencers": things that move SPXL. All available on Yahoo Finance.
MARKET_TICKERS = {
    "SPXL": "Direxion Daily S&P 500 Bull 3X",
    "SPY": "SPDR S&P 500 ETF (total-return proxy)",
    "^GSPC": "S&P 500 index",
    "^VIX": "VIX (30-day implied vol, %)",
    "^VIX3M": "VIX 3-month (%)",
    "^VIX6M": "VIX 6-month (%)",
    "^VVIX": "VVIX (vol of vol)",
    "^SKEW": "CBOE SKEW (tail-risk pricing)",
    "^IRX": "13-week T-bill yield (%)",
    "2YY=F": "2-year Treasury yield (%)",
    "^FVX": "5-year Treasury yield (%)",
    "^TNX": "10-year Treasury yield (%)",
    "^TYX": "30-year Treasury yield (%)",
    "DX-Y.NYB": "US Dollar Index",
    "HYG": "High-yield corporate bond ETF",
    "LQD": "Investment-grade corporate bond ETF",
    "TLT": "20y+ Treasury ETF",
    "CL=F": "WTI crude ($/bbl)",
    "GC=F": "Gold ($/oz)",
}

# Optional FRED series (fetched only if reachable). Values are in percent unless noted.
FRED_SERIES = {
    "DGS3MO": "3-month Treasury (%)",
    "DGS2": "2-year Treasury (%)",
    "DGS10": "10-year Treasury (%)",
    "DFII10": "10-year TIPS real yield (%)",
    "T10YIE": "10-year breakeven inflation (%)",
    "SOFR": "SOFR (%)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS (%)",
    "UNRATE": "Unemployment rate (%)",
    "CPIAUCSL": "CPI-U (index)",
}
