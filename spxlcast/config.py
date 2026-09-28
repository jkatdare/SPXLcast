"""Configuration for SPXLcast.

Every modelling assumption lives here so it can be inspected and overridden from the CLI.
Rates, yields and returns are expressed as decimals (0.05 == 5%) unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time, timedelta
from functools import lru_cache
from typing import FrozenSet, Optional, Tuple

# Version of the forecasting model, logged with every track-record row so results can be scored per
# version. Bump it whenever a change alters the numbers a run produces (inputs, assumptions, the
# simulation, the rating); pure plumbing, reporting or extra horizons do not need a bump.
# tests/test_golden.py fails when the numbers on its made-up market change and this does not (caps,
# floors, cut-offs and fallbacks that market never reaches are not checked).
MODEL_VERSION = "0.3.0"


@dataclass
class Config:
    # --- Universe -----------------------------------------------------------------
    etf: str = "SPXL"                 # the leveraged fund we forecast
    index_etf: str = "SPY"            # total-return proxy for the S&P 500 (dividends included)
    index: str = "^GSPC"              # S&P 500 price index (for display)
    leverage_target: float = 3.0      # the fund's stated daily leverage

    # --- Horizons (trading days) ----------------------------------------------------
    # 1W and 2W are logged for fast feedback: ~50 independent weekly outcomes a year against 12
    # monthly ones, and the news tilt only acts over the first 10 days. Extra horizons do not change
    # the other horizons' numbers (the simulation's random stream depends only on the longest).
    horizons: Tuple[int, ...] = (5, 10, 21, 63, 126, 252)
    rating_horizon: int = 126         # the rating is judged on this horizon

    # --- Monte Carlo ----------------------------------------------------------------
    n_paths: int = 50_000
    seed: Optional[int] = 42
    t_dof: float = 4.0                # Student-t degrees of freedom for daily index shocks (fat tails)
    skew_gamma: float = 0.9           # Fernandez-Steel skew of the daily shock; 1.0 = symmetric, <1 = negative skew
    max_daily_move: float = 0.20      # index circuit-breaker: a single day cannot move more than +/-20%
    drift_uncertainty_sd: float = 0.02   # sd of the per-path expected index return (0 = drift treated as known)
    # Stochastic volatility (backtested 2016-2026 with the haircuts below): vol clustering with a
    # leverage effect fixes the shape of the distribution and the drawdown-touch probabilities.
    sv_persistence: float = 0.97      # AR(1) persistence of the log-vol deviation (0 = off)
    sv_logvol_sd: float = 0.35        # stationary sd of the log-vol deviation (0 = off)
    sv_leverage: float = -0.5         # corr(today's return shock, tomorrow's vol innovation)

    # --- Expected-return model for the S&P 500 (annualised, nominal, total return) -------
    long_run_real_eps_growth: float = 0.03     # real per-share earnings growth; nominal = this + expected inflation
    expected_inflation_default: float = 0.025  # used when a market breakeven is unavailable
    # Optional countercyclical valuation term, 0.5 x (E/P - neutral) capped. Off by default: over
    # 1881-2026 (scripts/backtest_drift.py) it added no predictive value to the blend and biased the
    # post-1990 era low, because the market re-rated to structurally higher valuations.
    neutral_earnings_yield: float = 0.05       # trailing E/P at which the valuation term is zero (P/E 20)
    valuation_sensitivity: float = 0.0         # drift adj per 1.00 of E/P deviation from neutral (0 = off)
    valuation_adj_cap: float = 0.015
    # Risk-regime descriptors. They are small and capped in total: at a 6-month horizon these signals
    # describe risk, they do not forecast returns (inverted curves and wide spreads often precede rallies).
    inverted_curve_penalty: float = 0.01       # full penalty when 10y-3m <= -1.0%
    hy_stress_level: float = 0.05              # HY OAS above this is treated as credit stress
    hy_stress_cap: float = 0.015
    inflation_hot_level: float = 0.035         # CPI YoY above this is a headwind
    inflation_adj_cap: float = 0.01
    sahm_penalty: float = 0.01                 # 3m-avg unemployment >= 12m-min of 3m-avg + 0.5pp
    regime_adj_cap: float = 0.015              # cap on the sum of the regime penalties
    drift_floor: float = -0.10
    drift_cap: float = 0.20
    use_macro_adjustments: bool = True

    # --- Volatility -------------------------------------------------------------------
    # Implied vol exceeds subsequently realised vol by more at longer tenors (variance risk premium):
    # haircut in vol points for the 1-month, 3-month and 6-month pillars. Backtested 2016-2026
    # together with the stochastic-vol settings above (with stochastic vol off, 2/4/6 fits better).
    vrp_vol_points: Tuple[float, float, float] = (3.0, 5.0, 7.0)
    vol_floor: float = 0.08
    long_run_vol: float = 0.16        # S&P 500 long-run annualised vol, used beyond the VIX curve
    max_stale_sessions: int = 5       # ignore a market series whose last observation is older than this

    # --- Leveraged ETF costs -----------------------------------------------------------
    expense_ratio_default: float = 0.0091   # used if Yahoo does not report one
    swap_spread: float = 0.0075             # all-in financing spread over the 3m bill on the borrowed (L-1)x notional;
                                            # calibrated so that 3x SPY minus costs reproduces SPXL over 1-5 years
    calibration_lookback_days: int = 504    # ~2 years for the beta / tracking calibration
    calibration_outlier_mads: float = 5.0   # residuals beyond this many robust SDs are excluded from the fit
    calibration_min_r2: float = 0.99        # below this the price history is suspect (clean 2021-2026 fits: >= 0.998)

    # --- News sentiment --------------------------------------------------------------
    use_news: bool = True
    news_tickers: Tuple[str, ...] = ("SPXL", "SPY", "^GSPC", "^VIX", "^TNX")   # market-wide feeds
    news_holdings_top_n: int = 10           # also read news for the top-N S&P 500 constituents
    news_half_life_days: float = 2.0
    news_max_age_days: float = 7.0
    summary_weight: float = 0.2             # share of the article score taken from the teaser summary
    offtopic_weight: float = 0.25           # relevance of a story that is not about the market or the company
    sentiment_drift_scale: float = 0.05     # annualised drift shift when the sentiment score is +/-1 (uncalibrated)
    sentiment_days: int = 10                # ... applied to the first N trading days only

    # --- Rating ----------------------------------------------------------------------
    buy_score: float = 0.30
    sell_score: float = -0.30
    edge_scale: float = 0.15                # annualised median excess return that maps to score = 1
    sharpe_scale: float = 0.5               # annualised mean-excess-return / vol that maps to score = 1
    score_blocks: int = 20                  # path blocks used for the Monte Carlo standard error of the score

    # --- Data ------------------------------------------------------------------------
    history_period: str = "5y"
    cache_dir: str = ".cache"
    price_ttl_hours: float = 6.0           # after the close (the key also changes at the bell)
    price_ttl_hours_open: float = 0.5      # while the session is open, so hourly runs see a fresh quote
    info_ttl_hours: float = 12.0
    news_ttl_hours: float = 0.5            # well under the hourly job period, so every run reads fresh headlines
    use_fred: bool = True
    fred_timeout: float = 8.0
    refresh: bool = False               # ignore the cache

    # --- Overrides (None => derived from data) ---------------------------------------
    override_index_drift: Optional[float] = None   # annual expected S&P 500 total return
    override_vol: Optional[float] = None           # flat annualised vol
    override_dividend_yield: Optional[float] = None
    override_trailing_pe: Optional[float] = None
    override_eps_growth: Optional[float] = None    # nominal long-run EPS growth

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
    "^IRX": "13-week T-bill, discount basis (%)",
    "2YY=F": "2-year yield future, CBOT front month (%)",
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


# NYSE trading calendar (standard library only, so the health check can use it too). Holidays follow
# NYSE Rule 7.2: a Saturday holiday is observed on Friday and a Sunday one on Monday, except that a
# Saturday New Year's Day is not moved into the old year. Unscheduled closures are listed by hand.
NYSE_SPECIAL_CLOSURES = (date(2012, 10, 29), date(2012, 10, 30), date(2018, 12, 5), date(2025, 1, 9))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th ``weekday`` (Monday = 0) of the month; n = -1 is the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    h = (19 * a + b - b // 4 - (b - (b + 8) // 25 + 1) // 3 + 15) % 30
    l = (32 + 2 * (b % 4) + 2 * (c // 4) - h - c % 4) % 7
    m = (a + 11 * h + 22 * l) // 451
    return date(year, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)


def _observed(d: date) -> date:
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


@lru_cache(maxsize=None)
def _nyse_days(year: int) -> Tuple[FrozenSet[date], FrozenSet[date]]:
    """(full-day holidays, 13:00 early closes) of one year."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    holidays = {
        _nth_weekday(year, 1, 0, 3),            # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),            # Washington's Birthday
        _easter(year) - timedelta(days=2),      # Good Friday
        _nth_weekday(year, 5, 0, -1),           # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),            # Labor Day
        thanksgiving,
        _observed(date(year, 12, 25)),
    }
    if date(year, 1, 1).weekday() != 5:
        holidays.add(_observed(date(year, 1, 1)))
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))   # Juneteenth
    holidays.update(d for d in NYSE_SPECIAL_CLOSURES if d.year == year)
    # 3 July and 24 December close early on Monday to Thursday; on a Friday they are the observed holiday.
    early = {thanksgiving + timedelta(days=1)} | {d for d in (date(year, 7, 3), date(year, 12, 24)) if d.weekday() < 4}
    return frozenset(holidays), frozenset(early - holidays)


def nyse_session(day: date) -> Optional[Tuple[time, time]]:
    """Regular-session (open, close) New York times on ``day``, or None when NYSE is closed all day."""
    holidays, early = _nyse_days(day.year)
    if day.weekday() >= 5 or day in holidays:
        return None
    return time(9, 30), time(13, 0) if day in early else time(16, 0)
