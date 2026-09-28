"""End-to-end pipeline: data -> drivers -> simulation -> assessment (and the retired rating, still
logged for research), plus a JSON-able summary."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .assess import DIP_HORIZON, Assessment, assess, assessment_to_dict
from .config import FRED_SERIES, MODEL_VERSION, Config
from .data import MarketSnapshot, load_market
from .env import build_id
from .etf import ETFParams, build_etf_params
from .fundamentals import (ExpectedReturn, IndexFundamentals, MacroState, Sensitivity, VolTermStructure,
                           build_fundamentals, build_macro, expected_index_return, expected_inflation,
                           influencer_sensitivities, vol_term_structure)
from .live import GRID_PERCENTILES
from .montecarlo import SimulationResult, simulate
from .rating import Rating, rate
from .sentiment import SentimentResult, analyze_news


MONTHLY_FRED_MAX_AGE_DAYS = 100   # monthly series are dated the 1st and published 5-10 weeks later


def _stale_fred(snap: MarketSnapshot) -> List[str]:
    """FRED series that stopped updating: a daily series more than ``max_stale_sessions`` sessions
    behind the trading calendar, a monthly one older than MONTHLY_FRED_MAX_AGE_DAYS."""
    if snap.calendar is None or not len(snap.calendar):
        return []
    end = pd.Timestamp(snap.calendar[-1])
    out = []
    for sid, s in snap.fred.items():
        s = s.dropna() if s is not None else None
        if s is None or len(s) < 2:
            continue
        last = pd.Timestamp(s.index[-1])
        if pd.Series(s.index[-6:]).diff().median() > pd.Timedelta(days=20):   # a monthly series
            stale = (end.date() - last.date()).days > MONTHLY_FRED_MAX_AGE_DAYS
        else:
            stale = snap._sessions_between(last, end) > snap.max_stale_sessions
        if stale:
            out.append(sid)
    return out


@dataclass
class Forecast:
    cfg: Config
    snap: MarketSnapshot
    spot: float
    spot_date: str
    spot_status: str                      # "close" or "intraday"
    macro: MacroState
    fundamentals: IndexFundamentals
    expected: ExpectedReturn
    vol: VolTermStructure
    etf: ETFParams
    sentiment: Optional[SentimentResult]
    sim: SimulationResult
    rating: Rating                        # no longer shown: logged for continuity and research
    sensitivities: List[Sensitivity] = field(default_factory=list)
    mu_path: Optional[np.ndarray] = None
    # The exact arguments passed to montecarlo.simulate, so an archived run can be replayed.
    sim_inputs: Optional[Dict[str, Any]] = None
    rating_sim_inputs: Optional[Dict[str, Any]] = None   # the untilted run behind the rating, when separate
    assessment: Optional[Assessment] = None              # leverage cost, drawdown risk, 3-month range (assess.py)

    def data_quality(self) -> Dict[str, Any]:
        """Problems with this run's inputs (``flags`` empty = clean), logged with the track record."""
        flags: List[str] = []
        snap = self.snap
        n_fred = len(snap.fred)
        if self.cfg.use_fred:
            if n_fred == 0:
                flags.append("fred:none")
            elif n_fred < len(FRED_SERIES):
                flags.append("fred:partial")
            if _stale_fred(snap):
                flags.append("fred:stale")
        spot_date = snap.last_date(self.cfg.etf)
        if spot_date is not None and snap.calendar is not None and len(snap.calendar) \
                and snap._sessions_between(spot_date, snap.calendar[-1]) > 1:
            flags.append("spot:stale")   # one session behind is still the latest close (index shows today's bar)
        pillars = [(name, snap.last_date(ticker), value) for name, ticker, value in (
            ("vix", "^VIX", self.macro.vix), ("vix3m", "^VIX3M", self.macro.vix3m), ("vix6m", "^VIX6M", self.macro.vix6m))]
        # every pillar should be as of the newest of the spot and the pillars themselves (the spot can lag too)
        ref = max((d for d in [spot_date] + [d for _, d, v in pillars if v is not None] if d is not None), default=None)
        for name, last, value in pillars:
            if value is None:
                flags.append(f"{name}:missing")
            elif ref is not None and last is not None and last.date() < ref.date():
                flags.append(f"{name}:stale")   # e.g. at 09:40 ET Yahoo has no bar yet for today's VIX3M/VIX6M
        if self.etf.calibration is None:
            flags.append("etf:uncalibrated")
        src = self.fundamentals.sources
        if "default" in src.get("earnings_yield", ""):
            flags.append("earnings:default")
        elif "funds_data" in src.get("trailing_pe", ""):
            flags.append("pe:funds_data")
        if "default" in src.get("dividend_yield", ""):
            flags.append("dividend:default")
        if any(n.startswith("No price history for") for n in self.snap.notes):
            flags.append("prices:partial")
        news_fetched = None
        if self.cfg.use_news:
            news_fetched = self.sentiment.n_articles if self.sentiment else 0
            if self.sentiment is None or self.sentiment.n_used == 0:
                flags.append("news:none")
        return {"flags": flags, "fred_series": n_fred, "news_fetched": news_fetched}

    # ---- helpers used by the report and JSON export ------------------------------------
    def price_lookup(self, price: float) -> List[Dict[str, float]]:
        rows = []
        for h in self.sim.horizons:
            rows.append({
                "horizon": h,
                "price": price,
                "vs_spot": price / self.spot - 1.0,
                "percentile": self.sim.percentile_of_price(price, h),
                "p_touch_below": self.sim.prob_touch_below(price, h),
                "p_touch_above": self.sim.prob_touch_above(price, h),
                "p_end_above": 1.0 - self.sim.percentile_of_price(price, h) / 100.0,
            })
        return rows

    def limit_ladder(self, horizon: int, probs=(0.9, 0.75, 0.5, 0.25, 0.1)) -> List[Dict[str, float]]:
        """Buy-limit prices with the given probability of being touched before ``horizon``.

        Paths that never trade below the start have a path minimum of exactly the spot. When they are
        more than 1 - p of all paths (short horizons), no price below the spot fills with probability p:
        those targets collapse into a single at-market rung at the spot, which fills on every path."""
        mins = self.sim.path_min[horizon]
        ends = self.sim.terminal[horizon]
        p_below = float(np.mean(mins < self.spot))
        rows: List[Dict[str, float]] = []
        at_market = False
        for p in probs:
            if p >= p_below:
                if at_market:
                    continue
                at_market, level, p = True, self.spot, 1.0
            else:
                level = float(np.percentile(mins, 100.0 * p))  # P(min <= level) = p
            filled = ends[mins <= level]                    # only the paths where the order fills
            cond_median = float(np.median(filled)) if len(filled) else float("nan")
            rows.append({"p_fill": p, "price": level, "vs_spot": level / self.spot - 1.0,
                         "median_end_if_bought": cond_median / level - 1.0,
                         "p_profit_if_bought": float(np.mean(filled > level)) if len(filled) else float("nan")})
        return rows


def run_forecast(cfg: Config) -> Forecast:
    snap = load_market(cfg)
    spot = snap.last(cfg.etf)
    if spot is None:
        raise RuntimeError(f"No price history for {cfg.etf}; check the network connection")
    spot_date = str(snap.last_date(cfg.etf).date())
    spot_status = "intraday" if snap.intraday else "close"

    macro = build_macro(snap, cfg)
    fund = build_fundamentals(snap, cfg, inflation=expected_inflation(macro, cfg))
    expected = expected_index_return(fund, macro, cfg)

    horizons = sorted(set(int(h) for h in cfg.horizons) | {int(cfg.rating_horizon), DIP_HORIZON})
    T = max(horizons)
    vol = vol_term_structure(macro, snap, cfg, T)
    etf = build_etf_params(snap, cfg, macro.rf_3m)

    sentiment = None
    mu = np.full(T, expected.final)
    if cfg.use_news:
        weights, names = {}, {}
        if snap.holdings is not None:
            weights = dict(zip(snap.holdings["symbol"], snap.holdings["weight"]))
            names = dict(zip(snap.holdings["symbol"], snap.holdings["name"]))
        sentiment = analyze_news(snap.news, cfg, weights, now=snap.asof, holdings_names=names)
        n = min(cfg.sentiment_days, T)
        mu[:n] += sentiment.drift_adjustment

    def _sim_inputs(mu_path: np.ndarray) -> Dict[str, Any]:
        return dict(
            spot=spot, mu_annual=mu_path, sigma_annual=vol.daily, leverage=etf.leverage,
            daily_cost=etf.daily_cost, tracking_sd_daily=etf.tracking_sd_daily, rf_annual=macro.rf_3m,
            horizons=horizons, n_paths=cfg.n_paths, dof=cfg.t_dof, skew_gamma=cfg.skew_gamma,
            max_daily_move=cfg.max_daily_move, seed=cfg.seed, drift_sd_annual=cfg.drift_uncertainty_sd,
            sv_persistence=cfg.sv_persistence, sv_logvol_sd=cfg.sv_logvol_sd, sv_leverage=cfg.sv_leverage,
        )

    sim_inputs = _sim_inputs(mu)
    sim = simulate(**sim_inputs)
    # The rating is judged on the distribution WITHOUT the news tilt: the tilt is not calibrated and
    # must not be able to flip a label. The displayed tables keep it (it only moves the first days).
    sim_for_rating, rating_sim_inputs = sim, None
    if sentiment is not None and sentiment.drift_adjustment != 0.0:
        rating_sim_inputs = _sim_inputs(np.full(T, expected.final))
        sim_for_rating = simulate(**rating_sim_inputs)

    sigma_h = vol.total_vol(cfg.rating_horizon)
    sigma_1y = vol.one_year_vol if vol.one_year_vol is not None else vol.total_vol(252)
    context = {
        "drift": f"S&P 500 expected total return {expected.final:+.1%}/yr "
                 f"(E/P {fund.earnings_yield:.1%}, div yield {fund.dividend_yield:.1%}, "
                 f"valuation and regime adj {sum(expected.adjustments.values()):+.1%})",
        "vol": f"Implied vol {sigma_h:.0%} to the rating horizon -> leverage decay about "
               f"{etf.theoretical_drag(sigma_h):.0%}/yr; fund costs {etf.annual_cost:.1%}/yr; "
               f"the index needs about {etf.breakeven_index_return(sigma_1y):+.1%}/yr for SPXL to break even over the long run",
    }
    if sentiment is not None:
        context["sentiment"] = (f"News sentiment {sentiment.label.lower()} ({sentiment.score:+.2f}) "
                                f"-> {sentiment.drift_adjustment:+.1%} annualised drift for {cfg.sentiment_days} days "
                                f"in the price tables; excluded from this rating")
    if macro.curve_10y_3m is None:
        context["macro"] = "Yield curve unavailable (short rate defaulted)"
    elif macro.curve_10y_3m < 0:
        context["macro"] = f"Yield curve inverted ({macro.curve_10y_3m:+.2%})"
    else:
        context["macro"] = f"Yield curve 10y-3m {macro.curve_10y_3m:+.2%}; 3m bill {macro.rf_3m:.2%} sets the financing cost"

    rating = rate(sim_for_rating, cfg.rating_horizon, cfg, context)
    sens = influencer_sensitivities(snap, cfg)
    fc = Forecast(cfg=cfg, snap=snap, spot=spot, spot_date=spot_date, spot_status=spot_status, macro=macro,
                  fundamentals=fund, expected=expected, vol=vol, etf=etf, sentiment=sentiment, sim=sim,
                  rating=rating, sensitivities=sens, mu_path=mu, sim_inputs=sim_inputs,
                  rating_sim_inputs=rating_sim_inputs)
    fc.assessment = assess(fc)
    return fc


# ---------------------------------------------------------------------------------------
def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def forecast_to_dict(fc: Forecast, prices: Optional[List[float]] = None) -> Dict[str, Any]:
    sim = fc.sim
    out: Dict[str, Any] = {
        "run_at": fc.snap.asof.isoformat(),
        "model_version": MODEL_VERSION,
        "build": build_id(),
        "data_quality": fc.data_quality(),
        "prices_fetched_at": fc.snap.fetched_at.get("prices", fc.snap.asof).isoformat(),
        "etf": fc.cfg.etf,
        "spot": fc.spot,
        "spot_date": fc.spot_date,
        "spot_status": fc.spot_status,
        "assessment": assessment_to_dict(fc.assessment) if fc.assessment is not None else None,
        # the retired verdict, kept for continuity and research; not shown as a recommendation
        "rating": {
            "label": fc.rating.label, "conviction": fc.rating.conviction, "score": fc.rating.score,
            "score_se": fc.rating.score_se, "borderline": fc.rating.borderline,
            "horizon_days": fc.rating.horizon, "edge_annual": fc.rating.edge_annual,
            "sharpe_annual": fc.rating.sharpe_annual, "p_beat_rf": fc.rating.p_beat_rf,
            "p_positive": fc.rating.p_positive, "median_return": fc.rating.median_return,
            "mean_return": fc.rating.mean_return, "reasons": fc.rating.reasons,
        },
        "forecast": {str(h): {**sim.summary(h), "quantile_prices": sim.quantiles(h),
                              "path_min_quantile_prices": sim.path_min_quantiles(h),
                              # fine grids so the live layer can re-price any level without re-simulating
                              "grid": {"percentiles": list(GRID_PERCENTILES),
                                       "terminal": np.percentile(sim.terminal[h], GRID_PERCENTILES).tolist(),
                                       "path_min": np.percentile(sim.path_min[h], GRID_PERCENTILES).tolist(),
                                       "path_max": np.percentile(sim.path_max[h], GRID_PERCENTILES).tolist()}}
                     for h in sim.horizons},
        "drivers": {
            "index_expected_return": {
                "earnings_yield_model": fc.expected.earnings_yield_model,
                "dividend_growth_model": fc.expected.dividend_growth_model,
                "base": fc.expected.base, "adjustments": fc.expected.adjustments,
                "final": fc.expected.final, "notes": fc.expected.notes,
            },
            "fundamentals": {k: v for k, v in fc.fundamentals.__dict__.items()},
            "macro": {k: v for k, v in fc.macro.__dict__.items()},
            "vol": {"pillars": fc.vol.pillars, "notes": fc.vol.notes,
                    "total_vol_by_horizon": {str(h): fc.vol.total_vol(h) for h in sim.horizons}},
            "etf": {"leverage": fc.etf.leverage, "expense_ratio": fc.etf.expense_ratio,
                    "financing_rate": fc.etf.financing_rate, "annual_cost": fc.etf.annual_cost,
                    "tracking_sd_daily": fc.etf.tracking_sd_daily, "notes": fc.etf.notes,
                    "calibration": fc.etf.calibration.__dict__ if fc.etf.calibration else None},
        },
        "sentiment": None,
        "sensitivities": [s.__dict__ for s in fc.sensitivities],
        "notes": fc.snap.notes,
        "config": fc.cfg.to_dict(),
    }
    if fc.sentiment is not None:
        out["sentiment"] = {
            "score": fc.sentiment.score, "label": fc.sentiment.label, "n_articles": fc.sentiment.n_articles,
            "n_used": fc.sentiment.n_used, "by_ticker": fc.sentiment.by_ticker,
            "drift_adjustment": fc.sentiment.drift_adjustment,
            "top_positive": [{"title": x.item.title, "score": x.score, "ticker": x.item.ticker,
                              "published": x.item.published.isoformat()} for x in fc.sentiment.top_positive],
            "top_negative": [{"title": x.item.title, "score": x.score, "ticker": x.item.ticker,
                              "published": x.item.published.isoformat()} for x in fc.sentiment.top_negative],
        }
    if prices:
        out["price_lookup"] = {str(p): fc.price_lookup(p) for p in prices}
    out["limit_ladder"] = {str(h): fc.limit_ladder(h) for h in sim.horizons}
    return _clean(out)
