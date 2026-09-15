"""End-to-end pipeline: data -> drivers -> simulation -> rating, plus a JSON-able summary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Config
from .data import MarketSnapshot, load_market
from .etf import ETFParams, build_etf_params
from .fundamentals import (ExpectedReturn, IndexFundamentals, MacroState, Sensitivity, VolTermStructure,
                           build_fundamentals, build_macro, expected_index_return,
                           influencer_sensitivities, vol_term_structure)
from .montecarlo import QUANTILES, SimulationResult, simulate
from .rating import Rating, rate
from .sentiment import SentimentResult, analyze_news


@dataclass
class Forecast:
    cfg: Config
    snap: MarketSnapshot
    spot: float
    spot_date: str
    macro: MacroState
    fundamentals: IndexFundamentals
    expected: ExpectedReturn
    vol: VolTermStructure
    etf: ETFParams
    sentiment: Optional[SentimentResult]
    sim: SimulationResult
    rating: Rating
    sensitivities: List[Sensitivity] = field(default_factory=list)
    mu_path: Optional[np.ndarray] = None

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
        """Buy-limit prices with the given probability of being touched before ``horizon``."""
        mins = self.sim.path_min[horizon]
        ends = self.sim.terminal[horizon]
        rows = []
        for p in probs:
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

    macro = build_macro(snap, cfg)
    fund = build_fundamentals(snap, cfg)
    expected = expected_index_return(fund, macro, cfg)

    horizons = sorted(set(int(h) for h in cfg.horizons) | {int(cfg.rating_horizon)})
    T = max(horizons)
    vol = vol_term_structure(macro, snap, cfg, T)
    etf = build_etf_params(snap, cfg, macro.rf_3m)

    sentiment = None
    mu = np.full(T, expected.final)
    if cfg.use_news:
        weights = {}
        if snap.holdings is not None:
            weights = dict(zip(snap.holdings["symbol"], snap.holdings["weight"]))
        sentiment = analyze_news(snap.news, cfg, weights, now=snap.asof)
        n = min(cfg.sentiment_days, T)
        mu[:n] += sentiment.drift_adjustment

    sim = simulate(
        spot=spot, mu_annual=mu, sigma_annual=vol.daily, leverage=etf.leverage,
        daily_cost=etf.daily_cost, tracking_sd_daily=etf.tracking_sd_daily, rf_annual=macro.rf_3m,
        horizons=horizons, n_paths=cfg.n_paths, dof=cfg.t_dof, max_daily_move=cfg.max_daily_move,
        seed=cfg.seed,
    )

    sigma_h = vol.total_vol(cfg.rating_horizon)
    context = {
        "drift": f"S&P 500 expected total return {expected.final:+.1%}/yr "
                 f"(E/P {fund.earnings_yield:.1%}, div yield {fund.dividend_yield:.1%}, "
                 f"macro adj {sum(expected.adjustments.values()):+.1%})",
        "vol": f"Implied vol {sigma_h:.0%} to the rating horizon -> leverage decay about "
               f"{etf.theoretical_drag(sigma_h):.0%}/yr; fund costs {etf.annual_cost:.1%}/yr; "
               f"index needs about {etf.breakeven_index_return(sigma_h):+.1%}/yr for SPXL to break even",
    }
    if sentiment is not None:
        context["sentiment"] = (f"News sentiment {sentiment.label.lower()} ({sentiment.score:+.2f}) "
                                f"-> {sentiment.drift_adjustment:+.1%} annualised drift for {cfg.sentiment_days} days")
    if macro.curve_10y_3m < 0:
        context["macro"] = f"Yield curve inverted ({macro.curve_10y_3m:+.2%})"
    else:
        context["macro"] = (f"Yield curve 10y-3m {macro.curve_10y_3m:+.2%}; real 10y "
                            f"{macro.real_10y:+.2%}" if macro.real_10y is not None else "")

    rating = rate(sim, cfg.rating_horizon, cfg, context)
    sens = influencer_sensitivities(snap, cfg)
    return Forecast(cfg=cfg, snap=snap, spot=spot, spot_date=spot_date, macro=macro, fundamentals=fund,
                    expected=expected, vol=vol, etf=etf, sentiment=sentiment, sim=sim, rating=rating,
                    sensitivities=sens, mu_path=mu)


# ---------------------------------------------------------------------------------------
def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def forecast_to_dict(fc: Forecast, prices: Optional[List[float]] = None) -> Dict[str, Any]:
    sim = fc.sim
    out: Dict[str, Any] = {
        "asof": fc.snap.asof.isoformat(),
        "etf": fc.cfg.etf,
        "spot": fc.spot,
        "spot_date": fc.spot_date,
        "rating": {
            "label": fc.rating.label, "conviction": fc.rating.conviction, "score": fc.rating.score,
            "horizon_days": fc.rating.horizon, "edge_annual": fc.rating.edge_annual,
            "p_beat_rf": fc.rating.p_beat_rf, "p_positive": fc.rating.p_positive,
            "median_return": fc.rating.median_return, "mean_return": fc.rating.mean_return,
            "reasons": fc.rating.reasons,
        },
        "forecast": {str(h): {**sim.summary(h), "quantile_prices": sim.quantiles(h),
                              "path_min_quantile_prices": sim.path_min_quantiles(h)} for h in sim.horizons},
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
