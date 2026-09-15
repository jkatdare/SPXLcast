"""Monte Carlo engine: simulate daily S&P 500 total returns, compound them through the
leveraged-ETF mechanics, and record the terminal, path-minimum and path-maximum prices.

Daily index shocks are Student-t (fat tails) with a deterministic vol term structure and a drift
that may differ over the first weeks (news sentiment). Simple daily returns are simulated directly
so that the leveraged fund's compounding drag arises naturally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

QUANTILES = (5, 10, 25, 50, 75, 90, 95)


def standardized_t(rng: np.random.Generator, dof: float, size: int) -> np.ndarray:
    """Student-t draws rescaled to unit variance (requires dof > 2)."""
    z = rng.standard_t(dof, size=size)
    return z * np.sqrt((dof - 2.0) / dof)


def annual_to_daily_drift(mu_annual: np.ndarray | float) -> np.ndarray:
    """Daily simple-return mean whose compounding reproduces the annual arithmetic return."""
    return np.power(1.0 + np.asarray(mu_annual, dtype=float), 1.0 / 252.0) - 1.0


@dataclass
class SimulationResult:
    spot: float
    horizons: List[int]
    terminal: Dict[int, np.ndarray]        # ETF price at horizon
    path_min: Dict[int, np.ndarray]        # lowest ETF price on the path up to horizon
    path_max: Dict[int, np.ndarray]
    index_terminal: Dict[int, np.ndarray]  # index total-return level (1.0 = today)
    rf_growth: Dict[int, float]            # T-bill growth factor to horizon
    fan: np.ndarray                        # (T+1, len(QUANTILES)) ETF price quantiles by day
    fan_quantiles: Sequence[int] = QUANTILES
    n_paths: int = 0

    # ---- lookups --------------------------------------------------------------------
    def percentile_of_price(self, price: float, horizon: int) -> float:
        """Share of simulated horizon prices at or below ``price`` (0..100)."""
        return 100.0 * float(np.mean(self.terminal[horizon] <= price))

    def prob_touch_below(self, price: float, horizon: int) -> float:
        """Probability the ETF trades at or below ``price`` at some point before horizon
        (i.e. a buy-limit order at that price would fill)."""
        return float(np.mean(self.path_min[horizon] <= price))

    def prob_touch_above(self, price: float, horizon: int) -> float:
        return float(np.mean(self.path_max[horizon] >= price))

    def price_at_percentile(self, pct: float, horizon: int) -> float:
        return float(np.percentile(self.terminal[horizon], pct))

    def quantiles(self, horizon: int, qs: Iterable[float] = QUANTILES) -> Dict[float, float]:
        arr = self.terminal[horizon]
        return {float(q): float(np.percentile(arr, q)) for q in qs}

    def path_min_quantiles(self, horizon: int, qs: Iterable[float] = QUANTILES) -> Dict[float, float]:
        arr = self.path_min[horizon]
        return {float(q): float(np.percentile(arr, q)) for q in qs}

    def returns(self, horizon: int) -> np.ndarray:
        return self.terminal[horizon] / self.spot - 1.0

    def summary(self, horizon: int) -> Dict[str, float]:
        r = self.returns(horizon)
        rf = self.rf_growth[horizon] - 1.0
        idx = self.index_terminal[horizon] - 1.0
        q = self.quantiles(horizon)
        var5 = float(np.percentile(r, 5))
        return {
            "horizon": horizon,
            "mean_return": float(r.mean()),
            "median_return": float(np.median(r)),
            "std_return": float(r.std(ddof=1)),
            "p_positive": float(np.mean(r > 0)),
            "p_beat_rf": float(np.mean(r > rf)),
            "p_beat_index": float(np.mean(r > idx)),
            "rf_return": rf,
            "index_mean_return": float(idx.mean()),
            "index_median_return": float(np.median(idx)),
            "var_5": var5,
            "es_5": float(r[r <= var5].mean()) if np.any(r <= var5) else var5,
            "p_drawdown_20": float(np.mean(self.path_min[horizon] <= 0.8 * self.spot)),
            "p_drawdown_30": float(np.mean(self.path_min[horizon] <= 0.7 * self.spot)),
            "p_up_20": float(np.mean(self.path_max[horizon] >= 1.2 * self.spot)),
            **{f"q{int(k)}": v for k, v in q.items()},
        }


def simulate(
    spot: float,
    mu_annual: np.ndarray,          # (T,) annualised index total-return drift per day
    sigma_annual: np.ndarray,       # (T,) annualised index vol per day
    leverage: float,
    daily_cost: float,
    tracking_sd_daily: float,
    rf_annual: float,
    horizons: Sequence[int],
    n_paths: int = 20_000,
    dof: float = 4.0,
    max_daily_move: float = 0.20,
    seed: Optional[int] = 42,
) -> SimulationResult:
    horizons = sorted(int(h) for h in horizons)
    T = max(horizons)
    mu_annual = np.broadcast_to(np.asarray(mu_annual, dtype=float), (T,))
    sigma_annual = np.broadcast_to(np.asarray(sigma_annual, dtype=float), (T,))
    mu_d = annual_to_daily_drift(mu_annual)
    sig_d = sigma_annual / np.sqrt(252.0)
    rf_d = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

    rng = np.random.default_rng(seed)
    S = np.full(n_paths, float(spot))
    I = np.ones(n_paths)
    run_min = S.copy()
    run_max = S.copy()
    fan = np.empty((T + 1, len(QUANTILES)))
    fan[0] = spot

    res = SimulationResult(spot=float(spot), horizons=horizons, terminal={}, path_min={}, path_max={},
                           index_terminal={}, rf_growth={}, fan=fan, n_paths=n_paths)
    hset = set(horizons)
    for t in range(T):
        z = standardized_t(rng, dof, n_paths)
        r = mu_d[t] + sig_d[t] * z
        np.clip(r, -max_daily_move, max_daily_move, out=r)
        r_etf = leverage * r - daily_cost
        if tracking_sd_daily > 0:
            r_etf += rng.normal(0.0, tracking_sd_daily, n_paths)
        np.maximum(r_etf, -0.99, out=r_etf)
        S = S * (1.0 + r_etf)
        I = I * (1.0 + r)
        np.minimum(run_min, S, out=run_min)
        np.maximum(run_max, S, out=run_max)
        fan[t + 1] = np.percentile(S, QUANTILES)
        day = t + 1
        if day in hset:
            res.terminal[day] = S.copy()
            res.path_min[day] = run_min.copy()
            res.path_max[day] = run_max.copy()
            res.index_terminal[day] = I.copy()
            res.rf_growth[day] = (1.0 + rf_d) ** day
    return res
