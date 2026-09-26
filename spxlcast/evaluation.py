"""Forecast-evaluation tools shared by the live scorer (tracklog) and the backtests.

* Proper scoring: CRPS (continuous ranked probability score) of a whole predictive distribution
  against the outcome, from simulated samples, from a quantile grid, or in closed form for a normal.
  Lower is better; it rewards both sharpness and calibration, so two models can be compared with
  one number. Skill = 1 - CRPS(model) / CRPS(baseline) (positive = the model beats the baseline).
* PIT (where the outcome fell in the predicted distribution) from a quantile grid.
* Overlap-aware uncertainty: forecasts made a day (or a month) apart share most of their outcome
  window, so their errors are not independent. ``effective_n`` counts how many non-overlapping
  windows the sample really holds, and ``block_bootstrap`` gives confidence intervals that respect
  the overlap by resampling whole blocks of consecutive forecasts.

Everything works in log-return space so scores are comparable across price levels and dates.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# ---------------------------------------------------------------------------------------
# Proper scores
# ---------------------------------------------------------------------------------------
def crps_sample(samples: np.ndarray, y: float) -> float:
    """CRPS of an ensemble for outcome ``y``: E|X - y| - 0.5 E|X - X'| (exact for the ensemble)."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    mean_abs_pair = 2.0 * float(np.sum((2 * i - n - 1) * x)) / (n * n)     # E|X - X'| with replacement
    return float(np.mean(np.abs(x - y)) - 0.5 * mean_abs_pair)


def crps_normal(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of Normal(mu, sigma^2) for outcome ``y``."""
    if not sigma > 0:
        return abs(y - mu)
    z = (y - mu) / sigma
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI))


def crps_quantiles(levels_pct: Sequence[float], qvals: Sequence[float], y: float) -> float:
    """CRPS from a quantile grid: twice the pinball loss integrated over the quantile levels
    (trapezoid). The tails beyond the grid are left out, so a coarse grid understates CRPS; only
    compare scores computed on the same grid."""
    tau = np.asarray(levels_pct, dtype=float) / 100.0
    q = np.asarray(qvals, dtype=float)
    loss = ((y < q).astype(float) - tau) * (q - y)
    return float(2.0 * np.sum(0.5 * (loss[1:] + loss[:-1]) * np.diff(tau)))


def normal_quantiles(mu: float, sigma: float, levels_pct: Sequence[float]) -> np.ndarray:
    """Quantiles of Normal(mu, sigma^2) at the given percent levels (for same-grid comparisons)."""
    from scipy.stats import norm
    return mu + sigma * norm.ppf(np.asarray(levels_pct, dtype=float) / 100.0)


def pit_from_quantiles(levels_pct: Sequence[float], qvals: Sequence[float], y: float) -> float:
    """Where ``y`` falls in a distribution given by a quantile grid, piecewise-linear between the
    quantiles; beyond the grid it is placed halfway into the uncovered tail."""
    lv = np.asarray(levels_pct, dtype=float)
    qv = np.asarray(qvals, dtype=float)
    if y <= qv[0]:
        return float(lv[0] / 200.0)
    if y >= qv[-1]:
        return float(1.0 - (100.0 - lv[-1]) / 200.0)
    return float(np.interp(y, qv, lv) / 100.0)


def skill(model_scores: Sequence[float], baseline_scores: Sequence[float]) -> float:
    """1 - mean(model) / mean(baseline): positive when the model's CRPS is lower (better)."""
    b = float(np.mean(baseline_scores))
    return float("nan") if b <= 0 else 1.0 - float(np.mean(model_scores)) / b


# ---------------------------------------------------------------------------------------
# Overlap-aware uncertainty
# ---------------------------------------------------------------------------------------
def effective_n(positions: Sequence[int], horizon: int) -> float:
    """How many non-overlapping ``horizon``-long outcome windows fit in the span the forecasts
    cover (``positions`` are the trading-session indices of the forecast dates). Equals the count
    when the windows do not overlap; about span / horizon when they do."""
    pos = np.sort(np.asarray(positions, dtype=float))
    if len(pos) == 0 or horizon <= 0:
        return 0.0
    span = pos[-1] - pos[0] + horizon
    return float(min(len(pos), span / horizon))


def block_bootstrap(data: np.ndarray, block: int, stat: Callable[[np.ndarray], float] = np.mean,
                    n_boot: int = 2000, alpha: float = 0.10, seed: int = 0) -> Tuple[float, float]:
    """Moving-block bootstrap (1 - alpha) interval of ``stat`` for time-ordered rows whose
    neighbours are correlated. ``block`` is the number of consecutive rows per block (about the
    overlap length). ``data`` may be 1-D or 2-D (rows = time), so paired statistics work too.
    Returns (nan, nan) when there are fewer than two blocks' worth of rows."""
    x = np.asarray(data)
    n = len(x)
    b = int(max(1, block))
    if n < 2 * b or n < 3:
        return float("nan"), float("nan")
    k = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - b + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]
    vals = np.array([stat(x[i]) for i in idx], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def spearman(a: np.ndarray, b: Optional[np.ndarray] = None) -> float:
    """Rank correlation; ``a`` may be a 2-column array (for use as a bootstrap statistic)."""
    if b is None:
        a, b = a[:, 0], a[:, 1]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------------------
# A naive benchmark forecast
# ---------------------------------------------------------------------------------------
def naive_leveraged_lognormal(vix: float, rf: float, annual_cost: float, horizon: int,
                              leverage: float = 3.0) -> Tuple[float, float]:
    """(mean, sd) of the fund's log return over ``horizon`` sessions when the index follows a
    lognormal with the raw VIX as its vol and the T-bill as its drift (the market-implied,
    no-fundamentals view): d log L = (L rf - cost - L^2 sigma^2 / 2) dt + L sigma dW, where ``cost``
    is the fund's all-in annual cost (expense ratio plus financing of the borrowed notional)."""
    sigma = vix / 100.0
    tau = horizon / 252.0
    mean = (leverage * rf - annual_cost - 0.5 * leverage ** 2 * sigma ** 2) * tau
    return float(mean), float(leverage * sigma * math.sqrt(tau))
