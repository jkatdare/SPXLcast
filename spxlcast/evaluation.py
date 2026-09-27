"""Forecast-evaluation tools shared by the live scorer (tracklog) and the backtests.

* Proper scoring: CRPS (continuous ranked probability score) of a whole predictive distribution
  against the outcome, from simulated samples, from a quantile grid, or in closed form for a normal.
  Lower is better; it rewards both sharpness and calibration, so two models can be compared with
  one number. Skill = 1 - CRPS(model) / CRPS(baseline) (positive = the model beats the baseline).
* PIT (where the outcome fell in the predicted distribution) from a quantile grid.
* Overlap-aware uncertainty: forecasts made a day (or a month) apart share most of their outcome
  window, so their errors are not independent. ``effective_n`` counts how many non-overlapping
  windows the sample really holds, and ``mean_interval`` / ``proportion_interval`` /
  ``skill_interval`` give closed-form intervals on that many observations (t intervals, and a
  Wilson interval for a hit rate). In simulations of perfectly calibrated daily forecasts the
  nominal 90% intervals for mean PIT and skill cover about 82-90% at three to seven independent
  outcomes and 88-93% from about a dozen on; the hit-rate interval is conservative (94-98%), because
  hits of overlapping windows are less correlated than the windows. A moving-block bootstrap with a block
  of one window, the obvious alternative, keeps only about 2/3 of the variance of an
  overlapping-window mean and degenerates with few blocks.

Everything works in log-return space so scores are comparable across price levels and dates.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

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
    """How many non-overlapping ``horizon``-long outcome windows the forecasts amount to: the
    length of the union of their windows over ``horizon`` (``positions`` are the trading-session
    indices of the forecast dates). Equals the count when the windows do not overlap, and a gap in
    the log adds nothing."""
    pos = np.sort(np.asarray(positions, dtype=float))
    if len(pos) == 0 or horizon <= 0:
        return 0.0
    covered = float(np.minimum(np.diff(pos), horizon).sum()) + horizon
    return covered / horizon


def _t_quantile(alpha: float, n_eff: float) -> float:
    from scipy.stats import t
    return float(t.ppf(1.0 - alpha / 2.0, max(n_eff - 1.0, 1.0)))


def mean_interval(values: Sequence[float], n_eff: float, alpha: float = 0.10) -> Tuple[float, float]:
    """(1 - alpha) t interval for the mean of overlapping-window values: the row standard deviation
    over sqrt(n_eff), with n_eff - 1 degrees of freedom. (nan, nan) below two independent outcomes."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not n_eff >= 2 or len(x) < 2:
        return float("nan"), float("nan")
    half = _t_quantile(alpha, n_eff) * float(x.std(ddof=1)) / math.sqrt(n_eff)
    return float(x.mean() - half), float(x.mean() + half)


def proportion_interval(p_hat: float, n_eff: float, alpha: float = 0.10) -> Tuple[float, float]:
    """(1 - alpha) Wilson interval for a hit rate observed over n_eff independent trials; never
    zero-width, even at 0% or 100%."""
    if not (n_eff > 0 and np.isfinite(p_hat)):
        return float("nan"), float("nan")
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p, n = min(max(float(p_hat), 0.0), 1.0), float(n_eff)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def skill_interval(model_scores: Sequence[float], baseline_scores: Sequence[float], n_eff: float,
                   alpha: float = 0.10) -> Tuple[float, float]:
    """(1 - alpha) interval for skill = 1 - mean(model) / mean(baseline) = mean(baseline - model) /
    mean(baseline) on paired rows: a t interval on n_eff for the mean difference, as in
    ``mean_interval``, over mean(baseline). (A delta-method interval on the ratio is narrowest when
    its estimate is furthest off, and covered only 75% at three independent outcomes.)"""
    m = np.asarray(model_scores, dtype=float)
    b = np.asarray(baseline_scores, dtype=float)
    ok = np.isfinite(m) & np.isfinite(b)
    m, b = m[ok], b[ok]
    if not n_eff >= 2 or len(m) < 2 or not b.mean() > 0:
        return float("nan"), float("nan")
    d = b - m
    half = _t_quantile(alpha, n_eff) * float(d.std(ddof=1)) / math.sqrt(n_eff)
    return float((d.mean() - half) / b.mean()), float((d.mean() + half) / b.mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties, nan for a constant input)."""
    from scipy.stats import rankdata
    ra = rankdata(a)
    rb = rankdata(b)
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
