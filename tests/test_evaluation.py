import math

import numpy as np
import pytest

from spxlcast.evaluation import (crps_normal, crps_quantiles, crps_sample, effective_n, mean_interval,
                                 naive_leveraged_lognormal, normal_quantiles, pit_from_quantiles, proportion_interval,
                                 skill, skill_interval, spearman)
from spxlcast.live import GRID_PERCENTILES
from spxlcast.montecarlo import simulate

COARSE = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def test_crps_forms_agree_for_a_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.1, 200_000)
    for y in (-0.15, 0.0, 0.02, 0.3):
        exact = crps_normal(0.02, 0.1, y)
        assert crps_sample(x, y) == pytest.approx(exact, rel=0.01)
        fine = crps_quantiles(GRID_PERCENTILES, normal_quantiles(0.02, 0.1, GRID_PERCENTILES), y)
        coarse = crps_quantiles(COARSE, normal_quantiles(0.02, 0.1, COARSE), y)
        assert fine == pytest.approx(exact, rel=0.03)
        assert abs(fine - exact) < abs(coarse - exact)      # the archived grid is the better yardstick


def test_crps_is_proper_the_true_distribution_scores_best():
    rng = np.random.default_rng(1)
    ys = rng.normal(0.0, 1.0, 5000)
    true = np.mean([crps_normal(0.0, 1.0, y) for y in ys])
    for mu, sd in ((0.5, 1.0), (0.0, 2.0), (0.0, 0.5)):
        assert true < np.mean([crps_normal(mu, sd, y) for y in ys])
    assert skill([1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.5)


def test_pit_from_quantiles():
    lv, qv = [5, 50, 95], [-1.0, 0.0, 1.0]
    assert pit_from_quantiles(lv, qv, 0.0) == pytest.approx(0.5)
    assert pit_from_quantiles(lv, qv, -2.0) == pytest.approx(0.025)    # halfway into the uncovered tail
    assert pit_from_quantiles(lv, qv, 5.0) == pytest.approx(0.975)
    assert pit_from_quantiles(lv, qv, 0.5) == pytest.approx(0.725)


def test_effective_n_counts_non_overlapping_windows():
    daily = np.arange(252)
    assert effective_n(daily, 21) == pytest.approx((251 + 21) / 21)     # a year of daily 1-month forecasts: ~13
    monthly = np.arange(12) * 21
    assert effective_n(monthly, 5) == 12                                # no overlap: every forecast counts
    assert effective_n(monthly, 126) == pytest.approx((231 + 126) / 126)
    assert effective_n([], 21) == 0.0
    # an outage adds no outcomes: 60 daily rows, a year's gap, 5 more rows
    assert effective_n(np.r_[0:60, 312:317], 21) == pytest.approx((59 + 21 + 4 + 21) / 21)
    assert effective_n(np.r_[0:12, 32:38], 5) == pytest.approx(5.2)
    assert effective_n([3, 3, 3], 5) == pytest.approx(1.0)                # one session counts once


def test_closed_form_intervals():
    from scipy.stats import t
    x = np.array([0.2, 0.4, 0.6, 0.8])
    lo, hi = mean_interval(x, 4)
    assert (lo + hi) / 2 == pytest.approx(0.5) and hi - lo == pytest.approx(2 * t.ppf(0.95, 3) * x.std(ddof=1) / 2)
    lo4, hi4 = mean_interval(np.tile(x, 4), 4)             # more overlapping rows, same independent outcomes
    assert hi4 - lo4 > 0.8 * (hi - lo)
    assert all(math.isnan(v) for v in mean_interval(x, 1.9))
    lo, hi = proportion_interval(1.0, 4)                    # every row in the band: still an interval
    assert 0.4 < lo < 0.8 and hi == 1.0
    lo, hi = proportion_interval(0.9, 50)
    assert lo < 0.9 < hi and hi - lo < 0.2
    assert all(math.isnan(v) for v in proportion_interval(float("nan"), 5))
    m, b = np.array([1.0, 2.0, 3.0, 2.0]), np.array([2.0, 2.5, 3.5, 2.0])
    lo, hi = skill_interval(m, b, 4)
    assert lo < skill(m, b) < hi
    assert all(math.isnan(v) for v in skill_interval(m, np.zeros(4), 4))


def _crps_normal_v(sd, y):
    from scipy.stats import norm
    z = np.asarray(y) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def test_intervals_cover_the_truth_for_overlapping_windows():
    """Perfectly calibrated daily 1-month forecasts (true mean PIT 0.5, true 5-95 coverage 90%, a
    known skill against a too-wide and against a shifted benchmark): the 90% intervals must cover
    close to 90% of the time, also early on with few independent outcomes (the old block bootstrap
    managed 43-54%, a delta-method skill interval 75% against the shifted benchmark), and must not
    be far wider than needed."""
    from scipy.stats import norm
    rng = np.random.default_rng(11)
    h, reps = 21, 400
    big = rng.normal(0.0, 1.0, 1_000_000)
    wide, shifted = (lambda y: _crps_normal_v(1.3, y)), (lambda y: _crps_normal_v(1.0, y - 0.3))
    true_skill = [1.0 - _crps_normal_v(1.0, big).mean() / bench(big).mean() for bench in (wide, shifted)]
    for rows in (63, 252):                                  # about 4 and 13 independent outcomes
        hits = np.zeros(4)
        for _ in range(reps):
            walk = np.r_[0.0, np.cumsum(rng.normal(0.0, 1.0, rows + h))]
            y = (walk[h:h + rows] - walk[:rows]) / math.sqrt(h)
            ne = effective_n(np.arange(rows), h)
            lo, hi = mean_interval(norm.cdf(y), ne)
            hits[0] += lo <= 0.5 <= hi
            lo, hi = proportion_interval(float(np.mean(np.abs(y) <= norm.ppf(0.95))), ne)
            hits[1] += lo <= 0.9 <= hi
            for j, bench in enumerate((wide, shifted)):
                lo, hi = skill_interval(_crps_normal_v(1.0, y), bench(y), ne)
                hits[2 + j] += lo <= true_skill[j] <= hi
        cover = hits / reps
        assert np.all(cover >= 0.82), (rows, cover)
        assert cover[0] <= 0.96 and cover[1] <= 0.995 and np.all(cover[2:] <= 0.97), (rows, cover)


def test_spearman():
    a = np.arange(10.0)
    assert spearman(a, a ** 3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_averages_tied_ranks_and_is_nan_for_a_constant():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(5)
    x = np.clip(rng.normal(0.5, 0.5, 200), -1.0, 1.0)       # clipped like the rating score: many ties at +/-1
    y = x + rng.normal(0.0, 1.0, 200)
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
    p = rng.permutation(200)
    assert spearman(x[p], y[p]) == pytest.approx(spearman(x, y))       # row order does not matter
    assert math.isnan(spearman(np.ones(5), np.arange(5.0)))


def test_naive_lognormal_matches_the_engine_with_normal_shocks():
    """The closed form is the engine's own leveraged-fund dynamics with the frills switched off."""
    vix, rf, cost, h = 20.0, 0.04, 0.0091 + 2 * (0.04 + 0.0075), 63
    sim = simulate(spot=1.0, mu_annual=np.full(h, rf), sigma_annual=np.full(h, vix / 100), leverage=3.0,
                   daily_cost=cost / 252, tracking_sd_daily=0.0, rf_annual=rf, horizons=[h], n_paths=40000,
                   dof=500.0, skew_gamma=1.0, seed=3)
    x = np.log(sim.terminal[h])
    m, s = naive_leveraged_lognormal(vix, rf, cost, h)
    assert x.mean() == pytest.approx(m, abs=0.01)
    assert x.std() == pytest.approx(s, rel=0.03)
