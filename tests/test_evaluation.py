import math

import numpy as np
import pytest

from spxlcast.evaluation import (block_bootstrap, crps_normal, crps_quantiles, crps_sample, effective_n,
                                 naive_leveraged_lognormal, normal_quantiles, pit_from_quantiles, skill, spearman)
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


def test_block_bootstrap_widens_for_correlated_data_and_needs_two_blocks():
    rng = np.random.default_rng(2)
    iid = rng.normal(0.0, 1.0, 400)
    lo, hi = block_bootstrap(iid, 1)
    assert lo < 0.0 < hi and (hi - lo) == pytest.approx(2 * 1.645 / math.sqrt(400), rel=0.25)
    ar = np.zeros(400)
    for i in range(1, 400):                    # strongly autocorrelated, like overlapping windows
        ar[i] = 0.95 * ar[i - 1] + rng.normal()
    naive_width = np.subtract(*block_bootstrap(ar, 1)[::-1])
    block_width = np.subtract(*block_bootstrap(ar, 40)[::-1])
    assert block_width > 2 * naive_width
    assert all(math.isnan(v) for v in block_bootstrap(iid[:30], 21))  # fewer than two blocks: no interval
    pair = np.column_stack([iid, iid + 1.0])
    lo, hi = block_bootstrap(pair, 5, lambda a: (a[:, 1] - a[:, 0]).mean())
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_spearman():
    a = np.arange(10.0)
    assert spearman(a, a ** 3) == pytest.approx(1.0)
    assert spearman(np.column_stack([a, -a])) == pytest.approx(-1.0)


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
