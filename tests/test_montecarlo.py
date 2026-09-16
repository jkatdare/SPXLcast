import numpy as np
import pytest
from scipy import stats

from spxlcast.montecarlo import annual_to_daily_drift, simulate, skewed_standardized_t, standardized_t


def test_standardized_t_has_unit_variance():
    rng = np.random.default_rng(0)
    z = standardized_t(rng, 5.0, 400_000)
    assert abs(z.std() - 1.0) < 0.02
    assert abs(z.mean()) < 0.01


def test_skewed_t_is_standardised_and_negatively_skewed():
    rng = np.random.default_rng(1)
    z = skewed_standardized_t(rng, 6.0, 0.85, 400_000)
    assert abs(z.mean()) < 0.01
    assert abs(z.std() - 1.0) < 0.02
    assert stats.skew(z) < -0.2
    sym = skewed_standardized_t(np.random.default_rng(1), 6.0, 1.0, 100_000)
    assert abs(stats.skew(sym)) < 0.1


def test_annual_to_daily_drift_compounds_back():
    mu_d = annual_to_daily_drift(0.08)
    assert abs((1 + mu_d) ** 252 - 1.08) < 1e-12


@pytest.fixture(scope="module")
def sim():
    T = 252
    return simulate(spot=100.0, mu_annual=np.full(T, 0.07), sigma_annual=np.full(T, 0.16), leverage=3.0,
                    daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[21, 63, 126, 252],
                    n_paths=30_000, dof=4.0, seed=1)


def test_mean_index_return_matches_drift(sim):
    # the unlevered index compounds to the arithmetic drift on average
    idx = sim.index_terminal[252]
    assert abs(idx.mean() - 1.07) < 0.01


def test_skew_does_not_change_the_mean():
    T = 252
    res = simulate(spot=100.0, mu_annual=np.full(T, 0.07), sigma_annual=np.full(T, 0.16), leverage=3.0,
                   daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[252],
                   n_paths=30_000, dof=4.0, skew_gamma=0.9, seed=2)
    assert abs(res.index_terminal[252].mean() - 1.07) < 0.01


def test_leveraged_median_shows_volatility_decay(sim):
    # log-median of a 3x fund ~ 3*ln(1.07) - cost - 0.5*9*sigma^2 (approx.)
    expected_log = 3 * np.log(1.07) - 0.10 - 0.5 * 9 * 0.16 ** 2
    observed_log = np.log(np.median(sim.terminal[252]) / 100.0)
    assert abs(observed_log - expected_log) < 0.04


def test_percentile_is_monotone_and_bounded(sim):
    p = [sim.percentile_of_price(x, 126) for x in (50, 80, 100, 120, 200)]
    assert p == sorted(p)
    assert 0 <= p[0] <= p[-1] <= 100
    assert sim.percentile_of_price(1e-9, 126) == 0.0
    assert sim.percentile_of_price(1e9, 126) == 100.0


def test_touch_probability_dominates_terminal_probability(sim):
    for h in sim.horizons:
        for price in (85, 95, 99):
            assert sim.prob_touch_below(price, h) >= sim.percentile_of_price(price, h) / 100.0
        for price in (101, 110, 130):
            assert sim.prob_touch_above(price, h) >= 1 - sim.percentile_of_price(price, h) / 100.0


def test_touch_probability_grows_with_horizon(sim):
    below = [sim.prob_touch_below(90, h) for h in sim.horizons]
    assert below == sorted(below)
    above = [sim.prob_touch_above(110, h) for h in sim.horizons]
    assert above == sorted(above)


def test_summary_and_quantiles_are_consistent(sim):
    s = sim.summary(63)
    q = sim.quantiles(63)
    assert q[5.0] <= q[50.0] <= q[95.0]
    assert 0 <= s["p_positive"] <= 1 and 0 <= s["p_beat_rf"] <= 1
    assert s["rf_return"] == pytest.approx((1.04) ** (63 / 252) - 1, rel=1e-9)
    assert s["es_5"] <= s["var_5"]


def test_daily_move_is_capped():
    T = 21
    res = simulate(spot=100.0, mu_annual=np.zeros(T), sigma_annual=np.full(T, 3.0), leverage=3.0,
                   daily_cost=0.0, tracking_sd_daily=0.0, rf_annual=0.0, horizons=[T], n_paths=5000,
                   dof=3.0, max_daily_move=0.2, seed=3)
    assert res.path_min[T].min() > 0.0
    assert np.all(np.isfinite(res.terminal[T]))


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        simulate(100.0, np.zeros(5), np.full(5, 0.2), 3.0, 0.0, 0.0, 0.0, horizons=[0, 5], n_paths=100)
    with pytest.raises(ValueError):
        simulate(100.0, np.zeros(5), np.full(5, 0.2), 3.0, 0.0, 0.0, 0.0, horizons=[5], n_paths=1)
    with pytest.raises(ValueError):
        simulate(100.0, np.zeros(5), np.full(5, 0.2), 3.0, 0.0, 0.0, 0.0, horizons=[5], n_paths=100, skew_gamma=0.0)
    with pytest.raises(ValueError):
        simulate(100.0, np.zeros(5), np.full(5, 0.2), 3.0, 0.0, 0.0, 0.0, horizons=[5], n_paths=100, dof=2.0)
