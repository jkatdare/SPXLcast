import numpy as np

from spxlcast.config import Config
from spxlcast.montecarlo import simulate
from spxlcast.rating import annualize, horizon_phrase, rate


def _sim(mu, sigma, seed=0):
    T = 126
    return simulate(spot=200.0, mu_annual=np.full(T, mu), sigma_annual=np.full(T, sigma), leverage=3.0,
                    daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[63, 126],
                    n_paths=8000, seed=seed)


def test_annualize():
    assert abs(annualize(0.1, 126) - (1.1 ** 2 - 1)) < 1e-12
    assert annualize(-1.5, 126) == -1.0


def test_horizon_phrase():
    assert horizon_phrase(126) == "6-month"
    assert horizon_phrase(252) == "1-year"
    assert horizon_phrase(10) == "10-day"


def test_strong_drift_low_vol_is_buy():
    r = rate(_sim(0.15, 0.12), 126, Config())
    assert r.label == "BUY"
    assert r.edge_annual > 0 and r.sharpe_annual > 0 and r.p_beat_rf > 0.5
    assert r.conviction == "High"
    assert r.score_se >= 0 and not r.borderline   # every block saturates at +1, so the SE is zero
    assert any("T-bill" in x for x in r.reasons)


def test_negative_drift_high_vol_is_sell():
    r = rate(_sim(-0.05, 0.40), 126, Config())
    assert r.label == "SELL"
    assert r.edge_annual < 0 and r.sharpe_annual < 0


def test_middling_case_is_hold():
    # 3 ln(1.07) - 10% cost - 4.5 x 0.16^2 ~ -1%/yr median (below the 4% bill) while the mean excess
    # return is modestly positive: the two views disagree and the score lands in the HOLD band
    r = rate(_sim(0.07, 0.16), 126, Config())
    assert r.label == "HOLD"
    assert -0.3 < r.score < 0.3
    assert r.edge_annual < 0 < r.sharpe_annual
    assert 0 < r.score_se < 0.1


def test_buy_requires_positive_median_edge():
    cfg = Config(sharpe_scale=0.05)   # make the expected-value component saturate easily
    r = rate(_sim(0.07, 0.16), 126, cfg)
    assert r.edge_annual < 0
    assert r.label != "BUY"


def test_borderline_flags_the_buy_gate_when_the_edge_is_within_noise():
    cfg = Config(sharpe_scale=0.10)
    r = rate(_sim(0.084, 0.16), 126, cfg)
    assert r.score >= cfg.buy_score           # the score alone would say BUY
    assert abs(r.edge_annual) < 0.03          # but the median edge is a coin flip
    assert r.borderline is True
    assert any("Borderline" in x for x in r.reasons)


def test_non_finite_returns_are_rejected():
    import pytest
    sim = _sim(0.10, 0.15)
    sim.terminal[126] = sim.terminal[126] * np.nan
    with pytest.raises(ValueError):
        rate(sim, 126, Config())


def test_context_lines_are_appended():
    r = rate(_sim(0.10, 0.15), 63, Config(), context={"drift": "drift line", "sentiment": "news line"})
    assert "drift line" in r.reasons and "news line" in r.reasons
    assert r.horizon == 63
    assert isinstance(r.borderline, bool)
