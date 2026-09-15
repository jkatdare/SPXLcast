import numpy as np

from spxlcast.config import Config
from spxlcast.montecarlo import simulate
from spxlcast.rating import annualize, rate


def _sim(mu, sigma, seed=0):
    T = 126
    return simulate(spot=200.0, mu_annual=np.full(T, mu), sigma_annual=np.full(T, sigma), leverage=3.0,
                    daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[63, 126],
                    n_paths=8000, seed=seed)


def test_annualize():
    assert abs(annualize(0.1, 126) - (1.1 ** 2 - 1)) < 1e-12
    assert annualize(-1.5, 126) == -1.0


def test_strong_drift_low_vol_is_buy():
    r = rate(_sim(0.15, 0.12), 126, Config())
    assert r.label == "BUY"
    assert r.edge_annual > 0 and r.p_beat_rf > 0.5
    assert r.conviction in {"High", "Medium", "Low"}
    assert any("T-bill" in x for x in r.reasons)


def test_negative_drift_high_vol_is_sell():
    r = rate(_sim(-0.05, 0.40), 126, Config())
    assert r.label == "SELL"
    assert r.edge_annual < 0


def test_middling_case_is_hold():
    # 3 x ln(1.10) - 10% cost - 0.5 x 9 x 0.16^2 decay ~ +7%/yr log return: a few points over the
    # 4% T-bill, i.e. a modest edge and P(beat) near 50% -> neither BUY nor SELL
    r = rate(_sim(0.10, 0.16), 126, Config())
    assert r.label == "HOLD"
    assert -0.3 < r.score < 0.3


def test_context_lines_are_appended():
    r = rate(_sim(0.10, 0.15), 63, Config(), context={"drift": "drift line", "sentiment": "news line"})
    assert "drift line" in r.reasons and "news line" in r.reasons
    assert r.horizon == 63
