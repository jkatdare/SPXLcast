import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spxlcast import assess as A
from spxlcast.etf import ETFParams
from spxlcast.montecarlo import simulate

ROOT = Path(__file__).resolve().parents[1]

REF = {"source": "test", "period": "1990-01..2026-08", "n": 440,
       "hurdle": {"grid": [0.03 + 0.005 * i for i in range(21)], "cutoffs": [0.05, 0.07]},
       "p_dip20_3m": {"grid": [0.05 + 0.02 * i for i in range(21)], "cutoffs": [0.17, 0.32], "median": 0.24,
                      "by_level": {"low": {"pred": 0.12, "real": 0.09, "n": 147, "n_indep": 66.3},
                                   "normal": {"pred": 0.25, "real": 0.22, "n": 143, "n_indep": 81.7},
                                   "elevated": {"pred": 0.42, "real": 0.35, "n": 147, "n_indep": 63.8}}},
       "leverage_cost_check": {"pred_mean": 0.13, "real_mean": 0.14, "corr": 0.26, "n": 206}}


def _ref_file(tmp_path, ref=REF):
    p = tmp_path / "reference.json"
    p.write_text(json.dumps(ref), encoding="utf-8")
    return p


def _fc(horizons=(21, 63, 126), one_year_vol=0.19):
    sim = simulate(spot=200.0, mu_annual=np.full(126, 0.07), sigma_annual=np.full(126, 0.18), leverage=3.0,
                   daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=list(horizons),
                   n_paths=4000, seed=1)
    etf = ETFParams(leverage=3.0, expense_ratio=0.0091, financing_rate=2 * (0.04 + 0.0075), tracking_sd_daily=0.0,
                    calibration=None)
    return SimpleNamespace(etf=etf, vol=SimpleNamespace(one_year_vol=one_year_vol, total_vol=lambda h: 0.21),
                           expected=SimpleNamespace(final=0.061), sim=sim, spot=200.0)


# ---- levels and percentiles ----------------------------------------------------------------
def test_level_of_edges():
    c, labels = (0.05, 0.07), A.LEVERAGE_LEVELS
    assert A.level_of(0.049, c, labels) == "low"
    assert A.level_of(0.05, c, labels) == "normal"        # a cutoff belongs to the level above it
    assert A.level_of(0.069, c, labels) == "normal"
    assert A.level_of(0.07, c, labels) == "high"
    assert A.level_of(0.5, [0.05, 0.07], A.DRAWDOWN_LEVELS) == "elevated"
    assert A.level_of(0.06, (0.06, 0.06), labels) == "high"
    for bad in (None, float("nan"), float("inf"), "x"):
        assert A.level_of(bad, c, labels) == "n/a"
    for bad in (None, [], [0.05], [0.05, 0.07, 0.09], [0.07, 0.05], [0.05, float("nan")], ["a", "b"], "0.05,0.07"):
        assert A.level_of(0.06, bad, labels) == "n/a"


def test_percentile_of_edges():
    grid = [float(i) for i in range(21)]                  # value i sits at the 5i-th percentile
    assert A.percentile_of(10.0, grid) == pytest.approx(50.0)
    assert A.percentile_of(2.5, grid) == pytest.approx(12.5)
    assert A.percentile_of(0.0, grid) == 0.0 and A.percentile_of(20.0, grid) == 100.0
    assert A.percentile_of(-1.0, grid) == 0.0 and A.percentile_of(99.0, grid) == 100.0
    tied = [0.0] * 5 + [1.0 + i for i in range(16)]       # percentiles 0-20 share one value
    assert A.percentile_of(0.0, tied) == pytest.approx(10.0)
    assert A.percentile_of(0.5, tied) == pytest.approx(17.5)
    flat = [1.0] * 21
    assert (A.percentile_of(0.9, flat), A.percentile_of(1.0, flat), A.percentile_of(1.1, flat)) == (0.0, 50.0, 100.0)
    for bad in (None, float("nan"), "x"):
        assert A.percentile_of(bad, grid) is None
    for bad in (None, [], [1.0], [0.0, 2.0, 1.0], [0.0, float("nan"), 2.0], ["a", "b"], "0,1,2", {"0": 1}):
        assert A.percentile_of(1.0, bad) is None


def test_percentile_never_contradicts_the_level():
    ref = A.load_reference()
    for key, labels in (("hurdle", A.LEVERAGE_LEVELS), ("p_dip20_3m", A.DRAWDOWN_LEVELS)):
        g, c = ref[key]["grid"], ref[key]["cutoffs"]
        for v in np.linspace(g[0], g[-1], 5001):
            p, i = A.percentile_of(v, g, c), labels.index(A.level_of(v, c, labels))
            assert (0.0, 100 / 3, 200 / 3)[i] - 1e-9 <= p <= (100 / 3, 200 / 3, 100.0)[i] + 1e-9
            assert abs(p - A.percentile_of(v, g)) < 2                  # pinning moves the grid's reading little
        assert A.percentile_of(c[0], g, c) == pytest.approx(100 / 3)
        assert A.percentile_of(c[1], g, c) == pytest.approx(200 / 3)
    grid = [float(i) for i in range(21)]
    assert A.percentile_of(5.0, grid, [4.0, 14.0]) == pytest.approx(25.0)   # cutoffs off the grid's terciles: ignored
    assert A.percentile_of(5.0, grid, "bad") == pytest.approx(25.0)


# ---- the reference file --------------------------------------------------------------------
def test_load_reference_never_raises(tmp_path):
    assert A.load_reference(tmp_path / "missing.json") is None
    assert A.load_reference(tmp_path) is None                               # a directory
    p = tmp_path / "ref.json"
    for content in (b"", b"{not json", b"[1, 2]", b"null", b"\xff\xfe\x00garbage", b"[" * 100000 + b"]" * 100000):
        p.write_bytes(content)
        assert A.load_reference(p) is None
    assert A.load_reference(_ref_file(tmp_path)) == REF


def test_shipped_reference_is_well_formed():
    text = A.REFERENCE_PATH.read_text(encoding="utf-8")
    ref = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite {c} in reference.json"))
    assert A.load_reference() == ref and isinstance(ref["period"], str) and ref["n"] > 100
    for key in ("hurdle", "p_dip20_3m"):
        g, (lo, hi) = ref[key]["grid"], ref[key]["cutoffs"]
        assert len(g) == 21 and all(b >= a for a, b in zip(g, g[1:]))
        assert g[0] <= lo <= hi <= g[-1]
        assert abs(A.percentile_of(lo, g) - 100 / 3) < 3 and abs(A.percentile_of(hi, g) - 200 / 3) < 3
    by_level = ref["p_dip20_3m"]["by_level"]
    assert set(by_level) == set(A.DRAWDOWN_LEVELS)
    assert all(0 <= v["pred"] <= 1 and 0 <= v["real"] <= 1 and v["n"] > 0 for v in by_level.values())
    assert {"pred_mean", "real_mean", "corr", "n"} <= set(ref["leverage_cost_check"])


# ---- leverage cost -------------------------------------------------------------------------
@pytest.mark.parametrize("lev, expense, financing, sigma", [
    (3.0, 0.0091, 2 * (0.04 + 0.0075), 0.18), (3.0, 0.0087, 2 * 0.0075, 0.10),
    (2.9, 0.0100, 1.9 * 0.06, 0.35), (3.0, 0.0091, 2 * 0.0075, 0.0)])
def test_hurdle_is_the_etf_breakeven_and_its_parts_add_up(lev, expense, financing, sigma):
    lc = A.leverage_cost(lev, expense, financing, sigma, 0.061, None)
    etf = ETFParams(leverage=lev, expense_ratio=expense, financing_rate=financing, tracking_sd_daily=0.0,
                    calibration=None)
    assert lc.hurdle == pytest.approx(etf.breakeven_index_return(sigma), rel=1e-12, abs=1e-15)
    assert (lc.fees, lc.financing, lc.sigma, lc.expected_index_return) == (expense, financing, sigma, 0.061)
    assert lc.drag == pytest.approx(etf.theoretical_drag(sigma))
    # in the fund's terms: L x ln(1 + hurdle) = fees + financing + drag + L x the index's own sigma^2 / 2
    assert lev * math.log1p(lc.hurdle) == pytest.approx(lc.fees + lc.financing + lc.drag + 0.5 * lev * sigma ** 2)
    assert (lc.level, lc.percentile) == ("n/a", None)


def test_hurdle_rises_with_rates_and_vol_and_takes_its_level_from_the_reference():
    calm = A.leverage_cost(3.0, 0.0091, 2 * 0.0075, 0.12, 0.06, REF)           # near-zero rates, low vol
    dear = A.leverage_cost(3.0, 0.0091, 2 * (0.05 + 0.0075), 0.12, 0.06, REF)
    wild = A.leverage_cost(3.0, 0.0091, 2 * (0.05 + 0.0075), 0.25, 0.06, REF)
    assert calm.hurdle < dear.hurdle < wild.hurdle
    assert calm.level == "low" and 0 <= calm.percentile < 100 / 3
    assert (wild.level, wild.percentile) == ("high", 100.0)
    mid = A.leverage_cost(3.0, 0.0, 3 * math.log(1.06), 0.0, 0.06, REF)         # hurdle exactly 6%
    assert mid.hurdle == pytest.approx(0.06) and mid.level == "normal" and mid.percentile == pytest.approx(30.0)


# ---- drawdown risk -------------------------------------------------------------------------
def test_drawdown_risk_looks_up_the_history_of_its_level():
    for p, level in ((0.10, "low"), (0.17, "normal"), (0.25, "normal"), (0.32, "elevated"), (0.60, "elevated")):
        d = A.drawdown_risk(p, REF)
        h = REF["p_dip20_3m"]["by_level"][level]
        assert d.level == level and d.p_dip20_3m == p and d.typical == 0.24
        assert (d.history_pred, d.history_real, d.history_n) == (h["pred"], h["real"], h["n"])
    assert A.drawdown_risk(0.25, REF).percentile == pytest.approx(50.0)
    ref = copy.deepcopy(REF)
    del ref["p_dip20_3m"]["by_level"]["elevated"]
    d = A.drawdown_risk(0.5, ref)
    assert d.level == "elevated" and (d.history_pred, d.history_real, d.history_n) == (None, None, None)
    d = A.drawdown_risk(None, REF)
    assert math.isnan(d.p_dip20_3m) and d.level == "n/a" and d.percentile is None and d.history_n is None


@pytest.mark.parametrize("ref", [
    None, [], {"p_dip20_3m": "x"}, {"p_dip20_3m": {"cutoffs": "a", "by_level": [1], "grid": 3, "median": "m"}},
    {"p_dip20_3m": {"cutoffs": [0.1, 0.2], "by_level": {"normal": "x"}}},
    {"p_dip20_3m": {"cutoffs": [0.1, 0.2], "by_level": {"normal": {"pred": "x", "real": None, "n": "many"}}}},
    {"hurdle": {"grid": [None] * 21, "cutoffs": [None, 1]}}])
def test_garbage_reference_degrades_to_na(ref):
    d = A.drawdown_risk(0.15, ref)
    assert (d.history_pred, d.history_real, d.history_n, d.percentile, d.typical) == (None,) * 5
    assert A.leverage_cost(3.0, 0.0091, 0.1, 0.18, 0.06, ref).level == "n/a"


# ---- the whole assessment ------------------------------------------------------------------
def test_assess_reads_the_forecast(tmp_path):
    fc = _fc()
    a = A.assess(fc, _ref_file(tmp_path))
    assert a.leverage.hurdle == pytest.approx(fc.etf.breakeven_index_return(0.19))    # the pipeline's sigma_1y
    assert a.leverage.expected_index_return == 0.061
    assert a.drawdown.p_dip20_3m == fc.sim.summary(63)["p_drawdown_20"]
    q = fc.sim.quantiles(63)
    assert a.range_3m == (q[5.0], q[50.0], q[95.0]) and q[5.0] < 200.0 < q[95.0]
    assert a.leverage.level in A.LEVERAGE_LEVELS and a.drawdown.level in A.DRAWDOWN_LEVELS
    assert a.period == "1990-01..2026-08" and a.notes == []
    fc.vol = SimpleNamespace(one_year_vol=None, total_vol=lambda h: 0.21 if h == 252 else 0.0)
    assert A.assess(fc, _ref_file(tmp_path)).leverage.sigma == 0.21      # the pipeline's fallback


def test_assess_without_reference_or_3m_horizon(tmp_path):
    a = A.assess(_fc(horizons=(21, 126)), tmp_path / "missing.json")
    assert a.leverage.level == "n/a" and a.drawdown.level == "n/a" and a.range_3m is None and a.period is None
    assert math.isfinite(a.leverage.hurdle) and math.isnan(a.drawdown.p_dip20_3m)
    assert len(a.notes) == 2


def test_assessment_to_dict_is_strict_json(tmp_path):
    good = A.assess(_fc(), _ref_file(tmp_path))
    bad = A.assess(_fc(horizons=(21, 126), one_year_vol=float("nan")), tmp_path / "missing.json")
    for a in (good, bad):
        d = A.assessment_to_dict(a)
        assert json.loads(json.dumps(d, allow_nan=False)) == d
    d = A.assessment_to_dict(good)
    assert d["range_3m"]["p5"] < d["range_3m"]["median"] < d["range_3m"]["p95"]
    assert d["drawdown"]["history_n"] == REF["p_dip20_3m"]["by_level"][d["drawdown"]["level"]]["n"]
    assert d["horizon_days"] == 63 and d["period"] == "1990-01..2026-08"
    d = A.assessment_to_dict(bad)
    assert d["leverage"]["hurdle"] is None and d["drawdown"]["p_dip20_3m"] is None and d["range_3m"] is None


# ---- the backtest writes what the assessment reads -----------------------------------------
def test_backtest_reference_feeds_the_assessment():
    sys.path.insert(0, str(ROOT / "scripts"))
    import backtest_rating as br
    n = 120
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, n)
    o = pd.DataFrame({"date": pd.date_range("2000-01-31", periods=n, freq="ME"), "pos": np.arange(n) * 21,
                      "hurdle": rng.uniform(0.03, 0.12, n), "p_dip20_3m": p,
                      "real_dd20_63": (rng.random(n) < p).astype(float),
                      "lc_fees": 0.0091, "lc_financing": 0.06, "lc_drag": rng.uniform(0.05, 0.15, n)})
    o["real_cost"] = o["lc_fees"] + o["lc_financing"] + o["lc_drag"] + rng.normal(0.01, 0.03, n)
    ref = br.build_reference(o, "test")
    json.dumps(ref, allow_nan=False)
    assert ref["n"] == n and ref["period"] == "2000-01..2009-12" and len(ref["hurdle"]["grid"]) == 21
    by_level = ref["p_dip20_3m"]["by_level"]
    assert sum(v["n"] for v in by_level.values()) == n
    for lab, v in by_level.items():             # a level's mean prediction lies inside that level
        assert A.drawdown_risk(v["pred"], ref).level == lab
    check, spxl = ref["leverage_cost_check"], o[o["date"] >= "2009-01-01"]
    assert check["n"] == 12 and check["period"] == "2009-01..2009-12"             # SPXL months only
    assert check["pred_mean"] == pytest.approx(0.0691 + spxl["lc_drag"].mean(), abs=1e-6)
    assert check["real_mean"] == pytest.approx(spxl["real_cost"].mean(), abs=1e-6) and -1 <= check["corr"] <= 1
