"""Train-phase statistics of scripts/research_signals.py (PREREGISTRATION.md sections 7-10 and 15):
the one-sided p against the backtest's interval, Holm on known inputs, the composite on synthetic
data, the train quantities (predictor values at parameter-window month-ends only), the seal on the
train panel, and the frozen spec's determinism. Offline, synthetic data."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_signals as rs  # noqa: E402

T = pd.Timestamp


def _panel(seed: int = 0) -> pd.DataFrame:
    """A train panel shaped like panel_train.pkl: month-ends 1990-01..2007-12 on a business-day
    calendar, candidates, flags and targets only where the window ends by 2007-12-31."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("1989-11-01", "2008-01-15")          # month_end_positions drops the last month
    ends = [cal[i] for i in rs.br.month_end_positions(cal, "1990-01")]
    pos = np.array([cal.get_loc(t) for t in ends])
    idx = pd.DatetimeIndex(ends)
    n = len(idx)
    p = pd.DataFrame(index=idx)
    for c in rs.CAND_IDS:
        p[c] = rng.standard_normal(n)
    p["C10"] = 0.01 + 0.05 * rng.random(n)                     # IVAR is positive
    p.loc[idx[:6], "C07"] = np.nan                             # SLOOS starts in 1990-07
    p.loc[T("1996-01-31"), "C12"] = np.nan
    p["pos"] = pos
    p["gspc_tm"] = 300.0 * np.cumprod(1.0 + 0.03 * rng.standard_normal(n))
    p["in_param_window"] = (idx >= rs.PARAM_WINDOW[0]) & (idx <= rs.PARAM_WINDOW[1])
    seal_pos = pos[-1]
    for h in rs.HORIZONS:
        p[f"train_h{h}"] = pos + h <= seal_pos
        p[f"test_h{h}"] = False
    for k in rs.TARGETS:
        h = rs.DIP_H if k == "D20" else int(k[1:])
        v = (rng.random(n) < 0.2).astype(float) if k == "D20" else 0.1 * rng.standard_normal(n)
        p[k] = np.where(p[f"train_h{h}"], v, np.nan)
    return p


def _spec_inputs():
    prov = {"prereg": {"research/PREREGISTRATION.md": "a" * 64, "research/prereg.json": "b" * 64},
            "prereg_frozen_utc": "2026-09-27T16:49:20Z", "pinned": {"output/backtest/x": "c" * 64},
            "predictor_inputs": "d" * 64, "panel_train": "e" * 64, "panel_meta": "f" * 64, "script": "0" * 64}
    sloos = {"branch_shift_months": 0, "shift0": {"n_checked": 67, "n_violations": 0, "first_checked_obs": "2010-01-01"}}
    return prov, sloos


# ---- section 9: the one-sided p and Holm --------------------------------------------------
@pytest.mark.parametrize("n_eff", [5.5, 12.0, 35.8, 37.1, 74.2, 218.2])
def test_one_sided_p_agrees_with_the_backtest_interval(n_eff):
    for r in np.linspace(-0.6, 0.8, 57):
        z, p = rs.one_sided(float(r), n_eff)
        assert z == pytest.approx(math.atanh(r) * math.sqrt(n_eff - 3) / math.sqrt(1 + r * r / 2))
        assert p == pytest.approx(norm.sf(z))
        lo, _ = rs.br._corr_ci(float(r), n_eff)
        if abs(p - 0.05) > 1e-9:
            assert (p < 0.05) == (lo > 0)                      # the plan's stated equivalence


def test_one_sided_p_edge_cases():
    assert rs.one_sided(float("nan"), 30.0)[1] == 1.0
    assert rs.one_sided(0.5, 4.0)[1] == 1.0                    # n_eff <= 4
    assert rs.one_sided(1.0, 30.0)[1] == 0.0
    assert rs.one_sided(-1.0, 30.0)[1] == 1.0


def test_holm_on_known_inputs():
    got = rs.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted 0.005 (<= 0.0125), 0.01 (<= 0.01667), 0.03 (> 0.025: stop), 0.04
    assert [got[k]["reject"] for k in "abcd"] == [True, False, False, True]
    assert [got[k]["p_holm"] for k in "dacb"] == pytest.approx([0.02, 0.03, 0.06, 0.06])
    assert got["d"]["threshold"] == pytest.approx(0.05 / 4) and got["b"]["step"] == 4


def test_holm_stops_at_the_first_failure_and_matches_adjusted_p():
    rng = np.random.default_rng(3)
    for _ in range(200):
        ps = {f"H{i}": float(v) for i, v in enumerate(rng.random(13) ** 3)}
        got = rs.holm(ps)
        for k, v in got.items():
            assert v["reject"] == (v["p_holm"] <= 0.05 + 1e-15)
        order = sorted(ps, key=ps.get)
        rej = [got[k]["reject"] for k in order]
        assert rej == sorted(rej, reverse=True)                # a prefix of the sorted list


def test_holm_ties_keep_the_given_order():
    got = rs.holm({"x": 0.004, "y": 0.004, "z": 0.9})
    assert got["x"]["step"] == 1 and got["y"]["step"] == 2


def test_corr_test_is_overlap_aware():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(100), rng.standard_normal(100)
    pos = np.arange(100) * 21
    a = rs.corr_test(x, y, pos, 21)
    b = rs.corr_test(x, y, pos, 126)
    assert a["r"] == b["r"] and a["n"] == 100
    assert a["n_eff"] == pytest.approx(100.0) and b["n_eff"] == pytest.approx(rs.effective_n(pos, 126))
    assert b["p"] > a["p"] or a["r"] <= 0                       # fewer independent windows, weaker evidence
    x[:5] = np.nan
    assert rs.corr_test(x, y, pos, 21)["n"] == 95


# ---- section 7: the composite -------------------------------------------------------------
def test_composite_signs_clips_averages_and_needs_seven_members():
    std = {c: {"mu": 1.0, "sd": 2.0} for c in rs.CAND_IDS}
    base = {c: 1.0 for c in rs.MEMBERS}                         # every z = 0
    rows = [dict(base), dict(base, C01=11.0), dict(base, C02=11.0), dict(base, C01=np.nan, C02=np.nan),
            dict(base, C01=np.nan, C02=np.nan, C03=np.nan)]
    p = pd.DataFrame(rows, index=pd.date_range("2000-01-31", periods=5, freq="ME"))
    comp, n = rs.composite(p, std)
    assert comp.iloc[0] == 0.0
    assert comp.iloc[1] == pytest.approx(3.0 / 9)              # C01 sign +, z = 5 clipped to 3
    assert comp.iloc[2] == pytest.approx(-3.0 / 9)             # C02 sign -, clipped to -3
    assert n.iloc[3] == 7 and comp.iloc[3] == 0.0
    assert n.iloc[4] == 6 and np.isnan(comp.iloc[4])           # fewer than 7 members
    assert list(n) == [9, 9, 9, 7, 6]


def test_composite_ignores_non_members():
    std = {c: {"mu": 0.0, "sd": 1.0} for c in rs.CAND_IDS}
    p = pd.DataFrame({c: [0.5] for c in rs.CAND_IDS}, index=[T("2000-01-31")])
    a, _ = rs.composite(p, std)
    p[["C10", "C11", "C12"]] = 99.0
    b, _ = rs.composite(p, std)
    assert a.iloc[0] == b.iloc[0]


# ---- section 8: the train quantities --------------------------------------------------------
def test_train_quantities_follow_their_definitions():
    p = _panel()
    q = rs.train_quantities(p)
    pw = p.loc[p["in_param_window"]]
    assert len(pw) == 210
    for c in rs.CAND_IDS:
        x = pw[c].dropna()
        assert q["standardisation"][c] == {"n": len(x), "mu": pytest.approx(x.mean()), "sd": pytest.approx(x.std(ddof=1))}
        assert q["tau_j"][c] == pytest.approx(np.percentile(rs.SIGNS[c] * x, 33.33))
    comp, _ = rs.composite(pw, q["standardisation"])
    assert q["composite"]["tau"] == pytest.approx(np.percentile(comp.dropna(), 33.33))
    assert q["composite"]["tau_median"] == pytest.approx(np.median(comp.dropna()))
    c = np.median(pw["C10"])
    assert q["vol_managed"]["c"] == pytest.approx(c)
    assert q["vol_managed"]["w_bar"] == pytest.approx(np.mean(np.minimum(1.0, c / pw["C10"])))
    assert q["standardisation"]["C07"]["n"] == 204


def test_the_spec_never_depends_on_outcomes_or_on_month_ends_after_the_window():
    prov, sloos = _spec_inputs()
    p = _panel()
    ref = rs.spec_bytes(rs.frozen_spec(rs.train_quantities(p), prov, sloos))
    q = p.copy()
    q[rs.TARGETS] = q[rs.TARGETS] * -7.0 + 1.0                  # every outcome changed
    late = q.index > rs.PARAM_WINDOW[1]                        # 2007-07..2007-12 (embargo side)
    q.loc[late, rs.CAND_IDS] = 1e6
    assert rs.spec_bytes(rs.frozen_spec(rs.train_quantities(q), prov, sloos)) == ref
    q.loc[T("2000-06-30"), "C05"] += 1.0                       # a parameter-window predictor value
    assert rs.spec_bytes(rs.frozen_spec(rs.train_quantities(q), prov, sloos)) != ref


def test_the_frozen_spec_is_deterministic_json_with_lf():
    prov, sloos = _spec_inputs()
    q = rs.train_quantities(_panel())
    a = rs.spec_bytes(rs.frozen_spec(q, prov, sloos))
    b = rs.spec_bytes(rs.frozen_spec(rs.train_quantities(_panel()), prov, sloos))
    assert a == b and a.endswith(b"\n") and b"\r" not in a

    def no_constant(name):
        raise AssertionError(f"{name} in the spec")

    spec = json.loads(a, parse_constant=no_constant)            # no NaN or Infinity
    assert spec["train_set"]["composite"]["tau"] == q["composite"]["tau"]
    assert [c["id"] for c in spec["candidates"]] == rs.CAND_IDS
    assert spec["primary_test"]["K"] == 13 and spec["primary_test"]["predictors"][-1] == "COMP"


def test_train_quantities_stop_on_a_wrong_parameter_window():
    p = _panel()
    p.loc[T("2007-07-31"), "in_param_window"] = True
    with pytest.raises(SystemExit, match="parameter window"):
        rs.train_quantities(p)


# ---- the seal on the train panel -----------------------------------------------------------
def test_train_panel_seal_checks():
    p = _panel()
    assert rs.check_train_panel(p)["Y126"] == int(p["train_h126"].sum())
    bad = p.copy()
    bad.loc[T("2007-12-31"), "Y21"] = 0.1                      # an embargo month-end with an outcome
    with pytest.raises(rs.SealError, match="not a train origin"):
        rs.check_train_panel(bad)
    bad = p.copy()
    bad.loc[T("2007-09-28"), "train_h126"] = True              # its window would end in 2008
    with pytest.raises(rs.SealError, match="after 2007-12-31"):
        rs.check_train_panel(bad)
    bad = pd.concat([p, p.iloc[[-1]].rename(index={T("2007-12-31"): T("2008-01-31")})])
    with pytest.raises(rs.SealError, match="test period"):
        rs.check_train_panel(bad)
    bad = p.copy()
    bad["test_h21"] = True
    with pytest.raises(rs.SealError, match="test origin"):
        rs.check_train_panel(bad)


def test_the_train_phase_refuses_panel_test_before_opening_it(tmp_path):
    with pytest.raises(rs.SealError, match="panel_train.pkl"):
        rs.load_train_panel(tmp_path / "panel_test.pkl")        # does not exist: refused by name, never opened
    p = _panel()
    p.to_pickle(tmp_path / "panel_train.pkl")
    assert rs.load_train_panel(tmp_path / "panel_train.pkl").equals(p)


# ---- section 10: the replication ------------------------------------------------------------
def test_replication_covers_seven_families_of_thirteen_on_train_origins():
    from spxlcast.evaluation import spearman
    p = _panel()
    q = rs.train_quantities(p)
    comp, _ = rs.composite(p, q["standardisation"])
    rep = rs.replication(p, comp)
    assert list(rep) == ["primary", "F2", "F3", "F4", "F5", "F6", "F7"]
    for fam, f in rep.items():
        assert list(f["rows"]) == rs.PREDICTORS
    sel = p["train_h126"]
    v = rep["primary"]["rows"]["C02"]
    assert v["r"] == pytest.approx(spearman(-p.loc[sel, "C02"].values, p.loc[sel, "Y126"].values))
    assert v["n"] == int(sel.sum()) and v["n_eff"] == pytest.approx(rs.effective_n(p.loc[sel, "pos"], 126))
    d = rep["F7"]["rows"]["C01"]
    sel = p["train_h63"]
    assert d["r"] == pytest.approx(spearman(p.loc[sel, "C01"].values, -p.loc[sel, "D20"].values))
    assert rep["primary"]["rows"]["C07"]["n"] == 204


def test_ta_diagnostic_uses_trailing_price_returns_of_the_panel():
    from spxlcast.evaluation import spearman
    p = _panel()
    comp, _ = rs.composite(p, rs.train_quantities(p)["standardisation"])
    ta = rs.ta_diagnostic(p, comp)
    ret = p["gspc_tm"] / p["gspc_tm"].shift(3) - 1
    m = p["in_param_window"] & ret.notna()
    assert ta["C04"]["3m"]["r"] == pytest.approx(spearman(p.loc[m, "C04"].values, ret[m].values))
    assert ta["C04"]["12m"]["n"] == 210 - 12                   # the first 12 month-ends lack a trailing year


# ---- the hash files --------------------------------------------------------------------------
def test_hash_files_are_readable_by_the_code_guard(tmp_path):
    spec_sha, pi_sha = "1" * 64, rs.PREDICTOR_INPUTS_SHA256
    report = rs.render_train_report(pi_sha, spec_sha, "2026-09-27T18:00:00Z")
    assert rs._recorded_hash(report, "train_params.json") == spec_sha
    assert rs._recorded_hash(report, "predictor_inputs.pkl") == pi_sha
    fh = tmp_path / "FROZEN_HASH.txt"
    rs.write_lf(fh, rs.render_frozen_hash(spec_sha, "2026-09-27T18:00:00Z"))
    assert rs.read_prereg_hashes(fh) == {"research/frozen_spec.json": spec_sha}
    assert b"\r" not in fh.read_bytes()
