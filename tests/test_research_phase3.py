"""Test phase (Phase 3) of scripts/research_signals.py on synthetic data only: the economic test's
mechanics (switch costs, drifting sleeves, metrics), the bootstraps as frozen, the partial Spearman,
the expanding-window composite, the verdict in the registered words, the gate on the CSV's outcome
columns, and the whole `test` command end to end in a synthetic world (run once, `report`
reproduces the document). No real data is read and nothing is unsealed."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_signals as rs  # noqa: E402

T = pd.Timestamp


# ---- a synthetic world shaped like the real one ---------------------------------------------------
def _inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1989-06-01", "2026-09-25")
    r = 0.0003 + 0.011 * rng.standard_normal(len(idx))
    tr = 300.0 * np.cumprod(1.0 + r)
    vix = 12.0 + 25.0 * rng.random(len(idx))
    spxl = pd.Series(np.nan, index=idx)
    live = idx >= T("2008-11-05")
    spxl[live] = 10.0 * np.cumprod(1.0 + 3.0 * r[live] - 0.0001 + 0.0005 * rng.standard_normal(live.sum()))
    closes = pd.DataFrame({"^SP500TR": tr, "^GSPC": tr / 2.0, "^VIX": vix, "^VIX3M": np.nan, "^VIX6M": np.nan,
                           "SPXL": spxl.values}, index=idx)
    fred = {"DGS3MO": pd.Series(np.clip(3.0 + np.cumsum(0.01 * rng.standard_normal(len(idx))), 0.01, 9), index=idx),
            "DGS10": pd.Series(4.5, index=idx), "T10YIE": pd.Series(2.2, index=idx),
            "BAMLH0A0HYM2": pd.Series(4.0, index=pd.bdate_range("2023-09-01", "2026-09-25"))}
    return {"closes": closes, "fred": fred, "shiller": pd.DataFrame(), "fetched": "synthetic"}


def _world(seed: int = 0, informative: str = "C04"):
    """(inputs, cal, full daily frame, panel_train, panel_test) as `build` would write them, with one
    candidate made informative about the next 6 months so every branch of the verdict is reachable."""
    rng = np.random.default_rng(seed + 1)
    inputs = _inputs(seed)
    cal = rs.session_calendar(inputs)
    positions = rs.origin_positions(cal)
    flags = rs.split_flags(cal, positions)
    d_full = rs.br.build_daily(inputs, rs.Config())
    full_targets = rs.compute_targets(d_full, positions, cal)
    idx = flags.index
    n = len(idx)
    p = pd.DataFrame(index=idx)
    for c in rs.CAND_IDS:
        p[c] = rng.standard_normal(n)
    if informative:
        y = full_targets["Y126"].fillna(0.0).to_numpy()
        p[informative] = rs.SIGNS[informative] * (y / (y.std() + 1e-12) + 0.3 * rng.standard_normal(n))
    p["C10"] = 0.005 + 0.08 * rng.random(n)
    p.loc[idx[:6], "C07"] = np.nan
    p.loc[[T("1996-01-31"), T("2025-10-31")], "C12"] = np.nan
    p["pos"] = np.asarray(positions, dtype=np.int64)
    p["vix_tm"] = 10.0 + 30.0 * rng.random(n)
    p["y3_tm"] = 0.1 + 5.0 * rng.random(n)
    p["gspc_tm"] = 100.0 + rng.random(n)
    p = p.join(flags)
    sealed = rs.compute_targets(rs.outcome_frame(inputs), positions, cal)
    for k in rs.TARGETS:
        p[k] = sealed[k].to_numpy()
    p_train = p.loc[p.index < rs.TEST_START].copy()
    p_test = p.loc[p.index >= rs.TEST_START].copy()
    p_test[rs.TARGETS] = np.nan
    return inputs, cal, d_full.loc[d_full.index <= rs.DATA_CUTOFF], p_train, p_test, full_targets


def _prov(md_sha="a" * 64, js_sha="b" * 64, pinned=None, pi="d" * 64, panel_train="e" * 64):
    return {"prereg": {"research/PREREGISTRATION.md": md_sha, "research/prereg.json": js_sha},
            "prereg_frozen_utc": "2026-09-27T16:49:20Z", "pinned": pinned or {"output/backtest/x": "c" * 64},
            "predictor_inputs": pi, "panel_train": panel_train}


SLOOS = {"branch_shift_months": 0, "shift0": {"n_checked": 67, "n_violations": 0, "first_checked_obs": "2010-01-01"}}


def _csv_outcomes(full_targets, perturb=None):
    out = {k: {t: float(v) for t, v in full_targets[k].items()} for k in rs.TARGETS}
    if perturb:
        k, t, dv = perturb
        out[k][t] += dv
    return out


@pytest.fixture(scope="module")
def world():
    return _world()


@pytest.fixture(scope="module")
def computed(world):
    inputs, cal, d, p_train, p_test, full = world
    spec = json.loads(rs.spec_bytes(rs.frozen_spec(rs.train_quantities(p_train), _prov(), SLOOS)))
    legacy = pd.Series(np.random.default_rng(5).standard_normal(len(full)), index=full.index)
    d_spread = {s: rs.br.build_daily(inputs, rs.dataclasses.replace(rs.Config(), swap_spread=s)).loc[:rs.DATA_CUTOFF]
                for s in rs.SPREADS}
    res, frames = rs.phase3_compute(p_train, p_test, d, cal, spec, _csv_outcomes(full), legacy, d_spread)
    return spec, res, frames


# ---- the economic mechanics -----------------------------------------------------------------------
def _segments():
    a = np.array([0, 3, 6, 9])
    b = np.array([3, 6, 9, 12])
    fund = np.array([0.0, 0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.00, 0.01, -0.03, 0.02, 0.01, 0.05])
    bill = np.full(13, 0.0002)
    return a, b, fund, bill


def test_switching_rule_pays_the_cost_on_the_first_session_of_a_new_holding_only():
    a, b, fund, bill = _segments()
    s = rs.simulate_weights([1, 0, 0, 1], a, b, fund, bill, 0.001)
    want = np.concatenate([fund[1:4], bill[4:7], bill[7:10], fund[10:13]])
    want[3] = 0.999 * (1 + bill[4]) - 1                     # switch to T-bills
    want[9] = 0.999 * (1 + fund[10]) - 1                    # switch back
    assert np.allclose(s["r"], want, rtol=0, atol=1e-15)
    assert list(s["traded"]) == [0, 1, 0, 1] and s["cost_rate"].sum() == pytest.approx(0.002)
    w1 = np.prod(1 + want[:3])
    w2 = w1 * np.prod(1 + want[3:9])
    assert s["cost_paid"] == pytest.approx(0.001 * w1 + 0.001 * w2)
    assert rs.simulate_weights([1, 1, 1, 1], a, b, fund, bill, 0.001)["r"].tolist() == fund[1:13].tolist()
    # the first allocation is free
    assert rs.simulate_weights([0, 0, 0, 0], a, b, fund, bill, 0.001)["cost_rate"].sum() == 0.0


def test_mixed_weights_drift_within_the_month_and_pay_for_the_rebalance():
    a, b, fund, bill = _segments()
    w = 0.4
    s = rs.simulate_weights([w] * 4, a, b, fund, bill, 0.001)
    vf, vb, costs = w, 1 - w, []
    for i in range(4):
        if i:
            c = 0.001 * abs(w - vf / (vf + vb))
            costs.append(c)
            tot = (vf + vb) * (1 - c)
            vf, vb = w * tot, (1 - w) * tot
        for j in range(a[i] + 1, b[i] + 1):
            vf *= 1 + fund[j]
            vb *= 1 + bill[j]
    wealth = np.prod(1 + s["r"])
    assert wealth == pytest.approx(vf + vb, rel=1e-13)
    assert s["cost_rate"][1:] == pytest.approx(costs)
    assert s["monthly"].tolist() == pytest.approx([np.prod(1 + s["r"][3 * i: 3 * i + 3]) - 1 for i in range(4)])


def test_perf_follows_the_backtest_definitions():
    rng = np.random.default_rng(0)
    r = 0.001 + 0.02 * rng.standard_normal(500)
    bill = np.full(500, 0.0001)
    dates = pd.bdate_range("2015-01-01", periods=500)
    got = rs.perf(r, bill, dates)
    W = np.cumprod(1 + r)
    assert got["cagr"] == pytest.approx(W[-1] ** (252 / 500) - 1)
    assert got["sharpe"] == pytest.approx((r - bill).mean() / (r - bill).std(ddof=1) * math.sqrt(252))
    assert got["max_drawdown"] == pytest.approx(np.max(1 - W / np.maximum.accumulate(W)))
    m = dates <= T("2016-12-30")
    e = (r - bill)[m]
    assert got["sharpe_halves"][0] == pytest.approx(e.mean() / e.std(ddof=1) * math.sqrt(252))
    assert got["half_sessions"][0] == int(m.sum())


def test_rule_weights_keep_the_previous_holding_when_the_signal_is_missing():
    w, miss = rs.rule_weights([np.nan, 0.5, np.nan, -1.0, 0.2, np.nan], 0.2)
    assert w.tolist() == [0, 1, 1, 0, 1, 1] and miss == 3


def test_e_criteria_are_strict():
    base = {"cagr": 0.1, "sharpe": 0.5, "max_drawdown": 0.5, "sharpe_halves": [0.4, 0.6]}
    pf = {"rule": dict(base), "BH": dict(base), "MIX_w": dict(base, cagr=0.0, sharpe=0.0)}
    assert rs.e_criteria(pf) == {"E1": False, "E2": False, "E3": False, "E4": False}
    pf["rule"] = {"cagr": 0.2, "sharpe": 0.7, "max_drawdown": 0.3, "sharpe_halves": [0.5, 0.5]}
    assert rs.e_criteria(pf) == {"E1": True, "E2": True, "E3": True, "E4": False}


# ---- the bootstraps as frozen -------------------------------------------------------------------
def test_stationary_bootstrap_indices_follow_the_frozen_algorithm():
    idx = rs.stationary_bootstrap_idx(40, n_boot=300, mean_block=6, seed=rs.SEED)
    rng = np.random.default_rng(rs.SEED)
    U = rng.random((300, 40))
    S = rng.integers(0, 40, size=(300, 40))
    assert (idx[:, 0] == S[:, 0]).all()
    for i in range(1, 40):
        want = np.where(U[:, i] < 1 / 6, S[:, i], (idx[:, i - 1] + 1) % 40)
        assert (idx[:, i] == want).all()
    runs = (np.diff(idx, axis=1) % 40 == 1).mean()
    assert 0.75 < runs < 0.9                                        # about 5/6 continue the block
    assert (rs.stationary_bootstrap_idx(40, 300, 6, rs.SEED) == idx).all()


def test_stationary_bootstrap_of_a_strategy_against_itself_is_zero():
    rng = np.random.default_rng(1)
    m = 0.01 + 0.05 * rng.standard_normal(223)
    bill = np.full(223, 0.001)
    got = rs.stationary_bootstrap(m, m, bill)
    assert got["sharpe_diff"] == 0 and got["cagr_diff"] == 0
    assert got["sharpe_diff_ci90"] == [0.0, 0.0] and got["cagr_diff_ci90"] == [0.0, 0.0]
    assert rs.monthly_cagr(m) == pytest.approx(np.prod(1 + m) ** (12 / 223) - 1)


def test_moving_block_bootstrap_rows_are_blocks_of_twelve_from_a_fresh_rng():
    from spxlcast.evaluation import spearman
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal(50), rng.standard_normal(50)
    got = rs.block_bootstrap_ci(x, y, n_boot=200)
    r2 = np.random.default_rng(rs.SEED)
    starts = r2.integers(0, 50 - 12 + 1, size=(200, 5))
    rb = []
    for st in starts:
        rows = np.concatenate([np.arange(s, s + 12) for s in st])[:50]
        rb.append(spearman(x[rows], y[rows]))
    assert got["ci90"] == pytest.approx(list(np.percentile(rb, [5, 95])))
    assert rs.block_bootstrap_ci(x, y, n_boot=200) == got               # fresh rng each call


# ---- partial Spearman, expanding composite ------------------------------------------------------------
def test_partial_spearman_removes_a_shared_control():
    from scipy.stats import rankdata
    rng = np.random.default_rng(3)
    c1, c2 = rng.standard_normal(300), rng.standard_normal(300)
    x = c1 + 0.2 * rng.standard_normal(300)
    y = c1 + 0.2 * rng.standard_normal(300)
    pos = np.arange(300) * 21
    got = rs.partial_spearman(x, y, c1, c2, pos, 21)
    assert abs(got["r"]) < 0.2 and got["n"] == 300
    R = [rankdata(v) for v in (x, y, c1, c2)]
    X = np.column_stack([np.ones(300), R[2], R[3]])
    res = [v - X @ np.linalg.lstsq(X, v, rcond=None)[0] for v in R[:2]]
    assert got["r"] == pytest.approx(np.corrcoef(*res)[0, 1])
    z = math.atanh(got["r"]) * math.sqrt(got["n_eff"] - 5) / math.sqrt(1 + got["r"] ** 2 / 2)
    assert got["z"] == pytest.approx(z)
    x[:10] = np.nan
    assert rs.partial_spearman(x, y, c1, c2, pos, 21)["n"] == 290


def test_one_sided_with_five_degrees_lost():
    assert rs.one_sided(0.3, 6.0, dof_loss=5)[1] == 1.0
    z, p = rs.one_sided(0.3, 37.0, dof_loss=5)
    assert z == pytest.approx(math.atanh(0.3) * math.sqrt(32) / math.sqrt(1.045))
    assert rs.one_sided(0.3, 37.0) == rs.one_sided(0.3, 37.0, dof_loss=3)


def test_expanding_composite_matches_an_explicit_loop():
    rng = np.random.default_rng(4)
    idx = pd.date_range("1990-01-31", periods=150, freq="ME")
    p = pd.DataFrame({c: rng.standard_normal(150) for c in rs.CAND_IDS}, index=idx)
    p.loc[idx[:6], "C07"] = np.nan
    p.loc[idx[100], "C03"] = np.nan
    comp, n = rs.expanding_composite(p)
    for i in (58, 59, 64, 65, 100, 149):
        zs = []
        for c in rs.MEMBERS:
            hist = p[c].iloc[: i + 1].dropna()
            if np.isfinite(p[c].iloc[i]) and len(hist) >= 60:
                z = rs.SIGNS[c] * (p[c].iloc[i] - hist.mean()) / hist.std(ddof=1)
                zs.append(float(np.clip(z, -3, 3)))
        assert n.iloc[i] == len(zs)
        if len(zs) >= 7:
            assert comp.iloc[i] == pytest.approx(np.mean(zs))
        else:
            assert np.isnan(comp.iloc[i])
    assert np.isnan(comp.iloc[58]) and np.isfinite(comp.iloc[59])    # 60 values needed (C07 later)


# ---- the verdict ---------------------------------------------------------------------------------
def test_verdict_uses_the_registered_words():
    spec = json.loads(rs.FROZEN_SPEC.read_text(encoding="utf-8")) if rs.FROZEN_SPEC.exists() else None
    if spec is None:
        pytest.skip("no frozen spec")
    rows = {pid: {"reject": False} for pid in rs.PREDICTORS}
    rules = {pid: {"pass": False} for pid in rs.PREDICTORS}
    rob = {pid: {"labels": []} for pid in rs.PREDICTORS}
    assert rs.decide(spec, rows, rules, rob)["headline"] == "No timing signal found"
    rules["COMP"]["pass"] = True
    assert rs.decide(spec, rows, rules, rob)["headline"].startswith("Not confirmed (gain could be luck)")
    rows["C09"]["reject"] = True
    got = rs.decide(spec, rows, rules, rob)
    assert got["headline"] == "Predictive, not usable by the registered rule: C09 ECY"
    rules["C09"]["pass"] = True
    rob["C09"]["labels"] = ["one half only"]
    got = rs.decide(spec, rows, rules, rob)
    assert got["headline"] == "Timing signal confirmed: C09 ECY (partly exposed)"
    assert got["per_predictor"]["C09"]["robustness_labels"] == ["one half only"]


def test_the_real_frozen_spec_matches_this_code():
    if not rs.FROZEN_SPEC.exists():
        pytest.skip("no frozen spec")
    spec = json.loads(rs.FROZEN_SPEC.read_text(encoding="utf-8"))
    assert rs.check_spec_against_code(spec)["pass"]
    bad = json.loads(rs.FROZEN_SPEC.read_text(encoding="utf-8"))
    bad["economic_test"]["switch_cost"] = 0.002
    with pytest.raises(SystemExit, match="economic test"):
        rs.check_spec_against_code(bad)


# ---- the computation end to end on synthetic data -----------------------------------------------------
def test_phase3_compute_on_a_synthetic_world(world, computed):
    from spxlcast.evaluation import spearman
    inputs, cal, d, p_train, p_test, full = world
    spec, res, frames = computed
    pr = res["primary"]
    assert list(pr["rows"]) == rs.PREDICTORS and pr["h"] == 126
    sel = p_test["test_h126"].astype(bool)
    assert pr["origins"] == int(sel.sum())
    x = rs.signed(p_test, "C05")[sel].to_numpy()
    y = full.loc[p_test.index[sel], "Y126"].to_numpy()
    assert pr["rows"]["C05"]["r"] == pytest.approx(spearman(x, y))
    assert pr["rows"]["C05"]["n_eff"] == pytest.approx(rs.effective_n(p_test.loc[sel, "pos"], 126))
    assert pr["rows"]["C04"]["reject"]                              # the planted signal passes Holm
    assert list(res["families"]) == ["F2", "F3", "F4", "F5", "F6", "F7"]
    f7 = res["families"]["F7"]["rows"]["C01"]
    s63 = p_test["test_h63"].astype(bool)
    assert f7["r"] == pytest.approx(spearman(p_test.loc[s63, "C01"].to_numpy(), -full.loc[p_test.index[s63], "D20"].to_numpy()))
    # sanity checks pass in a consistent world
    san = res["sanity"]
    assert san["target_reproduction_all_origins"]["pass"]
    assert san["train_targets_unchanged_by_unsealing"]["pass"]
    assert san["synthetic_fund"]["corr"] > 0.99 and san["synthetic_fund"]["pass"]
    assert not san["calendar_test_origins"]["pass"]                  # a weekday calendar is not the plan's
    # the economic test: 223 holdings tiling 2008-02-01..2026-08-31
    econ = res["economic"]
    assert econ["holdings"] == 223 and econ["sessions"] == ["2008-02-01", "2026-08-31"]
    daily, hold = frames["daily"], frames["holdings"]
    assert daily.index[0] == T("2008-02-01") and daily.index[-1] == T("2026-08-31")
    bh = econ["rules"]["COMP"]["strategies"]["BH"]
    assert bh["final_wealth"] == pytest.approx(np.prod(1 + daily["fund_ret"]))
    w = hold["w_rule"].to_numpy()
    comp_at = hold["COMP"].to_numpy()
    assert (w == (comp_at >= spec["train_set"]["composite"]["tau"])).all()
    assert econ["rules"]["COMP"]["w_mix"] == pytest.approx(w.mean())
    assert econ["rules"]["COMP"]["strategies"]["rule"]["switches_or_rebalances"] == int((np.diff(w) != 0).sum())
    # rule returns: the fund or the bill each session, the switch cost on the first session of a new holding
    r = daily["rule"].to_numpy()
    plain = np.where(np.repeat(w, np.diff(np.append(hold.index.map(cal.get_loc), cal.get_loc(T("2026-08-31"))))) == 1,
                     daily["fund_ret"], daily["tbill_ret"])
    assert (np.abs(r - plain) > 1e-12).sum() == int((np.diff(w) != 0).sum())
    for pid in rs.CAND_IDS:
        assert set(econ["rules"][pid]["E"]) == {"E1", "E2", "E3", "E4"}
    assert econ["rules"]["C12"]["missing_signal"] == 1                 # 2025-10-31
    assert econ["stationary_bootstrap"]["n_boot"] == 10000
    # robustness and the verdict
    rob = res["robustness"]
    assert set(rob["per_predictor"]) == set(rs.PREDICTORS)
    assert [h["origins"] for h in rob["per_predictor"]["C01"]["halves"]] == [list(h) for h in rs.HALVES]
    assert set(rob["synthetic_spread"]) == {"0.0025", "0.0125"}
    changed = rob["synthetic_spread"]["0.0025"]
    assert changed["origins_changed"] > 0 and changed["changed_to"] <= "2008-12-31"
    assert "C04" in res["verdict"]["primary_passes"]
    assert res["verdict"]["headline"].startswith(("Timing signal confirmed", "Predictive, not usable"))


def test_results_render_from_json(computed):
    spec, res, frames = computed
    full = {"phase": "test", "run_utc": "2026-09-27T20:00:00Z", "spec_sha256": "1" * 64, "frozen_utc": "x",
            "preconditions": {"predictor_inputs_sha256": "2" * 64, "frozen_spec_sha256": "1" * 64,
                              "spec_matches_code": {"pass": True}, "spec_quantities": {"pass": True, "max_abs_diff": 0.0},
                              "test_panel": {"pass": True, "rows": 224, "first": "2008-01-31", "last": "2026-08-31"},
                              "rebuild": {"pass": True, "cells": 10, "n_mismatches": 0}},
            "look_log": [{"utc": "u", "phase": "test", "command": "py scripts/research_signals.py test --unseal x",
                          "git_describe": "g"}], **res}
    j = json.loads(json.dumps(rs.jsonable(full), allow_nan=False))
    text = rs.render_test_results(j)
    assert text.startswith("# SPXL timing-signal search: Test phase results")
    for part in ("## Verdict", "## 1. Primary test", "## 2. Economic test", "## 3. Secondary tests F2-F7",
                 "## 4. Train replication", "## 5. Secondary economic analyses", "## 6. Robustness",
                 "## 7. Sanity checks", "## 8. Test-period predictor values", "## 9. What was seen"):
        assert part in text
    import re
    assert not re.search(r"\bnan\b", text.lower())      # a missing number shows as the word nan


def test_target_reproduction_catches_a_difference(world):
    inputs, cal, d, p_train, p_test, full = world
    p_all = pd.concat([p_train, p_test])
    for k in rs.TARGETS:
        p_all[k] = full[k].to_numpy()
    assert rs.target_reproduction_all(p_all, _csv_outcomes(full))["pass"]
    bad = rs.target_reproduction_all(p_all, _csv_outcomes(full, ("Y126", T("2015-06-30"), 1e-6)))
    assert not bad["pass"] and bad["Y126"]["n_over_tol"] == 1


# ---- the gate on the CSV's outcome columns and the whole command -------------------------------------
def _write_csv(path, full, legacy):
    header = ["date", "pos", "hurdle", "score"] + [f"{k}_{h}" for h in rs.HORIZONS for k in ("excess", "sp", "tbill")] \
        + [f"real_dd20_{rs.DIP_H}"]
    fmt = lambda v: "" if not np.isfinite(v) else repr(float(v))  # noqa: E731
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (t, row) in enumerate(full.iterrows()):
            cells = [f"{t:%Y-%m-%d}", i, 0.08, legacy.iloc[i]]
            for h in rs.HORIZONS:
                cells += [fmt(row[f"Y{h}"]), fmt(row[f"S{h}"] + 0.01), fmt(0.01 if np.isfinite(row[f"S{h}"]) else np.nan)]
            cells.append(fmt(row["D20"]))
            w.writerow(cells)


def _synthetic_repo(tmp_path, world):
    inputs, cal, d, p_train, p_test, full = world
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    res_dir, bt, out = tmp_path / "research", tmp_path / "backtest", tmp_path / "out"
    for p in (res_dir, bt, out):
        p.mkdir()
    md, js = res_dir / "PREREGISTRATION.md", res_dir / "prereg.json"
    md.write_bytes(b"# synthetic plan\n")
    js.write_bytes(rs.PREREG_JSON.read_bytes())
    (res_dir / "PREREG_HASH.txt").write_text(f"x\n\n{sha(md)}  research/PREREGISTRATION.md\n{sha(js)}  research/prereg.json\n",
                                             encoding="utf-8")
    pd.to_pickle(inputs, bt / "rating_inputs.pkl")
    legacy = pd.Series(np.random.default_rng(9).standard_normal(len(full)), index=full.index)
    _write_csv(bt / "rating_backtest.csv", full, legacy)
    (bt / "ie_data.xls").write_bytes(b"shiller")
    pinned = {n: sha(bt / n) for n in ("rating_inputs.pkl", "rating_backtest.csv", "ie_data.xls")}
    pinned_meta = {f"output/backtest/{n}": h for n, h in pinned.items()}
    pd.to_pickle({"series": {}}, out / "predictor_inputs.pkl")
    pi_sha = sha(out / "predictor_inputs.pkl")
    p_train.to_pickle(out / "panel_train.pkl")
    p_test.to_pickle(out / "panel_test.pkl")
    prov = _prov(sha(md), sha(js), pinned_meta, pi_sha, sha(out / "panel_train.pkl"))
    q = rs.train_quantities(p_train)
    b = rs.spec_bytes(rs.frozen_spec(q, prov, SLOOS))
    spec_sha = hashlib.sha256(b).hexdigest()
    (res_dir / "frozen_spec.json").write_bytes(b)
    (out / "train_params.json").write_bytes(b)
    rs.write_lf(res_dir / "FROZEN_HASH.txt", rs.render_frozen_hash(spec_sha, "2026-09-27T17:00:00Z"))
    rs.write_lf(res_dir / "TRAIN_REPORT.md", rs.render_train_report(pi_sha, spec_sha, "2026-09-27T17:00:00Z"))
    meta = {"spec_sha256": {"research/PREREGISTRATION.md": sha(md), "research/prereg.json": sha(js)},
            "predictor_inputs": {"sha256": pi_sha}, "pinned_sha256": pinned_meta, "checks": {"calendar": {"pass": True}},
            "missing_counts": {"panel_test": {c: int(v) for c, v in p_test.isna().sum().items()}}}
    (out / "panel_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    comp, _ = rs.composite(p_train, q["standardisation"])
    (out / "train_stats.json").write_text(json.dumps(rs.jsonable({"replication": rs.replication(p_train, comp)})),
                                          encoding="utf-8")
    paths = rs.Phase3Paths(
        prereg_md=md, prereg_json=js, prereg_hash=res_dir / "PREREG_HASH.txt", train_report=res_dir / "TRAIN_REPORT.md",
        train_params=out / "train_params.json", frozen_spec=res_dir / "frozen_spec.json",
        frozen_hash=res_dir / "FROZEN_HASH.txt", backtest=bt, predictor_inputs=out / "predictor_inputs.pkl",
        panel_train=out / "panel_train.pkl", panel_test=out / "panel_test.pkl", panel_meta=out / "panel_meta.json",
        train_stats=out / "train_stats.json", test_dir=out / "test", results_json=out / "test_results.json",
        results_md=res_dir / "test_results.md", results_md_plan=res_dir / "RESULTS.md", look_log=out / "look_log.csv",
        pinned=pinned, predictor_inputs_sha256=pi_sha)
    cols = [c for c in p_train.columns if c not in rs.TARGETS and not c.startswith(("train_h", "test_h"))
            and c != "in_param_window"]
    rebuild = lambda inputs_, pi_, legacy_, cal_: pd.concat([p_train, p_test])[cols].copy()  # noqa: E731
    return paths, rebuild, sha(md)


def test_csv_outcomes_need_a_verified_unseal(tmp_path, world):
    paths, _, token = _synthetic_repo(tmp_path, world)
    csv_path = paths.backtest / "rating_backtest.csv"
    with pytest.raises(rs.SealError, match="token"):
        rs.read_csv_outcomes(csv_path, "0" * 64, paths.pinned["rating_backtest.csv"], **paths.verify_kwargs())
    got = rs.read_csv_outcomes(csv_path, token, paths.pinned["rating_backtest.csv"], **paths.verify_kwargs())
    full = world[5]
    assert got["Y126"][T("2010-06-30")] == pytest.approx(full.at[T("2010-06-30"), "Y126"])
    assert np.isnan(got["Y126"][T("2026-08-31")])                    # empty field = missing


def test_the_test_command_end_to_end_in_a_synthetic_world(tmp_path, world, monkeypatch):
    paths, rebuild, token = _synthetic_repo(tmp_path, world)
    # a wrong token is refused before anything is read or created
    assert rs.cmd_test(SimpleNamespace(unseal="f" * 64, after_error=False), paths, rebuild) == 2
    assert not paths.test_dir.exists()
    assert rs.cmd_test(SimpleNamespace(unseal=token.upper(), after_error=False), paths, rebuild) == 0
    res = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert (paths.test_dir / rs.COMPLETED).exists() and (paths.test_dir / "tests.csv").exists()
    assert paths.results_md.read_bytes() == paths.results_md_plan.read_bytes()
    assert b"\r" not in paths.results_md.read_bytes()
    pc = res["preconditions"]
    assert pc["rebuild"]["pass"] and pc["test_panel"]["pass"] and pc["spec_quantities"]["pass"]
    assert res["sanity"]["legacy_reproduction"]["n"] > 400
    assert res["sanity"]["target_reproduction_all_origins"]["pass"]
    assert res["sanity"]["train_replication_matches_train_stats"] is True
    tests = pd.read_csv(paths.test_dir / "tests.csv")
    assert len(tests) == 2 * 7 * 13
    # the code that produced the numbers is hashed and copied; every unseal is in the look log
    code = rs.ROOT / "scripts" / "research_signals.py"
    assert res["code_sha256"]["scripts/research_signals.py"] == rs.sha256_file(code)
    assert (paths.test_dir / "code" / "scripts" / "research_signals.py").read_bytes() == code.read_bytes()
    log = list(csv.DictReader(open(paths.look_log, encoding="utf-8")))
    unseals = [r["command"] for r in log if r["phase"] == "unseal"]
    assert len(unseals) == 4                                         # the frame, two spread variants, the CSV
    assert sum("[outcome_frame]" in c for c in unseals) == 3 and sum("[read_csv_outcomes]" in c for c in unseals) == 1
    # it runs once; `report` reproduces the registered document byte for byte
    with pytest.raises(SystemExit, match="already run"):
        rs.cmd_test(SimpleNamespace(unseal=token, after_error=False), paths, rebuild)
    before = paths.results_md.read_bytes()
    paths.results_md.write_text("x", encoding="utf-8")
    fast = SimpleNamespace(mc_reps=200, feedback_reps=200)
    assert rs.cmd_report(fast, paths) == 0
    assert paths.results_md.read_bytes() == before
    # ... and writes the owner's report: plain-language summary, registered tables, exploratory notes, appendices
    import re
    report = paths.results_md_plan.read_text(encoding="utf-8")
    assert report.startswith("# SPXL timing-signal search: results")
    for part in ("## 1. Summary in plain language", "### Recommendation", "## 2. Confirmatory results",
                 "## 3. Exploratory notes (not results)", "## 4. Limitations", "## 5. How to reproduce",
                 "## Appendix A. Registered results in full", "### Verdict", "### A1. Primary test", "#### A5.1",
                 "### A9. What was seen", "## Appendix B."):
        assert part in report, part
    assert not re.search(r"\bnan\b", report.lower()) and "None%" not in report
    assert b"\r" not in paths.results_md_plan.read_bytes()
    ex = json.loads((paths.results_json.parent / rs.EXPLORATORY_NAME).read_text(encoding="utf-8"))
    fams = ex["mc"]["families"]
    for f in fams.values():                                          # the outcomes rebuild exactly
        assert f["rebuild_max_abs_diff"] < 1e-9
        assert f["origins_outside_daily"] > 0 or f["r_max_abs_diff"] < 1e-12
    assert fams["F2"]["origins_outside_daily"] == 0                  # (a weekday calendar can push 6-month windows out)
    assert fams["primary"]["per_block"]["63"]["rows"]["C04"]["p_mc"] <= 0.01    # the planted signal
    assert ex["economics"]["cagr_check_max_abs_diff"] < 1e-12                   # the rules reproduce section 5.1
    assert ex["claims_vintage"] == {"available": False, "reason": "no ICNSA vintages in predictor_inputs.pkl"}
    assert ex["splice"]["available"] and ex["splice"]["origins_changed"] > 0
    manifest = (paths.results_md.parent / rs.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    listed = {Path(line.split("  ", 1)[1]): line.split("  ", 1)[0] for line in manifest}
    assert paths.results_json in listed and paths.panel_train in listed
    assert all(rs.sha256_file(f) == h for f, h in listed.items())
    # the exploratory analyses are cached until an input or the script changes
    monkeypatch.setattr(rs, "mc_calibration", lambda *a, **k: pytest.fail("recomputed although nothing changed"))
    assert rs.cmd_report(fast, paths) == 0
    assert paths.results_md_plan.read_text(encoding="utf-8") == report
    # a hand-written EXPLORATORY section survives a re-render of test_results.md; RESULTS.md leaves it out
    extra = "\n## EXPLORATORY (not a result)\n\n* a note\n"
    paths.results_md.write_bytes(before + extra.encode("utf-8"))
    assert rs.cmd_report(fast, paths) == 0
    assert paths.results_md.read_bytes() == before + extra.encode("utf-8")
    assert "a note" not in paths.results_md_plan.read_text(encoding="utf-8")


def test_the_test_command_stops_before_unsealing_on_a_changed_panel(tmp_path, world):
    paths, rebuild, token = _synthetic_repo(tmp_path, world)
    p = pd.read_pickle(paths.panel_test)
    p.loc[T("2012-03-30"), "C05"] += 0.01                              # the stored panel no longer matches a rebuild
    p.to_pickle(paths.panel_test)
    real = pd.concat([world[3], world[4]])
    cols = [c for c in real.columns if c not in rs.TARGETS and not c.startswith(("train_h", "test_h")) and c != "in_param_window"]
    with pytest.raises(SystemExit, match="before unsealing: rebuild"):
        rs.cmd_test(SimpleNamespace(unseal=token, after_error=False), paths, lambda *a: real[cols].copy())
    assert not paths.results_json.exists() and not (paths.test_dir / rs.STARTED).exists()
    # an unfinished run blocks a rerun unless --after-error
    paths.test_dir.mkdir()
    with pytest.raises(SystemExit, match="unfinished"):
        rs.cmd_test(SimpleNamespace(unseal=token, after_error=False), paths, rebuild)
