"""Point-in-time helpers of scripts/research_signals.py (PREREGISTRATION.md sections 3, 6 and 13.3).

All offline, on synthetic series: no network and no data files.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_signals as rs  # noqa: E402

T = pd.Timestamp
MAX = T(rs.MAX_DATE)


# ---- synthetic inputs ---------------------------------------------------------------------
def _rows(values: dict, release, revise=None) -> pd.DataFrame:
    """ALFRED-style rows: each observation first released at ``release(date)``; with ``revise``
    (date -> (revision date, new value)) the first row ends the day before the revision."""
    out = []
    for d, v in values.items():
        r1 = release(d)
        if revise and d in revise:
            r2, v2 = revise[d]
            out.append((d, r1, r2 - pd.Timedelta(days=1), v))
            out.append((d, r2, MAX, v2))
        else:
            out.append((d, r1, MAX, v))
    return pd.DataFrame(out, columns=["date", "realtime_start", "realtime_end", "value"])


def _monthly_release(day: int):
    return lambda d: (d.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=day - 1)


def _data(seed: int = 0) -> rs.PredictorData:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("1986-01-01", "1994-12-30")
    n = len(cal)
    gspc = pd.Series(300.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal(n)), index=cal)
    vix = pd.Series(15.0 + 10.0 * rng.random(n), index=cal).drop(cal[1000:1003])   # a 3-session gap
    daily = lambda lvl: pd.Series(lvl + np.cumsum(0.02 * rng.standard_normal(n)), index=cal)  # noqa: E731
    months = pd.date_range("1946-01-01", "1994-11-01", freq="MS")
    ip = {d: float(np.exp(0.002 * i + 0.02 * np.sin(i / 7.0))) for i, d in enumerate(months)}
    rev = {d: (_monthly_release(15)(d) + pd.Timedelta(days=61), v * 1.001) for d, v in ip.items() if d.month == 6}
    cpi_m = pd.date_range("1970-01-01", "1994-11-01", freq="MS")
    cpi = {d: 40.0 * 1.004 ** i for i, d in enumerate(cpi_m)}
    cpi[T("1992-10-01")] = np.nan                      # an unpublished month, as October 2025
    q = pd.date_range("1990-04-01", "1994-10-01", freq="QS")
    sloos = pd.Series(rng.uniform(-30, 60, len(q)), index=q)
    sats = pd.date_range("1984-01-07", "1994-12-31", freq="W-SAT")
    icnsa = pd.Series(300_000 + 50_000 * rng.random(len(sats)), index=sats)
    em = pd.date_range("1975-01-01", "1994-12-01", freq="MS")
    earnings = pd.Series(10.0 + 0.05 * np.arange(len(em)) + rng.random(len(em)), index=em)
    me = [cal[p] for p in rs.br.month_end_positions(cal, "1990-01")]
    legacy = pd.DataFrame({"pos": [cal.get_loc(t) for t in me], "hurdle": rng.uniform(0.03, 0.2, len(me)),
                           "score": rng.uniform(-1, 1, len(me))}, index=pd.DatetimeIndex(me, name="date"))
    return rs.PredictorData(
        cal=cal, gspc=gspc, vix=vix, dgs3mo=daily(5.0).drop(cal[::97]), dgs10=daily(7.0), dbaa=daily(9.0),
        daaa=daily(8.0), indpro=rs.Realtime(_rows(ip, _monthly_release(15), rev)),
        cpi=rs.Realtime(_rows(cpi, _monthly_release(13))), sloos=sloos, sloos_shift=0, icnsa=icnsa,
        earnings=earnings, legacy=legacy)


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.fixture(scope="module")
def origins(data):
    return [p for p in rs.br.month_end_positions(data.cal, "1990-01") if data.cal[p] <= T("1994-10-31")]


@pytest.fixture(scope="module")
def panel(data, origins):
    return rs.compute_panel(data, origins)


# ---- daily market data: values dated on or before t-, at most 5 sessions stale -----------
def test_asof_reads_the_session_before_the_origin_not_the_origin():
    cal = pd.bdate_range("2000-01-03", periods=30)
    s = pd.Series(np.arange(30.0), index=cal)
    assert rs.asof(s, cal, 10) == 10.0
    assert rs.asof(s.drop(cal[10]), cal, 10) == 9.0              # a missing day: the last value before it
    assert math.isnan(rs.asof(s, cal, -1))


def test_asof_staleness_limit_is_five_sessions():
    cal = pd.bdate_range("2000-01-03", periods=30)
    s = pd.Series(np.arange(30.0), index=cal).drop(cal[5:20])      # last value on session 4
    assert rs.asof(s, cal, 9) == 4.0                               # 5 sessions old: kept
    assert math.isnan(rs.asof(s, cal, 10))                         # 6 sessions old: missing
    hol = pd.Series([1.0], index=[T("2000-01-08")])                # dated on a Saturday
    assert rs.asof(hol, cal, cal.get_loc(T("2000-01-10"))) == 1.0  # counts from the last session before it


def test_realised_variance_is_the_21_sessions_ending_at_t_minus():
    rng = np.random.default_rng(1)
    g = 100.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal(60))
    r = 100.0 * np.diff(np.log(g))
    assert rs.realised_var21(g, 40) == pytest.approx(float(np.sum(r[19:40] ** 2)), rel=1e-12)
    g2 = g.copy()
    g2[41:] *= 1.5                                                 # later closes do not matter
    assert rs.realised_var21(g2, 40) == rs.realised_var21(g, 40)
    g2[25] = np.nan
    assert math.isnan(rs.realised_var21(g2, 40))
    assert math.isnan(rs.realised_var21(g, 20))                    # needs 22 closes


# ---- ALFRED vintages ----------------------------------------------------------------------
def test_realtime_vintage_in_effect_on_a_date():
    rows = _rows({T("2000-01-01"): 100.0, T("2000-02-01"): 102.0}, _monthly_release(15),
                 {T("2000-01-01"): (T("2000-03-15"), 101.0)})
    rt = rs.Realtime(rows)
    assert rt.at(T("2000-02-14")).empty
    assert rt.at(T("2000-02-15")).to_dict() == {T("2000-01-01"): 100.0}
    assert rt.at(T("2000-03-14")).to_dict() == {T("2000-01-01"): 100.0}   # Feb data released Mar 15
    assert rt.at(T("2000-03-15")).to_dict() == {T("2000-01-01"): 101.0, T("2000-02-01"): 102.0}
    assert rt.at(T("2000-03-14")).to_dict() == {T("2000-01-01"): 100.0}   # the cache keys on the vintage
    assert rt.truncated(T("2000-03-14")).at(T("2000-03-14")).to_dict() == {T("2000-01-01"): 100.0}
    assert rt.truncated(T("2000-03-14")).rows["realtime_start"].max() <= T("2000-03-14")
    assert rt.first_release().to_dict() == {T("2000-01-01"): T("2000-02-15"), T("2000-02-01"): T("2000-03-15")}


def test_cpi_yoy_uses_the_latest_published_month_and_never_interpolates():
    idx = pd.date_range("1999-01-01", "2000-06-01", freq="MS")
    v = pd.Series(100.0 * 1.01 ** np.arange(len(idx)), index=idx)
    val, m = rs.cpi_yoy(v)
    assert m == T("2000-06-01") and val == pytest.approx(12 * math.log(1.01))
    val, m = rs.cpi_yoy(pd.concat([v, pd.Series([np.nan], index=[T("2000-07-01")])]))
    assert m == T("2000-06-01")                                    # a '.' month is not the latest month
    v2 = v.copy()
    v2[T("1999-06-01")] = np.nan
    val, m = rs.cpi_yoy(v2)
    assert m == T("2000-06-01") and math.isnan(val)                # base missing: no value


def test_inflation_is_carried_forward_when_the_base_month_is_missing(data, panel):
    # the synthetic CPI has no October 1992: m = 1993-10 at the 1993-11 origin, base missing
    row = panel.loc[T("1993-11-30")]
    assert row["cpi_m"] == T("1993-10-01") and bool(row["infl_carried"])
    assert row["C06"] == panel.loc[T("1993-10-29"), "C06"]
    assert not bool(panel.loc[T("1993-12-31"), "infl_carried"])


def test_cpi_and_indpro_use_only_vintages_released_by_the_origin(data, panel):
    # released on the 13th / 15th of the next month: at a month-end, m is the previous month
    for t in (T("1991-03-29"), T("1993-07-30")):
        assert panel.loc[t, "cpi_m"] == rs.add_months(t, -1)
        assert panel.loc[t, "indpro_m"] == rs.add_months(t, -1)


# ---- output gap -----------------------------------------------------------------------------
def test_output_gap_is_the_end_residual_of_a_quadratic_log_trend():
    idx = pd.date_range("1948-01-01", "1990-12-01", freq="MS")
    k = np.arange(len(idx), dtype=float)
    exact = pd.Series(np.exp(4.0 + 0.003 * k - 1e-6 * k ** 2), index=idx)
    gap, m = rs.output_gap(exact)
    assert m == idx[-1] and abs(gap) < 1e-9
    bumped = exact.copy()
    bumped.iloc[-1] *= math.exp(0.05)
    assert rs.output_gap(bumped)[0] > 0.04                         # most of a last-month bump is gap
    early = pd.concat([pd.Series(1.0, index=pd.date_range("1919-01-01", "1947-12-01", freq="MS")), bumped])
    assert rs.output_gap(early)[0] == rs.output_gap(bumped)[0]     # the fit starts in 1948-01
    late = bumped.loc["1960-01-01":]
    kk = np.arange(len(late), dtype=float)
    X = np.column_stack([np.ones_like(kk), kk, kk ** 2])
    beta = np.linalg.lstsq(X, np.log(late.values), rcond=None)[0]
    assert rs.output_gap(late)[0] == pytest.approx(float(np.log(late.values[-1]) - X[-1] @ beta), abs=1e-10)


# ---- SLOOS usable date and QA ------------------------------------------------------------
def test_sloos_observation_usable_from_last_session_of_first_month_after_quarter():
    cal = pd.bdate_range("1990-01-01", "1991-12-31")
    u = rs.sloos_usable_dates([T("1990-04-01"), T("1990-01-01"), T("1990-10-01")], cal)
    assert u[T("1990-04-01")] == T("1990-07-31")                   # the plan's example
    assert u[T("1990-01-01")] == T("1990-04-30")
    assert u[T("1990-10-01")] == T("1991-01-31")
    assert rs.sloos_usable_dates([T("1990-04-01")], cal, shift=1)[T("1990-04-01")] == T("1990-08-31")
    cal2 = cal.drop(T("1990-07-31"))                               # last session of July is the 30th
    assert rs.sloos_usable_dates([T("1990-04-01")], cal2)[T("1990-04-01")] == T("1990-07-30")
    vals = pd.Series([10.0, 20.0], index=[T("1990-01-01"), T("1990-04-01")])
    use = rs.sloos_usable_dates(vals.index, cal)
    assert rs.sloos_at(vals, use, T("1990-07-30")) == (10.0, T("1990-01-01"))
    assert rs.sloos_at(vals, use, T("1990-07-31")) == (20.0, T("1990-04-01"))
    assert math.isnan(rs.sloos_at(vals, use, T("1990-04-27"))[0])


def test_sloos_qa_flags_a_release_after_the_usable_date():
    cal = pd.bdate_range("2010-01-01", "2011-12-31")
    use = rs.sloos_usable_dates([T("2010-04-01"), T("2010-07-01")], cal)
    assert list(use) == [T("2010-07-30"), T("2010-10-29")]
    ok = pd.Series([T("2010-05-03"), T("2010-08-16")], index=use.index)
    qa = rs.sloos_qa(ok, use)
    assert qa["n_checked"] == 2 and qa["n_violations"] == 0
    early = pd.Series([T("2010-04-30"), T("2010-08-31")], index=use.index)   # the first fires too soon
    bad = rs.sloos_qa(pd.Series([T("2010-05-03"), T("2010-08-16")], index=use.index), early)
    assert bad["n_checked"] == 2 and bad["n_violations"] == 1 and bad["violations"][0]["obs"] == "2010-04-01"
    old = pd.Series([T("2010-05-03"), T("2011-03-01")], index=use.index)       # > 120 days: not checked
    assert rs.sloos_qa(old, early)["n_checked"] == 1


# ---- claims: the Thursday release rule ------------------------------------------------------
@pytest.mark.parametrize("origin, week", [
    ("2020-01-30", "2020-01-25"),     # Thursday: that morning's release (week ending the Saturday before)
    ("2020-01-29", "2020-01-18"),     # Wednesday: the previous Thursday's release
    ("2020-01-31", "2020-01-25"),     # Friday
    ("2020-02-01", "2020-01-25"),     # Saturday: that week is not out until Thursday
    ("2020-02-06", "2020-02-01"),
])
def test_claims_week_is_the_latest_released(origin, week):
    w = rs.claims_week(T(origin))
    assert w == T(week) and w.weekday() == 5 and w + pd.Timedelta(days=5) <= T(origin)


def test_claims_yoy_definition():
    sats = pd.date_range("2018-01-06", "2020-03-28", freq="W-SAT")
    s = pd.Series(np.arange(1.0, len(sats) + 1.0) * 1000.0, index=sats)
    val, w = rs.claims_yoy(s, T("2020-01-31"))
    now = s.loc[[w - pd.Timedelta(weeks=j) for j in range(4)]].sum()
    ago = s.loc[[w - pd.Timedelta(weeks=j) for j in range(52, 56)]].sum()
    assert val == pytest.approx(math.log(now) - math.log(ago), rel=1e-12)
    later = s.copy()
    later[later.index > w] *= 10.0                                 # weeks not yet released do not matter
    assert rs.claims_yoy(later, T("2020-01-31"))[0] == val
    assert math.isnan(rs.claims_yoy(s.drop(w - pd.Timedelta(weeks=53)), T("2020-01-31"))[0])


# ---- excess CAPE yield: earnings of the last reported quarter only ---------------------------
def test_ecy_reads_earnings_only_through_the_last_reported_quarter():
    months = pd.date_range("1975-01-01", "2001-12-01", freq="MS")
    e = pd.Series(10.0 + 0.1 * np.arange(len(months)), index=months)
    cpi = pd.Series(50.0 * 1.003 ** np.arange(len(months)), index=months).loc[:"2001-05-01"]
    t = T("2001-06-29")
    q = rs.br.reported_quarter(t)
    assert q == T("2001-03-01")
    out = rs.excess_cape_yield(t, 1200.0, 5.5, cpi, e)
    ks = pd.date_range(rs.add_months(q, -119), q, freq="MS")
    cape = (1200.0 / cpi.iloc[-1]) / float((e[ks] / cpi[ks]).mean())
    rr10 = 0.055 - ((cpi.iloc[-1] / cpi[rs.add_months(cpi.index[-1], -120)]) ** 0.1 - 1.0)
    assert out["C09"] == pytest.approx(1.0 / cape - rr10, rel=1e-12) and out["ecy_months"] == 120
    e2 = e.copy()
    e2[e2.index > q] = 999.0                                       # interpolated months after q: unused
    assert rs.excess_cape_yield(t, 1200.0, 5.5, cpi, e2)["C09"] == out["C09"]
    e3 = e.copy()
    e3[q] = 999.0
    assert rs.excess_cape_yield(t, 1200.0, 5.5, cpi, e3)["C09"] != out["C09"]
    assert math.isnan(rs.excess_cape_yield(t, 1200.0, 5.5, cpi, e.loc[:"2000-12-01"])["C09"])   # q not out


def test_ecy_skips_an_unpublished_cpi_month_without_interpolating():
    months = pd.date_range("1975-01-01", "2001-12-01", freq="MS")
    e = pd.Series(10.0 + 0.1 * np.arange(len(months)), index=months)
    cpi = pd.Series(50.0 * 1.003 ** np.arange(len(months)), index=months).loc[:"2001-05-01"]
    t = T("2001-06-29")
    gap = cpi.copy()
    gap[T("1999-10-01")] = np.nan
    out = rs.excess_cape_yield(t, 1200.0, 5.5, gap, e)
    q = rs.br.reported_quarter(t)
    ks = [k for k in pd.date_range(rs.add_months(q, -119), q, freq="MS") if k != T("1999-10-01")]
    cape = (1200.0 / cpi.iloc[-1]) / float((e[ks] / cpi[ks]).mean())
    assert out["ecy_months"] == 119 and out["cape"] == pytest.approx(cape, rel=1e-12)
    holes = cpi.copy()
    holes[pd.date_range("1995-01-01", "1995-07-01", freq="MS")] = np.nan      # 7 missing: below 114
    assert math.isnan(rs.excess_cape_yield(t, 1200.0, 5.5, holes, e)["C09"])


# ---- the panel is point in time ------------------------------------------------------------
CANDS = [f"C{i:02d}" for i in range(1, 13)]


def test_panel_candidates_are_defined(panel):
    late = panel.loc[panel.index >= T("1991-01-01"), CANDS]
    assert late.notna().all().all()
    assert math.isnan(panel.loc[T("1990-06-29"), "C07"]) and panel.loc[T("1990-07-31"), "sloos_obs"] == T("1990-04-01")
    row = panel.loc[T("1992-05-29")]
    assert row["C01"] == pytest.approx(row["vix_tm"] ** 2 / 12.0 - row["rv21_tm"])
    assert row["C10"] == pytest.approx((row["vix_tm"] / 100.0) ** 2)
    assert row["C04"] == pytest.approx(row["y10_tm"] - row["y3_tm"])
    assert row["C03"] == pytest.approx(row["y3_tm"] - row["y3_tprev"])
    assert row["C05"] == pytest.approx(row["baa_tm"] - row["aaa_tm"])
    assert row["t_minus"] == T("1992-05-28")


@pytest.mark.parametrize("k", [0, 7, 19, 33, 46])
def test_truncation_invariance(data, origins, panel, k):
    """Section 13.3: deleting everything the origin could not know leaves its row unchanged."""
    pos = origins[k]
    sub = [p for p in origins if p <= pos]
    got = rs.compute_panel(rs.truncate(data, pos), sub).iloc[-1]
    want = panel.loc[data.cal[pos]]
    for c in panel.columns:
        assert rs._same(want[c], got[c]), c


def test_changing_the_future_leaves_an_origin_unchanged(data, origins, panel):
    """An independent leak test: scramble every value dated at or after the origin (market data, so
    the origin's own close too), every vintage released after it, claims weeks not yet released,
    earnings after the reported quarter and backtest rows after it."""
    pos = origins[30]
    t = data.cal[pos]
    rng = np.random.default_rng(9)
    bump = lambda s, m: s.where(~m, s * (1.5 + rng.random(len(s))))  # noqa: E731

    def rt_bump(rt):
        rows = rt.rows.copy()
        m = rows["realtime_start"] > t
        rows.loc[m, "value"] = rows.loc[m, "value"] * 1.37
        return rs.Realtime(rows)

    q = rs.br.reported_quarter(t)
    usable = rs.sloos_usable_dates(data.sloos.index, data.cal)
    fut = rs.PredictorData(
        cal=data.cal, gspc=bump(data.gspc, data.gspc.index >= t), vix=bump(data.vix, data.vix.index >= t),
        dgs3mo=bump(data.dgs3mo, data.dgs3mo.index >= t), dgs10=bump(data.dgs10, data.dgs10.index >= t),
        dbaa=bump(data.dbaa, data.dbaa.index >= t), daaa=bump(data.daaa, data.daaa.index >= t),
        indpro=rt_bump(data.indpro), cpi=rt_bump(data.cpi),
        sloos=bump(data.sloos, (usable > t).reindex(data.sloos.index).values), sloos_shift=0,
        icnsa=bump(data.icnsa, data.icnsa.index + pd.Timedelta(days=5) > t),
        earnings=bump(data.earnings, data.earnings.index > q),
        legacy=data.legacy.assign(hurdle=bump(data.legacy["hurdle"], data.legacy.index > t),
                                  score=bump(data.legacy["score"], data.legacy.index > t)))
    got = rs.compute_panel(fut, [pos]).iloc[0]
    for c in CANDS:
        assert rs._same(panel.loc[t, c], got[c]), c
    # and the checks have teeth: the close at t- does move C01 and C09
    moved = rs.dataclasses.replace(data, gspc=bump(data.gspc, data.gspc.index == data.cal[pos - 1]))
    row = rs.compute_panel(moved, [pos]).iloc[0]
    assert row["C01"] != panel.loc[t, "C01"] and row["C09"] != panel.loc[t, "C09"]


# ---- targets and split ---------------------------------------------------------------------
def test_targets_match_the_backtest_definitions():
    cal = pd.bdate_range("2000-01-03", periods=300)
    rng = np.random.default_rng(3)
    fund = np.cumprod(1.0 + 0.02 * rng.standard_normal(300))
    tr = np.cumprod(1.0 + 0.006 * rng.standard_normal(300))
    tb = np.full(300, 0.05 / 252)
    d = pd.DataFrame({"fund": fund, "tr": tr, "tbill_ret": tb}, index=cal)
    out = rs.compute_targets(d, [10, 200], cal)
    rb = float(np.prod(1.0 + tb[11:137]) - 1.0)
    assert out.loc[cal[10], "Y126"] == pytest.approx(fund[136] / fund[10] - 1.0 - rb, rel=1e-12)
    assert out.loc[cal[10], "S126"] == pytest.approx(tr[136] / tr[10] - 1.0 - rb, rel=1e-12)
    assert out.loc[cal[10], "D20"] == float((fund[11:74] / fund[10]).min() <= 0.8)
    assert math.isnan(out.loc[cal[200], "Y126"]) and not math.isnan(out.loc[cal[200], "Y63"])
    flat = d.assign(fund=1.0)
    flat.iloc[50, flat.columns.get_loc("fund")] = 0.8                           # exactly 20% down counts
    assert rs.compute_targets(flat, [10], cal).loc[cal[10], "D20"] == 1.0
    flat.iloc[50, flat.columns.get_loc("fund")] = 0.81
    assert rs.compute_targets(flat, [10], cal).loc[cal[10], "D20"] == 0.0
    flat.iloc[74, flat.columns.get_loc("fund")] = 0.5                            # session t+64: outside
    assert rs.compute_targets(flat, [10], cal).loc[cal[10], "D20"] == 0.0


def test_split_flags_follow_the_window_end():
    cal = pd.bdate_range("2006-01-02", "2009-12-31")
    pos = rs.br.month_end_positions(cal, "2006-01")
    f = rs.split_flags(cal, pos)
    for p, t in zip(pos, f.index):
        for h in rs.HORIZONS:
            ends = cal[p + h] if p + h < len(cal) else None
            assert f.at[t, f"train_h{h}"] == bool(ends is not None and ends <= rs.SEAL_END)
            assert f.at[t, f"test_h{h}"] == bool(t >= rs.TEST_START and ends is not None and ends <= rs.DATA_CUTOFF)
    assert not (f.filter(like="train_") & f.filter(like="test_").values).any().any()


# ---- point-in-time audit fixes (research/DEVIATIONS.md D3-D5) --------------------------------
def _audited(data: rs.PredictorData) -> rs.PredictorData:
    """The synthetic data with the audit's three point-in-time inputs: SLOOS vintages whose first
    recorded vintage (1992-01-15, standing in for ALFRED's 2010-04-20) is later than the early
    observations and which revise the 1990-1991 values in 1993; a claims release calendar with a
    shutdown (weeks 1993-09-25..1993-11-06 published 1993-11-18) and a holiday Wednesday; and one
    origin whose legacy score used an input not yet published."""
    first = T("1992-01-15")
    rel = lambda d: max(first, d + pd.Timedelta(days=35))  # noqa: E731
    revise = {d: (T("1993-06-01"), v + 5.0) for d, v in data.sloos.items() if d <= T("1991-01-01")}
    sloos_rt = rs.Realtime(_rows(data.sloos.to_dict(), rel, revise))
    weeks = data.icnsa.index
    fr = pd.Series(weeks + pd.Timedelta(days=5), index=weeks)
    fr[fr.index <= T("1991-01-05")] = T("1991-01-10")                     # the recorder's first vintage
    shut = (fr.index >= T("1993-09-25")) & (fr.index <= T("1993-11-06"))
    fr[shut] = T("1993-11-18")
    fr[T("1992-11-21")] = T("1992-11-25")                                  # a Wednesday before a holiday
    blocked = pd.Series({T("1992-01-31"): "CPI 1991-12 first published 1992-02-01"}, dtype=object)
    return rs.dataclasses.replace(data, sloos_rt=sloos_rt, icnsa_release=rs.claims_release_dates(weeks, fr),
                                  legacy_blocked=blocked)


@pytest.fixture(scope="module")
def audited(data):
    return _audited(data)


@pytest.fixture(scope="module")
def audited_panel(audited, origins):
    return rs.compute_panel(audited, origins)


def test_sloos_uses_the_value_published_at_t_not_a_later_revision():
    rows = _rows({T("1990-04-01"): 56.9, T("2012-04-01"): 10.0},
                 lambda d: max(T("2010-04-20"), d + pd.Timedelta(days=30)),
                 {T("1990-04-01"): (T("2019-11-04"), 54.4), T("2012-04-01"): (T("2019-11-04"), 12.0)})
    rt = rs.Realtime(rows)
    assert rs.sloos_values(rt, T("1990-07-31"))[T("1990-04-01")] == 56.9     # before ALFRED: first vintage
    assert rs.sloos_values(rt, T("2019-11-01"))[T("1990-04-01")] == 56.9
    assert rs.sloos_values(rt, T("2019-11-04"))[T("1990-04-01")] == 54.4
    assert rs.sloos_values(rt, T("2012-07-31"))[T("2012-04-01")] == 10.0
    assert rs.sloos_values(rt, T("2020-01-31"))[T("2012-04-01")] == 12.0


def test_sloos_panel_value_ignores_revisions_after_the_origin(audited, audited_panel, panel):
    t = T("1990-10-31")                                   # obs 1990-07-01, revised only in 1993
    assert audited_panel.loc[t, "sloos_obs"] == T("1990-07-01")
    assert audited_panel.loc[t, "C07"] == audited.sloos[T("1990-07-01")]
    t2 = T("1994-01-31")                                  # obs 1993-10-01, never revised
    assert audited_panel.loc[t2, "sloos_obs"] == T("1993-10-01")
    assert audited_panel.loc[t2, "C07"] == panel.loc[t2, "C07"]
    moved = rs.dataclasses.replace(audited, sloos=audited.sloos + 5.0)    # cutoff-vintage values are unused
    assert rs.compute_panel(moved, [audited.cal.get_loc(t)]).iloc[0]["C07"] == audited_panel.loc[t, "C07"]


def test_claims_release_dates_keep_the_thursday_rule_unless_published_later():
    weeks = pd.date_range("2009-05-02", "2009-06-13", freq="W-SAT")
    fr = pd.Series(weeks + pd.Timedelta(days=5), index=weeks)
    fr[weeks <= T("2009-05-23")] = T("2009-05-28")      # first seen at the recorder's first vintage
    fr[T("2009-06-06")] = T("2009-06-10")               # a Wednesday: earlier than the rule
    fr[T("2009-06-13")] = T("2009-06-30")               # published late
    rel = rs.claims_release_dates(weeks, fr)
    assert rel[T("2009-05-02")] == T("2009-05-07")      # the first vintage is not a release date
    assert rel[T("2009-06-06")] == T("2009-06-11")      # never earlier than the registered rule
    assert rel[T("2009-06-13")] == T("2009-06-30")
    assert rs.claims_week(T("2009-06-26"), rel) == T("2009-06-06")
    assert rs.claims_week(T("2009-06-26")) == T("2009-06-20")                # the rule alone
    assert rs.claims_week(T("2009-06-30"), rel) == T("2009-06-13")


def test_claims_skip_weeks_not_yet_published_in_a_shutdown(audited, audited_panel, panel):
    t = T("1993-10-29")
    assert panel.loc[t, "claims_week"] == T("1993-10-23")                    # the Thursday rule alone
    assert audited_panel.loc[t, "claims_week"] == T("1993-09-18")            # last week published by t
    s = audited.icnsa
    w = T("1993-09-18")
    want = math.log(s[[w - pd.Timedelta(weeks=j) for j in range(4)]].sum()) \
        - math.log(s[[w - pd.Timedelta(weeks=j) for j in range(52, 56)]].sum())
    assert audited_panel.loc[t, "C08"] == pytest.approx(want, rel=1e-12)
    later = T("1993-11-30")
    assert audited_panel.loc[later, "claims_week"] == panel.loc[later, "claims_week"]
    for t, w in audited_panel["claims_week"].items():                        # no week before publication
        for j in range(4):
            assert audited.icnsa_release[w - pd.Timedelta(weeks=j)] <= t


def test_legacy_score_is_missing_where_its_inputs_were_not_yet_published(audited_panel, panel):
    t = T("1992-01-31")
    assert math.isnan(audited_panel.loc[t, "C12"]) and bool(audited_panel.loc[t, "c12_unreleased_input"])
    assert audited_panel.loc[t, "C11"] == panel.loc[t, "C11"]                # the hurdle uses neither input
    other = audited_panel.index != t
    assert not audited_panel.loc[other, "c12_unreleased_input"].any()
    assert (audited_panel.loc[other, "C12"] == panel.loc[other, "C12"]).all()


def test_legacy_unreleased_inputs_follows_the_backtest_release_rule():
    months = pd.date_range("1994-01-01", "1996-03-01", freq="MS")
    cpi = pd.Series(100.0 + np.arange(len(months)), index=months)
    un = pd.Series(5.0, index=months)
    un[T("1995-10-01")] = np.nan                                             # never published
    rel = lambda d: (d.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=14)  # noqa: E731
    fr_cpi = pd.Series([rel(d) for d in months], index=months)
    fr_cpi[T("1995-12-01")] = T("1996-02-01")                                # the 1996 shutdown case
    fr_un = pd.Series([rel(d) for d in months], index=months)
    fr_un[T("1995-09-01")] = T("1995-11-20")                                 # published after 1995-10-31
    origins = [T("1995-10-31"), T("1995-11-30"), T("1996-01-31"), T("1996-02-29")]
    got = rs.legacy_unreleased_inputs(origins, {"CPI": cpi, "UNRATE": un}, {"CPI": fr_cpi, "UNRATE": fr_un})
    assert list(got.index) == [T("1995-10-31"), T("1996-01-31")]
    assert "UNRATE 1995-09" in got[T("1995-10-31")] and "CPI 1995-12" in got[T("1996-01-31")]


def test_first_published_ignores_placeholder_rows():
    rows = pd.DataFrame({"date": [T("2025-10-01"), T("2025-09-01")],
                         "realtime_start": [T("2025-12-18"), T("2025-10-24")],
                         "realtime_end": [MAX, MAX], "value": [np.nan, 1.0]})
    assert rs.first_published(rows).to_dict() == {T("2025-09-01"): T("2025-10-24")}


@pytest.mark.parametrize("k", [0, 6, 23, 45, 46])
def test_truncation_invariance_with_the_audit_inputs(audited, origins, audited_panel, k):
    pos = origins[k]
    sub = [p for p in origins if p <= pos]
    got = rs.compute_panel(rs.truncate(audited, pos), sub).iloc[-1]
    want = audited_panel.loc[audited.cal[pos]]
    for c in audited_panel.columns:
        assert rs._same(want[c], got[c]), c
