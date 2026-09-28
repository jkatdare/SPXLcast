"""Historical backtest of the full model, rating included, at every month-end since 1990.

At each month-end the live model's own functions (expected_index_return, vol_term_structure,
simulate, rate) are fed only what was known that day:

* S&P 500 trailing earnings and dividends from Shiller's data at the last quarter-end already
  reported (two months after it ends), over the S&P 500 close. Shiller interpolates the months
  between quarter-ends, so reading an in-between month would leak part of an unreported quarter.
* expected inflation: the 10-year breakeven (FRED T10YIE) from 2003; before that, trailing 10-year
  CPI inflation (the proxy the drift backtest uses)
* regime inputs from FRED: 3-month bill and 10-year yields (curve), CPI and unemployment (only
  months already released; CPI YoY and the Sahm gap by the live model's own functions), and the
  high-yield spread, which FRED serves for the last three years only: the credit-stress penalty
  cannot fire before then, and the report says for how many month-ends
* VIX, VIX3M and VIX6M from Yahoo. VIX3M/VIX6M start in 2008; before that they are imputed from the
  VIX with a log-linear fit on 2008-2026, the only use of later data (``--term-structure flat``
  uses the live model's own fallback instead: a flat curve at the VIX)

The realised outcome is SPXL itself from 2009. Before that it is a synthetic 3x fund: the daily
S&P 500 total return times three, less the costs the model charges (expense ratio plus financing
of the borrowed 2x at the 3-month bill plus the spread), checked against SPXL where both exist.

Reported:
1. Does the rating carry information? Forward 6-month SPXL returns by label and by score quintile,
   and the rank correlation of score with forward return, with overlap-aware 90% intervals
   (closed form on the number of independent outcome windows, as the live scorer's).
2. Where do the thresholds belong? BUY/SELL threshold sweep, full sample and both halves.
3. What following the rating would have done (illustration only): monthly rules vs buy-and-hold.
4. Calibration of the full model at 1W to 6M: PIT, band coverage, 20% drawdown odds.
5. Skill (CRPS, lower is better) against the same engine with a constant 7% drift, which isolates
   the fundamentals, and against a naive lognormal at the raw VIX with a T-bill drift, which
   isolates everything.
6. Leverage cost and drawdown risk, the assessment the page shows (spxlcast/assess.py, computed
   here by the same functions): their tercile cutoffs, how often a 20% dip within 3 months followed
   at each drawdown-risk level, the predicted fund cost plus volatility drag against the realised
   (3 x the index's log return minus the fund's, per year over the next 6 months), and, for
   information only, forward returns by leverage-cost level. These go to reference.json, which
   ``--write-reference`` also copies into the package for the live page.

``--engine-grid FILE`` compares engine settings instead (research/CALIBRATION.md): for each setting
in the JSON file it runs only the model's simulation at each month-end (no rating, no constant-drift
twin) and reports, per horizon, the Brier score of the 20% dip, CRPS and band coverage, the
pre-registered guardrails and the setting the rule picks. It writes engine_<tag>.txt and .csv and
leaves the backtest's own outputs alone.

Usage:  py scripts/backtest_rating.py [--paths 20000] [--start 1990-01] [--end 2026-08] [--term-structure impute]
        [--refresh]   (re-download Yahoo, FRED and Shiller data; otherwise a local cache is used)
        [--write-reference]   (copy output/backtest/reference.json to spxlcast/reference.json)
        [--engine-grid research/calibration_grid.json [--only NAME ...] [--tag train]]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_drift import load_shiller                                         # noqa: E402
from spxlcast.assess import (DIP_HORIZON, DRAWDOWN_LEVELS, LEVERAGE_LEVELS, REFERENCE_PATH,  # noqa: E402
                             leverage_cost, level_of, sigma_1y)
from spxlcast.config import Config                                             # noqa: E402
from spxlcast.data import MarketSnapshot, _fred_api                            # noqa: E402
from spxlcast.env import fred_api_key                                          # noqa: E402
from spxlcast.evaluation import (crps_normal, crps_sample, effective_n, mean_interval,  # noqa: E402
                                 naive_leveraged_lognormal, proportion_interval, skill_interval,
                                 spearman)
from spxlcast.fundamentals import (IndexFundamentals, MacroState, _calendar_months, cpi_yoy_from,  # noqa: E402
                                   expected_index_return, sahm_gap_from, vol_term_structure)
from spxlcast.montecarlo import simulate                                       # noqa: E402
from spxlcast.rating import rate                                               # noqa: E402

HORIZONS = [5, 10, 21, 63, 126]
RATING_H = 126
YAHOO = ["^SP500TR", "^GSPC", "^VIX", "^VIX3M", "^VIX6M", "SPXL"]
FRED = ["DGS3MO", "DGS10", "T10YIE", "BAMLH0A0HYM2", "CPIAUCSL", "UNRATE"]
EARNINGS_LAG_MONTHS = 2          # S&P reports a quarter's earnings about two months after it ends
SPXL_FROM = pd.Timestamp("2009-01-01")   # SPXL launched Nov 2008; synthetic fund before this
TRACKING_SD = 0.0008             # daily tracking noise, as calibrated on SPXL vs SPY
CONSTANT_DRIFT = 0.07
LABELS = ("BUY", "HOLD", "SELL")


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def load_inputs(out_dir: Path, refresh: bool) -> Dict:
    cache = out_dir / "rating_inputs.pkl"
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    import yfinance as yf
    key = fred_api_key()
    if not key:
        raise SystemExit("FRED_API_KEY is needed (put it in .env); the backtest reads 35 years of FRED data")
    raw = yf.download(YAHOO, period="max", group_by="ticker", auto_adjust=True, progress=False, threads=True)
    closes = pd.DataFrame({t: raw[t]["Close"] for t in YAHOO})
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes.index = closes.index.normalize()
    fred = {}
    for sid in FRED:
        s = _fred_api(sid, 60.0, key, years=50)
        if s is None:
            raise SystemExit(f"FRED returned nothing for {sid}")
        fred[sid] = s
    data = {"closes": closes, "fred": fred, "shiller": load_shiller(out_dir / "ie_data.xls", refresh),
            "fetched": datetime.now(timezone.utc).isoformat()}
    pd.to_pickle(data, cache)
    return data


def _daily(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """A FRED daily series on the trading calendar, carried forward over its own holidays."""
    s = s.sort_index()
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def build_daily(inputs: Dict, cfg: Config) -> pd.DataFrame:
    c = inputs["closes"]
    tr = c["^SP500TR"].dropna()
    d = pd.DataFrame(index=tr.index)
    d["tr"] = tr
    d["gspc"] = c["^GSPC"].reindex(d.index).ffill()
    for col, t in (("vix", "^VIX"), ("vix3m", "^VIX3M"), ("vix6m", "^VIX6M"), ("spxl", "SPXL")):
        d[col] = c[t].reindex(d.index)
    d["vix"] = d["vix"].ffill(limit=5)
    f = inputs["fred"]
    d["rf"] = _daily(f["DGS3MO"], d.index) / 100.0
    d["y10"] = _daily(f["DGS10"], d.index) / 100.0
    first_bei, first_hy = f["T10YIE"].index.min(), f["BAMLH0A0HYM2"].index.min()
    d["bei"] = (_daily(f["T10YIE"], d.index) / 100.0).where(d.index >= first_bei)
    d["hy"] = (_daily(f["BAMLH0A0HYM2"], d.index) / 100.0).where(d.index >= first_hy)

    lev = cfg.leverage_target
    rf_prev = d["rf"].shift(1)
    d["tbill_ret"] = rf_prev / 252.0
    d["syn_ret"] = lev * d["tr"].pct_change() - (cfg.expense_ratio_default + (lev - 1) * (rf_prev + cfg.swap_spread)) / 252.0
    d["spxl_ret"] = d["spxl"].pct_change()
    # the realised fund: synthetic before SPXL_FROM, SPXL after (one continuous daily series)
    d["fund_ret"] = d["syn_ret"].where(d.index < SPXL_FROM, d["spxl_ret"])
    d["fund"] = (1.0 + d["fund_ret"].fillna(0.0)).cumprod()
    d["syn"] = (1.0 + d["syn_ret"].fillna(0.0)).cumprod()
    return d


def fit_term_structure(d: pd.DataFrame) -> Dict[str, tuple]:
    """log(VIX3M) and log(VIX6M) on log(VIX), fitted where all three exist (2008 onward)."""
    m = d[["vix", "vix3m", "vix6m"]].dropna()
    fits = {}
    for col in ("vix3m", "vix6m"):
        x, y = np.log(m["vix"].values), np.log(m[col].values)
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        fits[col] = (float(a), float(b), float(1.0 - resid.var() / y.var()), len(m))
    return fits


def validate_synthetic(d: pd.DataFrame) -> Dict[str, float]:
    m = d.loc[d.index >= SPXL_FROM, ["syn_ret", "spxl_ret"]].dropna()
    years = len(m) / 252.0
    g_syn, g_act = np.log1p(m["syn_ret"]).sum() / years, np.log1p(m["spxl_ret"]).sum() / years
    return {"days": len(m), "corr": float(m.corr().iloc[0, 1]), "syn_growth": float(np.expm1(g_syn)),
            "spxl_growth": float(np.expm1(g_act)), "gap_per_year": float(g_syn - g_act),
            "tracking_sd_annual": float((m["syn_ret"] - m["spxl_ret"]).std() * np.sqrt(252))}


# ---------------------------------------------------------------------------------------
# Point-in-time model inputs
# ---------------------------------------------------------------------------------------
def month_end_positions(idx: pd.DatetimeIndex, start: str, end: Optional[str] = None) -> List[int]:
    """Positions of the complete months' last sessions from month ``start`` through month ``end``."""
    s = pd.Series(np.arange(len(idx)), index=idx)
    last = s.groupby([idx.year, idx.month]).max()
    current = (idx[-1].year, idx[-1].month)          # the data's last month is still in progress
    stop = pd.Period(end, "M") if end else None
    return [int(p) for p in last.values if idx[p] >= pd.Timestamp(start) and (idx[p].year, idx[p].month) != current
            and (stop is None or idx[p].to_period("M") <= stop)]


def released(series: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """Monthly observations already published at a month-end: up to the previous month."""
    cutoff = (origin.to_period("M") - 1).to_timestamp()
    return series[series.index <= cutoff].dropna()


def reported_quarter(origin: pd.Timestamp) -> pd.Timestamp:
    """The month of the last quarter-end whose earnings are out at a month-end origin."""
    m = origin.to_period("M") - EARNINGS_LAG_MONTHS
    return (m - m.month % 3).to_timestamp()


def inputs_at(d: pd.DataFrame, pos: int, inputs: Dict, cfg: Config, ts_mode: str, fits: Dict) -> Optional[Dict]:
    row = d.iloc[pos]
    origin = d.index[pos]
    sh = inputs["shiller"]
    q = reported_quarter(origin)
    if q not in sh.index or not np.isfinite(row["vix"]) or not np.isfinite(row["rf"]):
        return None
    E, D, P = float(sh.at[q, "E"]), float(sh.at[q, "D"]), float(row["gspc"])
    cpi = released(inputs["fred"]["CPIAUCSL"], origin)
    un = released(inputs["fred"]["UNRATE"], origin)
    bei = row["bei"] if np.isfinite(row["bei"]) else None
    if len(un) < 15:
        return None
    infl = bei
    if bei is None:     # before the breakeven series: trailing 10-year CPI, by calendar month as for the YoY
        months = _calendar_months(cpi)
        base = months.get(months.index[-1] - 120) if months is not None else None
        if base is None or not np.isfinite(base):
            return None
        infl = float((months.iloc[-1] / base) ** 0.1 - 1.0)

    vix3m = row["vix3m"] if np.isfinite(row["vix3m"]) else None
    vix6m = row["vix6m"] if np.isfinite(row["vix6m"]) else None
    imputed = False
    if ts_mode == "impute" and (vix3m is None or vix6m is None):
        a3, b3 = fits["vix3m"][:2]
        a6, b6 = fits["vix6m"][:2]
        vix3m = vix3m or float(np.exp(a3 + b3 * np.log(row["vix"])))
        vix6m = vix6m or float(np.exp(a6 + b6 * np.log(row["vix"])))
        imputed = True

    macro = MacroState(
        rf_3m=float(row["rf"]), y2=None, y5=None, y10=float(row["y10"]), y30=None,
        curve_10y_3m=float(row["y10"] - row["rf"]), breakeven_10y=infl, real_10y=None, sofr=None,
        hy_oas=float(row["hy"]) if np.isfinite(row["hy"]) else None,
        unemployment=float(un.iloc[-1]) / 100.0, unemployment_sahm_gap=sahm_gap_from(un),
        cpi_yoy=cpi_yoy_from(cpi), vix=float(row["vix"]), vix3m=vix3m, vix6m=vix6m,
        vvix=None, skew=None, dxy=None, oil=None, gold=None)
    fund = IndexFundamentals(index_level=P, trailing_pe=P / E, earnings_yield=E / P, dividend_yield=D / P,
                             eps_growth=cfg.long_run_real_eps_growth + infl, book_to_price=None, sales_to_price=None)
    return {"macro": macro, "fund": fund, "infl_source": "breakeven" if bei is not None else "trailing CPI",
            "ts_imputed": imputed}


# ---------------------------------------------------------------------------------------
# One origin
# ---------------------------------------------------------------------------------------
def _model_sim(inp: Dict, cfg: Config):
    """The live model's simulation at one origin: (expected return, vol term structure, financing,
    annual cost, simulator arguments without the drift, simulation)."""
    macro, fund = inp["macro"], inp["fund"]
    expected = expected_index_return(fund, macro, cfg)
    T = max(HORIZONS)
    vol = vol_term_structure(macro, MarketSnapshot(asof=datetime.now(timezone.utc)), cfg, T)
    financing = (cfg.leverage_target - 1) * (macro.rf_3m + cfg.swap_spread)
    annual_cost = cfg.expense_ratio_default + financing
    kw = dict(spot=1.0, sigma_annual=vol.daily, leverage=cfg.leverage_target, daily_cost=annual_cost / 252.0,
              tracking_sd_daily=TRACKING_SD, rf_annual=macro.rf_3m, horizons=HORIZONS, n_paths=cfg.n_paths,
              dof=cfg.t_dof, skew_gamma=cfg.skew_gamma, max_daily_move=cfg.max_daily_move, seed=cfg.seed,
              drift_sd_annual=cfg.drift_uncertainty_sd, sv_persistence=cfg.sv_persistence,
              sv_logvol_sd=cfg.sv_logvol_sd, sv_leverage=cfg.sv_leverage)
    return expected, vol, financing, annual_cost, kw, simulate(mu_annual=np.full(T, expected.final), **kw)


def _fund_window(d: pd.DataFrame, pos: int, h: int) -> np.ndarray:
    """The realised fund's closes over the h sessions after ``pos``, relative to its close at ``pos``."""
    fund_level = d["fund"].values
    return fund_level[pos + 1: pos + h + 1] / fund_level[pos]


def _model_scores(sim, window: np.ndarray, h: int) -> Dict:
    """Where the outcome fell in the model's simulation at h: PIT, band hits, CRPS of the log return,
    and the predicted and realised 20% dip (lowest close at most 0.8 x the start)."""
    y = float(np.log(window[-1]))
    x = np.log(sim.terminal[h])
    q5, q25, q75, q95 = np.percentile(x, [5, 25, 75, 95])
    return {f"pit_{h}": float(np.mean(x <= y)), f"in90_{h}": bool(q5 <= y <= q95), f"in50_{h}": bool(q25 <= y <= q75),
            f"crps_model_{h}": crps_sample(x, y),
            f"pred_dd20_{h}": float(np.mean(sim.path_min[h] <= 0.8)), f"real_dd20_{h}": float(window.min() <= 0.8)}


def run_origin(d: pd.DataFrame, pos: int, inp: Dict, cfg: Config) -> Dict:
    macro, fund = inp["macro"], inp["fund"]
    expected, vol, financing, annual_cost, kw, sim = _model_sim(inp, cfg)
    T = max(HORIZONS)
    lc = leverage_cost(cfg.leverage_target, cfg.expense_ratio_default, financing, sigma_1y(vol), expected.final, None)
    sim7 = simulate(mu_annual=np.full(T, CONSTANT_DRIFT), **kw)
    rating = rate(sim, RATING_H, cfg)
    rating7 = rate(sim7, RATING_H, cfg)

    origin = d.index[pos]
    rec = {"date": origin, "pos": pos, "drift": expected.final, "drift_base": expected.base,
           "regime_adj": sum(expected.adjustments.values()), "ep": fund.earnings_yield, "dy": fund.dividend_yield,
           "infl": macro.breakeven_10y, "infl_source": inp["infl_source"], "rf": macro.rf_3m, "curve": macro.curve_10y_3m,
           "hy_oas": macro.hy_oas, "cpi_yoy": macro.cpi_yoy, "vix": macro.vix, "vix6m_used": macro.vix6m,
           "ts_imputed": inp["ts_imputed"], "vol_6m": vol.total_vol(RATING_H), "annual_cost": annual_cost,
           "label": rating.label, "score": rating.score, "score_se": rating.score_se, "edge": rating.edge_annual,
           "sharpe": rating.sharpe_annual, "borderline": rating.borderline,
           "label_const7": rating7.label, "score_const7": rating7.score,
           "hurdle": lc.hurdle, "lc_fees": lc.fees, "lc_financing": lc.financing, "lc_drag": lc.drag,
           "sigma_1y": lc.sigma, "p_dip20_3m": sim.summary(DIP_HORIZON)["p_drawdown_20"]}
    for h in HORIZONS:
        if pos + h >= len(d):
            continue
        window = _fund_window(d, pos, h)
        y = float(np.log(window[-1]))
        ms = _model_scores(sim, window, h)
        m, s = naive_leveraged_lognormal(macro.vix, macro.rf_3m, annual_cost, h, cfg.leverage_target)
        tbill = float(np.prod(1.0 + d["tbill_ret"].values[pos + 1: pos + h + 1]) - 1.0)
        rec.update({
            f"ret_{h}": float(window[-1] - 1.0), f"tbill_{h}": tbill, f"excess_{h}": float(window[-1] - 1.0 - tbill),
            f"sp_{h}": float(d["tr"].values[pos + h] / d["tr"].values[pos] - 1.0),
            **{k: ms[k] for k in (f"pit_{h}", f"in90_{h}", f"in50_{h}", f"crps_model_{h}")},
            f"crps_const7_{h}": crps_sample(np.log(sim7.terminal[h]), y),
            f"crps_naive_{h}": crps_normal(m, s, y),
            f"pred_dd20_{h}": ms[f"pred_dd20_{h}"], f"real_dd20_{h}": ms[f"real_dd20_{h}"],
        })
        if h == RATING_H:   # realised fund cost + volatility drag per year: L x index log return - fund log return
            rec["real_cost"] = (cfg.leverage_target * float(np.log(d["tr"].values[pos + h] / d["tr"].values[pos]))
                                - y) * 252.0 / h
    return rec


def engine_origin(d: pd.DataFrame, pos: int, inp: Dict, cfg: Config) -> Dict:
    """The model's simulation alone at one origin, scored at every horizon with an outcome: all an
    engine-setting comparison needs (no rating, no constant-drift twin)."""
    sim = _model_sim(inp, cfg)[-1]
    rec = {"date": d.index[pos], "pos": pos}
    for h in HORIZONS:
        if pos + h < len(d):
            rec.update(_model_scores(sim, _fund_window(d, pos, h), h))
    return rec


# ---------------------------------------------------------------------------------------
# Engine settings compared (--engine-grid; the rule is research/CALIBRATION.md's)
# ---------------------------------------------------------------------------------------
HLABEL = {5: "1W", 10: "2W", 21: "1M", 63: "3M", 126: "6M"}


def load_grid(path: Path) -> Dict:
    """The settings file: {"reference": name, "settings": {name: {Config field: value}}, "rule": {...}}."""
    grid = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(Config)}
    for name, over in grid["settings"].items():
        unknown = set(over) - fields
        if unknown:
            raise SystemExit(f"setting {name!r}: {sorted(unknown)} are not Config fields")
    if grid["reference"] not in grid["settings"]:
        raise SystemExit(f"the reference setting {grid['reference']!r} is not in the settings")
    return grid


def engine_config(base: Config, overrides: Dict) -> Config:
    return dataclasses.replace(base, **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})


def engine_table(o: pd.DataFrame, rule: Dict, reference: str) -> pd.DataFrame:
    """One row per setting: Brier score of the 20% dip, CRPS and 90% band coverage per horizon, the
    guardrails against the reference setting, and the rule's pick (lowest mean Brier score among the
    settings within the guardrails; the reference always counts and wins an exact tie)."""
    bh, ch, vh = rule["brier_horizons"], rule["crps_horizons"], rule["coverage_horizons"]
    lo, hi = rule["coverage_range"]
    rows = []
    for name, g in o.groupby("setting", sort=False):
        r = {"setting": name}
        for h in sorted(set(bh) | set(ch) | set(vh)):
            e = g.dropna(subset=[f"pit_{h}"])
            r[f"n_{h}"] = len(e)
            r[f"brier_{h}"] = float(((e[f"pred_dd20_{h}"] - e[f"real_dd20_{h}"]) ** 2).mean())
            r[f"pred_{h}"], r[f"real_{h}"] = float(e[f"pred_dd20_{h}"].mean()), float(e[f"real_dd20_{h}"].mean())
            r[f"crps_{h}"] = float(e[f"crps_model_{h}"].mean())
            r[f"cov90_{h}"], r[f"cov50_{h}"] = float(e[f"in90_{h}"].mean()), float(e[f"in50_{h}"].mean())
            r[f"pit_{h}"] = float(e[f"pit_{h}"].mean())
        r["brier_mean"] = float(np.mean([r[f"brier_{h}"] for h in bh]))
        rows.append(r)
    t = pd.DataFrame(rows).set_index("setting")
    ref = t.loc[reference]
    for h in ch:
        t[f"crps_vs_ref_{h}"] = t[f"crps_{h}"] / ref[f"crps_{h}"] - 1.0
    t["g1_crps"] = np.all([t[f"crps_vs_ref_{h}"] <= rule["crps_tolerance"] for h in ch], axis=0)
    t["g2_coverage"] = np.all([(t[f"cov90_{h}"] >= lo) & (t[f"cov90_{h}"] <= hi) for h in vh], axis=0)
    t["eligible"] = (t["g1_crps"] & t["g2_coverage"]) | (t.index == reference)
    order = t[t["eligible"]].assign(is_ref=lambda x: x.index != reference).sort_values(["brier_mean", "is_ref"])
    t["rank"] = pd.Series(np.arange(1, len(order) + 1), index=order.index).reindex(t.index)
    t.attrs["pick"] = order.index[0]
    return t


def brier_diff(o: pd.DataFrame, name: str, reference: str, h: int) -> tuple:
    """Mean Brier-score difference of ``name`` minus ``reference`` at h over the same origins, with
    an overlap-aware 90% interval (for information; the rule does not use it)."""
    cols = ["pos", f"pred_dd20_{h}", f"real_dd20_{h}"]
    a = o.loc[o["setting"] == name, cols].dropna().set_index("pos")
    b = o.loc[o["setting"] == reference, cols].dropna().set_index("pos").reindex(a.index)
    diff = (a[f"pred_dd20_{h}"] - a[f"real_dd20_{h}"]) ** 2 - (b[f"pred_dd20_{h}"] - b[f"real_dd20_{h}"]) ** 2
    return (float(diff.mean()), *mean_interval(diff.values, effective_n(a.index, h)))


def engine_report(o: pd.DataFrame, grid: Dict) -> List[str]:
    rule, reference = grid["rule"], grid["reference"]
    bh, ch, vh = rule["brier_horizons"], rule["crps_horizons"], rule["coverage_horizons"]
    lo, hi = rule["coverage_range"]
    t = engine_table(o, rule, reference)
    names = list(t.index)
    out = [f"Reference setting (the guardrails compare with it): {reference}", "",
           "Brier score of the 20% dip (lower is better; the objective is the mean over "
           + "/".join(HLABEL[h] for h in bh) + "), and the chance predicted / how often it happened"]
    rows = []
    for n in names:
        r = t.loc[n]
        rows.append({"setting": n, **{f"Brier {HLABEL[h]}": f"{r[f'brier_{h}']:.4f}" for h in bh},
                     "mean": f"{r['brier_mean']:.5f}",
                     **{f"dip {HLABEL[h]} pred/real": f"{r[f'pred_{h}']:.1%} / {r[f'real_{h}']:.1%}" for h in bh}})
    out.append(pd.DataFrame(rows).to_string(index=False))

    out.append("\nBrier score minus the reference's, same month-ends [90% CI, overlap-aware; information only]")
    rows = []
    for n in names:
        rec = {"setting": n}
        for h in bh:
            m, a, b = brier_diff(o, n, reference, h)
            rec[HLABEL[h]] = "0 (reference)" if n == reference else f"{m:+.4f} {_ci(a, b, '{:+.4f}')}"
        rows.append(rec)
    out.append(pd.DataFrame(rows).to_string(index=False))

    out.append(f"\nCRPS of the fund's log return (lower is better) and its change from the reference's "
               f"(guardrail G1: at most {rule['crps_tolerance']:+.1%} at each of " + "/".join(HLABEL[h] for h in ch) + ")")
    out.append(pd.DataFrame([{"setting": n, **{HLABEL[h]: f"{t.at[n, f'crps_{h}']:.4f} ({t.at[n, f'crps_vs_ref_{h}']:+.2%})"
                                               for h in ch}} for n in names]).to_string(index=False))

    out.append(f"\n90% band coverage (guardrail G2: {lo:.0%} to {hi:.0%} at each of " + "/".join(HLABEL[h] for h in vh)
               + "); 50% band and mean PIT at 3M for information")
    out.append(pd.DataFrame([{"setting": n, **{HLABEL[h]: f"{t.at[n, f'cov90_{h}']:.1%}" for h in vh},
                              "50% band 3M": f"{t.at[n, 'cov50_63']:.1%}" if "cov50_63" in t else "",
                              "mean PIT 3M": f"{t.at[n, 'pit_63']:.3f}" if "pit_63" in t else ""}
                             for n in names]).to_string(index=False))

    out.append("\nGuardrails and rule (the reference always counts; ranked by the mean Brier score)")
    out.append(pd.DataFrame([{"setting": n, "G1 CRPS": "ok" if t.at[n, "g1_crps"] else "fails",
                              "G2 coverage": "ok" if t.at[n, "g2_coverage"] else "fails",
                              "counts": "yes" if t.at[n, "eligible"] else "no",
                              "rank": "" if pd.isna(t.at[n, "rank"]) else f"{int(t.at[n, 'rank'])}"}
                             for n in names]).to_string(index=False))
    n_obs = ", ".join(f"{HLABEL[h]} {int(t[f'n_{h}'].iloc[0])}" for h in sorted(set(bh) | set(vh)))
    out.append(f"(month-ends with an outcome: {n_obs})")
    out.append(f"\nThe rule picks: {t.attrs['pick']}"
               + (" (the reference: no change)" if t.attrs["pick"] == reference else ""))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_engine_grid(args, out_dir: Path, d: pd.DataFrame, inputs: Dict, fits: Dict, positions: List[int]) -> None:
    grid = load_grid(args.engine_grid)
    names = list(grid["settings"])
    if args.only:
        missing = [n for n in args.only if n not in grid["settings"]]
        if missing:
            raise SystemExit(f"--only: {missing} not in {args.engine_grid}")
        names = [n for n in names if n in args.only]
    if grid["reference"] not in names:
        raise SystemExit(f"the reference setting {grid['reference']!r} must be run too")
    base = Config(n_paths=args.paths, use_news=False, horizons=tuple(HORIZONS), rating_horizon=RATING_H)
    origins = [(pos, inputs_at(d, pos, inputs, base, args.term_structure, fits)) for pos in positions]
    origins = [(pos, inp) for pos, inp in origins if inp is not None]
    if not origins:
        raise SystemExit("no month-end had the inputs the model needs")
    rule = grid["rule"]
    short = sorted(h for h in set(rule["brier_horizons"]) | set(rule["crps_horizons"]) | set(rule["coverage_horizons"])
                   if origins[0][0] + h >= len(d))
    if short:
        raise SystemExit("no month-end in the window has an outcome at " + "/".join(HLABEL.get(h, str(h)) for h in short)
                         + " yet: use an earlier --start")
    rows, t0 = [], time.time()
    for i, name in enumerate(names):
        cfg = engine_config(base, grid["settings"][name])
        # inputs rebuilt per setting: they depend on the Config (earnings growth); which month-ends
        # have inputs does not
        rows += [{"setting": name, **engine_origin(d, pos, inputs_at(d, pos, inputs, cfg, args.term_structure, fits), cfg)}
                 for pos, _ in origins]
        print(f"  {i + 1}/{len(names)} {name} ({time.time() - t0:.0f}s)", flush=True)
    o = pd.DataFrame(rows)
    tag = args.tag or "grid"
    o.to_csv(out_dir / f"engine_{tag}.csv", index=False)
    first, last = d.index[origins[0][0]], d.index[origins[-1][0]]
    lines = [f"SPXLcast engine settings compared, run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"{len(origins)} month-ends {first:%Y-%m} .. {last:%Y-%m} ({first.date()} .. {last.date()}); "
             f"{args.paths} paths per simulation, seed {base.seed}; model simulation only; "
             f"VIX term structure before 2008: {args.term_structure}",
             f"settings: {args.engine_grid} (SHA-256 {_sha256(args.engine_grid)[:16]}...), {len(names)} run; "
             f"inputs: rating_inputs.pkl (SHA-256 {_sha256(out_dir / 'rating_inputs.pkl')[:16]}...)",
             "", *engine_report(o, grid)]
    report = "\n".join(lines)
    (out_dir / f"engine_{tag}.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nwrote {out_dir / f'engine_{tag}.txt'} and engine_{tag}.csv ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------
def _ci(lo: float, hi: float, fmt: str = "{:+.1%}") -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[n/a]"
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def _diff_ci(y: pd.Series, sel: pd.Series, pos: pd.Series, h: int) -> tuple:
    """90% interval for mean(y where sel) - mean(y elsewhere): the t interval of ``mean_interval`` on
    the difference's influence values (Welch's variance with the whole sample's independent-window
    count, which stays conservative when the selected months cluster). n/a unless each group spans
    at least three independent windows: with two, the spread rests on two episodes and the interval
    covers only about 82%."""
    if min(effective_n(pos[sel], h), effective_n(pos[~sel], h)) < 3:
        return float("nan"), float("nan")
    p = float(sel.mean())
    m1, m0 = y[sel].mean(), y[~sel].mean()
    psi = np.where(sel, (y - m1) / p, -(y - m0) / (1.0 - p))
    lo, hi = mean_interval(psi, effective_n(pos, h))
    return m1 - m0 + lo, m1 - m0 + hi


def _corr_ci(r: float, n_eff: float, alpha: float = 0.10) -> tuple:
    """Fisher-z interval for a Spearman correlation on n_eff independent pairs, with Bonett and
    Wright's variance (1 + r^2 / 2) / (n - 3)."""
    from scipy.stats import norm
    if not (n_eff > 4 and abs(r) < 1):
        return float("nan"), float("nan")
    half = float(norm.ppf(1.0 - alpha / 2.0)) * math.sqrt((1.0 + r * r / 2.0) / (n_eff - 3.0))
    return math.tanh(math.atanh(r) - half), math.tanh(math.atanh(r) + half)


def rating_section(o: pd.DataFrame) -> List[str]:
    out = []
    h = RATING_H
    ev = o.dropna(subset=[f"excess_{h}"])
    n_eff = effective_n(ev["pos"], h)
    out.append(f"Forward 6-month SPXL return by the rating given ({len(ev)} month-ends with a full 6 months after; "
               f"consecutive windows overlap, about {n_eff:.0f} independent)")
    rows = []
    for lab in LABELS:
        g = ev[ev["label"] == lab]
        if len(g) == 0:
            rows.append({"rating": lab, "months": 0, "share": "0%"})
            continue
        diff_lo, diff_hi = _diff_ci(ev[f"excess_{h}"], ev["label"] == lab, ev["pos"], h)
        rows.append({"rating": lab, "months": len(g), "share": f"{len(g) / len(ev):.0%}",
                     "SPXL mean": f"{g[f'ret_{h}'].mean():+.1%}", "SPXL median": f"{g[f'ret_{h}'].median():+.1%}",
                     "excess vs T-bill": f"{g[f'excess_{h}'].mean():+.1%}",
                     "P(beat T-bill)": f"{(g[f'excess_{h}'] > 0).mean():.0%}",
                     "S&P 500 mean": f"{g[f'sp_{h}'].mean():+.1%}",
                     "excess minus the other months [90% CI]":
                         (f"{g[f'excess_{h}'].mean() - ev.loc[ev['label'] != lab, f'excess_{h}'].mean():+.1%} "
                          f"{_ci(diff_lo, diff_hi)}" if (ev["label"] != lab).any() else "n/a (every month)")})
    out.append(pd.DataFrame(rows).fillna("").to_string(index=False))

    out.append("\nRank correlation of the score with the forward SPXL excess return [90% CI, overlap-aware]")
    rows = []
    for hh in (21, 63, 126):
        e = o.dropna(subset=[f"excess_{hh}"])
        r, ne = spearman(e["score"].values, e[f"excess_{hh}"].values), effective_n(e["pos"], hh)
        rows.append({"horizon": f"{hh // 21}M", "n": len(e), "independent": f"{ne:.0f}",
                     "model score": f"{r:+.2f} {_ci(*_corr_ci(r, ne), '{:+.2f}')}",
                     "score with constant 7% drift": f"{spearman(e['score_const7'].values, e[f'excess_{hh}'].values):+.2f}"})
    out.append(pd.DataFrame(rows).to_string(index=False))

    if len(ev) < 5:
        return out
    out.append("\nForward 6-month SPXL excess return by score quintile")
    # ranked first, so scores tied at the +1.00 cap cannot collapse two quintile edges into one
    q = pd.qcut(ev["score"].rank(method="first"), 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    split = [s for s, k in q.groupby(ev["score"]).nunique().items() if k > 1]
    if split:
        out.append(f"(months tied at a score of {', '.join(f'{s:+.2f}' for s in split)} are split between "
                   f"quintiles in date order)")
    t = ev.groupby(q, observed=True).agg(months=("score", "size"), score_from=("score", "min"), score_to=("score", "max"),
                                         mean_excess=(f"excess_{h}", "mean"),
                                         p_beat_tbill=(f"excess_{h}", lambda s: (s > 0).mean()))
    t["mean_excess"] = t["mean_excess"].map("{:+.1%}".format)
    t["p_beat_tbill"] = t["p_beat_tbill"].map("{:.0%}".format)
    t["score_from"] = t["score_from"].map("{:+.2f}".format)
    t["score_to"] = t["score_to"].map("{:+.2f}".format)
    out.append(t.to_string())
    return out


def threshold_section(o: pd.DataFrame) -> List[str]:
    h = RATING_H
    ev = o.dropna(subset=[f"excess_{h}"])
    halves = {"all": ev, "1990-2007": ev[ev["date"] < "2008-01-01"], "2008-2026": ev[ev["date"] >= "2008-01-01"]}
    out = ["Threshold sweep: mean forward 6-month excess return of the months a rule selects vs the rest "
           "(BUY = score >= t and median edge > 0; SELL = score <= -t)"]
    rows = []
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for side in ("BUY", "SELL"):
            rec = {"rule": f"{side} at {'+' if side == 'BUY' else '-'}{t:.1f}"}
            for name, g in halves.items():
                sel = (g["score"] >= t) & (g["edge"] > 0) if side == "BUY" else (g["score"] <= -t)
                if sel.sum() == 0:
                    rec[name] = "never"
                    continue
                gap = g.loc[sel, f"excess_{h}"].mean() - (g.loc[~sel, f"excess_{h}"].mean() if (~sel).any() else np.nan)
                rec[name] = f"{sel.mean():>4.0%} of months, {g.loc[sel, f'excess_{h}'].mean():+.1%} ({gap:+.1%} vs rest)"
            rows.append(rec)
    out.append(pd.DataFrame(rows).to_string(index=False))
    return out


def strategy_section(o: pd.DataFrame, d: pd.DataFrame) -> List[str]:
    """Monthly decisions at each month-end close, held until the next month-end."""
    o = o.sort_values("pos").reset_index(drop=True)
    fund, sp, bill = d["fund_ret"].fillna(0).values, d["tr"].pct_change().fillna(0).values, d["tbill_ret"].fillna(0).values
    rules = {
        "buy and hold SPXL": lambda lab: "fund",
        "buy and hold S&P 500": lambda lab: "sp",
        "SPXL, T-bills when SELL": lambda lab: "bill" if lab == "SELL" else "fund",
        "SPXL only when BUY, else T-bills": lambda lab: "fund" if lab == "BUY" else "bill",
        "BUY: SPXL, HOLD: S&P 500, SELL: T-bills": lambda lab: {"BUY": "fund", "HOLD": "sp", "SELL": "bill"}[lab],
    }
    series = {"fund": fund, "sp": sp, "bill": bill}
    rows = []
    for name, rule in rules.items():
        rets, in_fund = [], 0
        for i in range(len(o) - 1):
            a, b = int(o.at[i, "pos"]), int(o.at[i + 1, "pos"])
            asset = rule(o.at[i, "label"])
            in_fund += asset == "fund"
            rets.append(series[asset][a + 1: b + 1])
        r = np.concatenate(rets)
        wealth = np.cumprod(1.0 + r)
        years = len(r) / 252.0
        dd = 1.0 - wealth / np.maximum.accumulate(wealth)
        rows.append({"rule": name, "CAGR": f"{wealth[-1] ** (1 / years) - 1:+.1%}",
                     "vol": f"{r.std() * np.sqrt(252):.0%}", "max drawdown": f"{-dd.max():.0%}",
                     "months in SPXL": f"{in_fund / (len(o) - 1):.0%}", "$1 becomes": f"{wealth[-1]:,.2f}"})
    first, last = d.index[int(o['pos'].iloc[0])].date(), d.index[int(o['pos'].iloc[-1])].date()
    return [f"Following the rating, rebalanced monthly {first} to {last} (illustration: no trading costs or taxes; "
            f"SPXL before 2009 is the synthetic fund)", pd.DataFrame(rows).to_string(index=False)]


NO_OUTCOMES = "  (no month-end has an outcome at any horizon yet)"


def calibration_section(o: pd.DataFrame) -> List[str]:
    out = ["Calibration of the full model (targets: mean PIT 0.50, 90% band 90%, 50% band 50%) [90% CI, overlap-aware]"]
    rows = []
    for h in HORIZONS:
        if f"pit_{h}" not in o:
            continue
        e = o.dropna(subset=[f"pit_{h}"])
        n_eff = effective_n(e["pos"], h)
        pit_ci = mean_interval(e[f"pit_{h}"].values, n_eff)
        c90_ci = proportion_interval(e[f"in90_{h}"].astype(float).mean(), n_eff)
        rec = {"horizon": {5: "1W", 10: "2W", 21: "1M", 63: "3M", 126: "6M"}[h], "n": len(e),
               "independent": f"{n_eff:.0f}",
               "mean PIT": f"{e[f'pit_{h}'].mean():.2f} {_ci(*pit_ci, '{:.2f}')}",
               "in 90% band": f"{e[f'in90_{h}'].mean():.0%} {_ci(*c90_ci, '{:.0%}')}",
               "in 50% band": f"{e[f'in50_{h}'].mean():.0%}",
               "below 5% / above 95%": f"{(e[f'pit_{h}'] < 0.05).mean():.0%} / {(e[f'pit_{h}'] > 0.95).mean():.0%}"}
        if h >= 21:
            rec["P(-20% dip) pred/real"] = f"{e[f'pred_dd20_{h}'].mean():.0%} / {e[f'real_dd20_{h}'].mean():.0%}"
        rows.append(rec)
    out.append(pd.DataFrame(rows).fillna("").to_string(index=False) if rows else NO_OUTCOMES)
    return out


def skill_section(o: pd.DataFrame) -> List[str]:
    out = ["Skill: CRPS of the SPXL log return (lower = better forecast); skill = 1 - CRPS(model)/CRPS(benchmark), "
           "positive = the model is better [90% CI]"]
    rows = []
    for h in HORIZONS:
        if f"crps_model_{h}" not in o:
            continue
        e = o.dropna(subset=[f"crps_model_{h}"])
        n_eff = effective_n(e["pos"], h)
        rec = {"horizon": {5: "1W", 10: "2W", 21: "1M", 63: "3M", 126: "6M"}[h], "n": len(e),
               "CRPS model": f"{e[f'crps_model_{h}'].mean():.4f}"}
        for bench, label in (("naive", "vs naive lognormal"), ("const7", "vs constant 7% drift")):
            sk = 1.0 - e[f"crps_model_{h}"].mean() / e[f"crps_{bench}_{h}"].mean()
            rec[label] = f"{sk:+.1%} {_ci(*skill_interval(e[f'crps_model_{h}'], e[f'crps_{bench}_{h}'], n_eff))}"
        rows.append(rec)
    out.append(pd.DataFrame(rows).to_string(index=False) if rows else NO_OUTCOMES)
    for era, sel in (("1990-2007", o["date"] < "2008-01-01"), ("2008-2026", o["date"] >= "2008-01-01")):
        if f"crps_model_{RATING_H}" not in o:
            break
        e = o[sel].dropna(subset=[f"crps_model_{RATING_H}"])
        if len(e):
            out.append(f"  6M skill {era}: vs naive {1 - e[f'crps_model_{RATING_H}'].mean() / e[f'crps_naive_{RATING_H}'].mean():+.1%}, "
                       f"vs constant 7% {1 - e[f'crps_model_{RATING_H}'].mean() / e[f'crps_const7_{RATING_H}'].mean():+.1%} "
                       f"({len(e)} months)")
    return out


# ---------------------------------------------------------------------------------------
# Leverage cost and drawdown risk: the assessment the page shows, and its reference.json
# ---------------------------------------------------------------------------------------
GRID_PCTS = list(range(0, 101, 5))


def _r(v, nd: int = 6):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def _terciles(x: pd.Series) -> Dict:
    x = x.dropna().values
    return {"grid": [_r(v) for v in np.percentile(x, GRID_PCTS)],
            "cutoffs": [_r(v) for v in np.percentile(x, [100.0 / 3.0, 200.0 / 3.0])]}


def _cost_frame(o: pd.DataFrame, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    e = o.dropna(subset=["real_cost"]) if "real_cost" in o else o.iloc[0:0].assign(real_cost=np.nan)
    return e[e["date"] >= since] if since is not None else e


def _cost_check(e: pd.DataFrame, pred: pd.Series) -> Dict:
    """Predicted fund cost + volatility drag per year against the realised over the next 6 months."""
    n_eff = effective_n(e["pos"], RATING_H)
    corr = float(pred.corr(e["real_cost"])) if len(e) > 2 else float("nan")
    return {"pred_mean": float(pred.mean()), "real_mean": float(e["real_cost"].mean()), "corr": corr, "n": int(len(e)),
            "n_indep": n_eff, "diff_ci": mean_interval((e["real_cost"] - pred).values, n_eff),
            "corr_ci": _corr_ci(corr, n_eff)}


def build_reference(o: pd.DataFrame, source: str) -> Dict:
    """The tercile cutoffs and history spxlcast/assess.py places a live run against."""
    h = DIP_HORIZON
    ref = {"source": source, "period": f"{o['date'].iloc[0]:%Y-%m}..{o['date'].iloc[-1]:%Y-%m}", "n": int(len(o)),
           "hurdle": _terciles(o["hurdle"]),
           "p_dip20_3m": {**_terciles(o["p_dip20_3m"]), "median": _r(o["p_dip20_3m"].median())}}
    by_level = {}
    if f"real_dd20_{h}" in o:
        e = o.dropna(subset=[f"real_dd20_{h}"])
        # classified on the rounded cutoffs the live page reads back
        lvl = e["p_dip20_3m"].map(lambda p: level_of(p, ref["p_dip20_3m"]["cutoffs"], DRAWDOWN_LEVELS))
        for lab in DRAWDOWN_LEVELS:
            g = e[lvl == lab]
            if len(g):
                by_level[lab] = {"pred": _r(g["p_dip20_3m"].mean()), "real": _r(g[f"real_dd20_{h}"].mean()),
                                 "n": int(len(g)), "n_indep": _r(effective_n(g["pos"], h), 1)}
    ref["p_dip20_3m"]["by_level"] = by_level
    e = _cost_frame(o, SPXL_FROM)       # SPXL itself: before it the synthetic fund charges the model's own costs
    c = _cost_check(e, e["lc_fees"] + e["lc_financing"] + e["lc_drag"])
    ref["leverage_cost_check"] = {"pred_mean": _r(c["pred_mean"]), "real_mean": _r(c["real_mean"]), "corr": _r(c["corr"], 3),
                                  "n": c["n"], "n_indep": _r(c["n_indep"], 1),
                                  "period": f"{e['date'].iloc[0]:%Y-%m}..{e['date'].iloc[-1]:%Y-%m}" if len(e) else None}
    return ref


def assessment_section(o: pd.DataFrame, ref: Dict, cfg: Config) -> List[str]:
    hu, dp = ref["hurdle"], ref["p_dip20_3m"]
    (hlo, hhi), (dlo, dhi) = hu["cutoffs"], dp["cutoffs"]
    out = [f"Leverage cost and drawdown risk (what the page shows; levels are terciles of these {ref['n']} month-ends)",
           f"Leverage cost = the S&P 500 total return per year SPXL needs to break even over the long run (its average "
           f"log growth is then zero): low below "
           f"{hlo:.1%}, normal {hlo:.1%} to {hhi:.1%}, high {hhi:.1%} and above (median {hu['grid'][10]:.1%}, "
           f"range {hu['grid'][0]:.1%} to {hu['grid'][-1]:.1%})",
           f"Drawdown risk = the predicted chance SPXL closes at least 20% below the month-end price within 3 months: low below "
           f"{dlo:.1%}, normal {dlo:.1%} to {dhi:.1%}, elevated {dhi:.1%} and above (median {dp['median']:.1%}, "
           f"range {dp['grid'][0]:.1%} to {dp['grid'][-1]:.1%})"]

    h = DIP_HORIZON
    out.append("\nDid a 20% dip within 3 months follow as often as predicted? By drawdown-risk level [90% CI, overlap-aware]")
    rows = []
    if f"real_dd20_{h}" in o:
        e = o.dropna(subset=[f"real_dd20_{h}"])
        lvl = e["p_dip20_3m"].map(lambda p: level_of(p, dp["cutoffs"], DRAWDOWN_LEVELS))
        for lab in DRAWDOWN_LEVELS + ("all",):
            g = e if lab == "all" else e[lvl == lab]
            if not len(g):
                continue
            n_eff, real = effective_n(g["pos"], h), g[f"real_dd20_{h}"].mean()
            rows.append({"level": lab, "predicted from..to": f"{g['p_dip20_3m'].min():.0%} to {g['p_dip20_3m'].max():.0%}",
                         "months": len(g), "independent": f"{n_eff:.0f}", "predicted (mean)": f"{g['p_dip20_3m'].mean():.1%}",
                         "happened [90% CI]": f"{real:.1%} {_ci(*proportion_interval(real, n_eff), '{:.0%}')}"})
    out.append(pd.DataFrame(rows).to_string(index=False) if rows else NO_OUTCOMES)

    L = cfg.leverage_target
    out.append("\nLeverage cost check: predicted fund cost + volatility drag per year (fees + financing + L(L-1)/2 x vol^2, "
               "vol the 1-year implied as on the page) vs realised over the next 6 months (3 x the S&P 500's log return "
               "minus the fund's, per year) [90% CI, overlap-aware]")
    rows = []
    for name, since, vol_col in (("SPXL", SPXL_FROM, "sigma_1y"), ("SPXL, drag at the 6-month vol", SPXL_FROM, "vol_6m"),
                                 ("all, synthetic fund before 2009", None, "sigma_1y")):
        e = _cost_frame(o, since)
        if not len(e):
            continue
        c = _cost_check(e, e["lc_fees"] + e["lc_financing"] + 0.5 * L * (L - 1.0) * e[vol_col] ** 2)
        rows.append({"sample": f"{name} {e['date'].iloc[0]:%Y-%m}..{e['date'].iloc[-1]:%Y-%m}", "months": c["n"],
                     "independent": f"{c['n_indep']:.0f}", "predicted": f"{c['pred_mean']:.1%}",
                     "realised": f"{c['real_mean']:.1%}",
                     "realised minus predicted": f"{c['real_mean'] - c['pred_mean']:+.1%} {_ci(*c['diff_ci'])}",
                     "correlation": f"{c['corr']:+.2f} {_ci(*c['corr_ci'], '{:+.2f}')}"})
    out.append(pd.DataFrame(rows).to_string(index=False) if rows else NO_OUTCOMES)
    if rows:
        out.append("(before 2009 the synthetic fund charges the model's own fees and financing, so there only the drag is tested)")

    H = RATING_H
    out.append("\nFor information only, not a signal: forward 6-month SPXL return minus T-bills by leverage-cost level "
               "[90% CI, overlap-aware]")
    rows = []
    if f"excess_{H}" in o:
        ev = o.dropna(subset=[f"excess_{H}"])
        lvl = ev["hurdle"].map(lambda x: level_of(x, hu["cutoffs"], LEVERAGE_LEVELS))
        for lab in LEVERAGE_LEVELS:
            sel = lvl == lab
            g = ev[sel]
            if not len(g):
                continue
            lo, hi = _diff_ci(ev[f"excess_{H}"], sel, ev["pos"], H)
            rows.append({"level": lab, "hurdle from..to": f"{g['hurdle'].min():.1%} to {g['hurdle'].max():.1%}",
                         "months": len(g), "independent": f"{effective_n(g['pos'], H):.0f}",
                         "mean": f"{g[f'excess_{H}'].mean():+.1%}", "median": f"{g[f'excess_{H}'].median():+.1%}",
                         "P(beat T-bill)": f"{(g[f'excess_{H}'] > 0).mean():.0%}",
                         "minus the other months [90% CI]":
                             (f"{g[f'excess_{H}'].mean() - ev.loc[~sel, f'excess_{H}'].mean():+.1%} {_ci(lo, hi)}"
                              if (~sel).any() else "n/a (every month)")})
    out.append(pd.DataFrame(rows).to_string(index=False) if rows else NO_OUTCOMES)
    return out


def plot(o: pd.DataFrame, cfg: Config, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"BUY": "#2e9e4f", "HOLD": "#d9a400", "SELL": "#d0453b"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.plot(o["date"], o["score"], color="#333", lw=1.1, label="model score")
    ax1.plot(o["date"], o["score_const7"], color="#999", lw=0.8, ls="--", label="score with a constant 7% drift")
    ax1.axhline(cfg.buy_score, color=colors["BUY"], lw=0.8, ls=":")
    ax1.axhline(cfg.sell_score, color=colors["SELL"], lw=0.8, ls=":")
    ax1.set_ylabel("rating score")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("SPXLcast rating at each month-end (dotted: BUY / SELL thresholds)")
    if f"excess_{RATING_H}" in o:
        e = o.dropna(subset=[f"excess_{RATING_H}"])
        ax2.bar(e["date"], e[f"excess_{RATING_H}"], width=25, color=[colors[x] for x in e["label"]])
    ax2.axhline(0, color="#333", lw=0.6)
    ax2.set_ylabel("SPXL return over the next 6 months\nminus T-bills")
    ax2.set_title("What happened next, coloured by the rating given (green BUY, amber HOLD, red SELL)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(n_paths=args.paths, use_news=False, horizons=tuple(HORIZONS), rating_horizon=RATING_H)
    inputs = load_inputs(out_dir, args.refresh)
    d = build_daily(inputs, cfg)
    fits = fit_term_structure(d)
    end = getattr(args, "end", None)
    positions = month_end_positions(d.index, args.start, end)
    if not positions:
        raise SystemExit(f"no complete month-end on or after --start {args.start}" + (f" through --end {end}" if end else ""))
    if getattr(args, "engine_grid", None):
        run_engine_grid(args, out_dir, d, inputs, fits, positions)
        return
    val = validate_synthetic(d)
    print(f"{len(positions)} month-ends {d.index[positions[0]].date()} .. {d.index[positions[-1]].date()}, "
          f"{args.paths} paths, term structure before 2008: {args.term_structure}", flush=True)
    rows, skipped, t0 = [], [], time.time()
    for i, pos in enumerate(positions):
        inp = inputs_at(d, pos, inputs, cfg, args.term_structure, fits)
        if inp is None:
            skipped.append(d.index[pos])
            continue
        rows.append(run_origin(d, pos, inp, cfg))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(positions)} ({time.time() - t0:.0f}s)", flush=True)
    if not rows:
        raise SystemExit("no month-end had the inputs the model needs")
    o = pd.DataFrame(rows)
    o.to_csv(out_dir / "rating_backtest.csv", index=False)

    # inputs the backtest had to do without, in the report and on stderr
    gaps = []
    if skipped:
        gaps.append(f"Skipped {len(skipped)} month-ends without the model's inputs ({skipped[0]:%Y-%m} .. "
                    f"{skipped[-1]:%Y-%m}); Shiller earnings end {inputs['shiller'].index[-1]:%Y-%m}")
    n_hy = int(o["hy_oas"].isna().sum())
    if n_hy:
        gaps.append(f"High-yield spread (FRED BAMLH0A0HYM2) only from {inputs['fred']['BAMLH0A0HYM2'].index.min():%Y-%m} "
                    f"(FRED keeps three years of the ICE BofA data): the credit-stress penalty cannot fire at {n_hy} of "
                    f"{len(o)} month-ends")
    for sid, s in inputs["fred"].items():     # the breakeven's start is in the header; the spread's is above
        if sid not in ("T10YIE", "BAMLH0A0HYM2") and s.index.min() > o["date"].iloc[0] + pd.DateOffset(years=1):
            gaps.append(f"FRED {sid} starts {s.index.min():%Y-%m}, after the first month-end")
    for g in gaps:
        print(f"WARNING: {g}", file=sys.stderr)
    lines = [f"SPXLcast rating backtest, run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"{len(o)} month-ends {o['date'].iloc[0]:%Y-%m} .. {o['date'].iloc[-1]:%Y-%m}; {args.paths} paths per "
             f"simulation; earnings of the last quarter reported ({EARNINGS_LAG_MONTHS} months after its end); "
             f"inflation: breakeven from 2003, trailing CPI before",
             *gaps,
             f"VIX term structure before 2008: {args.term_structure}"
             + (f" (log fit on {fits['vix6m'][3]} days: R2 {fits['vix3m'][2]:.2f} for VIX3M, {fits['vix6m'][2]:.2f} for VIX6M)"
                if args.term_structure == "impute" else ""),
             f"Synthetic 3x fund vs SPXL {SPXL_FROM:%Y}-now ({val['days']} days): daily correlation {val['corr']:.4f}, "
             f"growth {val['syn_growth']:+.1%}/yr vs {val['spxl_growth']:+.1%}/yr (gap {val['gap_per_year']:+.2%}/yr), "
             f"tracking {val['tracking_sd_annual']:.1%}/yr",
             f"Labels given: " + ", ".join(f"{lab} {(o['label'] == lab).mean():.0%}" for lab in LABELS)
             + f"; with a constant 7% drift: " + ", ".join(f"{lab} {(o['label_const7'] == lab).mean():.0%}" for lab in LABELS),
             "Intervals: 90%, closed form on the number of independent outcome windows (t for means, Wilson for "
             "band hits, a t interval on the difference for skill and for rating differences, Fisher z for rank correlations)",
             ""]
    ref = build_reference(o, f"scripts/backtest_rating.py, run {datetime.now(timezone.utc):%Y-%m-%d}, "
                             f"{args.paths} paths per month-end")
    lines += assessment_section(o, ref, cfg) + [""]
    if f"excess_{RATING_H}" in o and o[f"excess_{RATING_H}"].notna().any():
        lines += rating_section(o) + [""] + threshold_section(o) + [""]
    else:
        lines += ["No month-end has a full 6-month outcome yet: rating and threshold sections skipped", ""]
    if len(o) > 1:
        lines += strategy_section(o, d) + [""]
    lines += calibration_section(o) + [""] + skill_section(o)
    report = "\n".join(lines)
    (out_dir / "rating_backtest.txt").write_text(report + "\n", encoding="utf-8")
    ref_path = out_dir / "reference.json"
    ref_path.write_text(json.dumps(ref, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    plot(o, cfg, out_dir / "rating_backtest.png")
    print(report)
    print(f"\nwrote {out_dir / 'rating_backtest.csv'}, rating_backtest.txt, rating_backtest.png, reference.json "
          f"({time.time() - t0:.0f}s)")
    if getattr(args, "write_reference", False):
        shutil.copyfile(ref_path, REFERENCE_PATH)
        print(f"copied reference.json to {REFERENCE_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--start", default="1990-01")
    ap.add_argument("--end", default=None, help="last month-end origin, YYYY-MM (default: the last complete month)")
    ap.add_argument("--term-structure", choices=("impute", "flat"), default="impute")
    ap.add_argument("--engine-grid", default=None, metavar="FILE",
                    help="compare the engine settings in this JSON file (model simulation only; see "
                         "research/CALIBRATION.md) instead of running the full backtest")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME", help="with --engine-grid: only these settings")
    ap.add_argument("--tag", default=None, help="with --engine-grid: output names engine_<tag>.txt/.csv (default grid)")
    ap.add_argument("--refresh", action="store_true", help="re-download the input data")
    ap.add_argument("--write-reference", action="store_true",
                    help="also copy reference.json into the package (spxlcast/reference.json), where the page reads it")
    ap.add_argument("--out", default=str(ROOT / "output" / "backtest"))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    run(ap.parse_args())
