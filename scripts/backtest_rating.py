"""Historical backtest of the full model, rating included, at every month-end since 1990.

At each month-end the live model's own functions (expected_index_return, vol_term_structure,
simulate, rate) are fed only what was known that day:

* S&P 500 earnings and dividends from Shiller's data, lagged three months for publication, over
  the S&P 500 close
* expected inflation: the 10-year breakeven (FRED T10YIE) from 2003; before that, trailing 10-year
  CPI inflation (the proxy the drift backtest uses)
* regime inputs from FRED: 3-month bill and 10-year yields (curve), high-yield spread (from 1997),
  CPI and unemployment (only months already released)
* VIX, VIX3M and VIX6M from Yahoo. VIX3M/VIX6M start in 2008; before that they are imputed from the
  VIX with a log-linear fit on 2008-2026, the only use of later data (``--term-structure flat``
  uses the live model's own fallback instead: a flat curve at the VIX)

The realised outcome is SPXL itself from 2009. Before that it is a synthetic 3x fund: the daily
S&P 500 total return times three, less the costs the model charges (expense ratio plus financing
of the borrowed 2x at the 3-month bill plus the spread), checked against SPXL where both exist.

Reported:
1. Does the rating carry information? Forward 6-month SPXL returns by label and by score quintile,
   and the rank correlation of score with forward return, with overlap-aware 90% intervals.
2. Where do the thresholds belong? BUY/SELL threshold sweep, full sample and both halves.
3. What following the rating would have done (illustration only): monthly rules vs buy-and-hold.
4. Calibration of the full model at 1W to 6M: PIT, band coverage, 20% drawdown odds.
5. Skill (CRPS, lower is better) against the same engine with a constant 7% drift, which isolates
   the fundamentals, and against a naive lognormal at the raw VIX with a T-bill drift, which
   isolates everything.

Usage:  py scripts/backtest_rating.py [--paths 20000] [--start 1990-01] [--term-structure impute]
        [--refresh]   (re-download Yahoo, FRED and Shiller data; otherwise a local cache is used)
"""
from __future__ import annotations

import argparse
import math
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
from spxlcast.config import Config                                             # noqa: E402
from spxlcast.data import MarketSnapshot, _fred_api                            # noqa: E402
from spxlcast.env import fred_api_key                                          # noqa: E402
from spxlcast.evaluation import (block_bootstrap, crps_normal, crps_sample, effective_n,  # noqa: E402
                                 naive_leveraged_lognormal, spearman)
from spxlcast.fundamentals import (IndexFundamentals, MacroState, expected_index_return,  # noqa: E402
                                   vol_term_structure)
from spxlcast.montecarlo import simulate                                       # noqa: E402
from spxlcast.rating import rate                                               # noqa: E402

HORIZONS = [5, 10, 21, 63, 126]
RATING_H = 126
YAHOO = ["^SP500TR", "^GSPC", "^VIX", "^VIX3M", "^VIX6M", "SPXL"]
FRED = ["DGS3MO", "DGS10", "T10YIE", "BAMLH0A0HYM2", "CPIAUCSL", "UNRATE"]
EARNINGS_LAG_MONTHS = 3          # S&P reports a quarter's earnings about two months after it ends
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
    shiller_path = out_dir / "ie_data.xls"
    if refresh and shiller_path.exists():
        shiller_path.unlink()
    data = {"closes": closes, "fred": fred, "shiller": load_shiller(shiller_path),
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
def month_end_positions(idx: pd.DatetimeIndex, start: str) -> List[int]:
    s = pd.Series(np.arange(len(idx)), index=idx)
    last = s.groupby([idx.year, idx.month]).max()
    current = (idx[-1].year, idx[-1].month)          # the data's last month is still in progress
    return [int(p) for p in last.values if idx[p] >= pd.Timestamp(start) and (idx[p].year, idx[p].month) != current]


def released(series: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """Monthly observations already published at a month-end: up to the previous month."""
    cutoff = (origin.to_period("M") - 1).to_timestamp()
    return series[series.index <= cutoff].dropna()


def inputs_at(d: pd.DataFrame, pos: int, inputs: Dict, cfg: Config, ts_mode: str, fits: Dict) -> Optional[Dict]:
    row = d.iloc[pos]
    origin = d.index[pos]
    sh = inputs["shiller"]
    e_month = (origin.to_period("M") - EARNINGS_LAG_MONTHS).to_timestamp()
    if e_month not in sh.index or not np.isfinite(row["vix"]) or not np.isfinite(row["rf"]):
        return None
    E, D, P = float(sh.at[e_month, "E"]), float(sh.at[e_month, "D"]), float(row["gspc"])
    cpi = released(inputs["fred"]["CPIAUCSL"], origin)
    un = released(inputs["fred"]["UNRATE"], origin)
    if len(cpi) < 121 or len(un) < 15:
        return None
    trailing_infl = float((cpi.iloc[-1] / cpi.iloc[-121]) ** 0.1 - 1.0)
    bei = row["bei"] if np.isfinite(row["bei"]) else None
    infl = bei if bei is not None else trailing_infl
    roll = un.rolling(3).mean()

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
        unemployment=float(un.iloc[-1]) / 100.0, unemployment_sahm_gap=float(roll.iloc[-1] - roll.iloc[-13:-1].min()),
        cpi_yoy=float(cpi.iloc[-1] / cpi.iloc[-13] - 1.0), vix=float(row["vix"]), vix3m=vix3m, vix6m=vix6m,
        vvix=None, skew=None, dxy=None, oil=None, gold=None)
    fund = IndexFundamentals(index_level=P, trailing_pe=P / E, earnings_yield=E / P, dividend_yield=D / P,
                             eps_growth=cfg.long_run_real_eps_growth + infl, book_to_price=None, sales_to_price=None)
    return {"macro": macro, "fund": fund, "infl_source": "breakeven" if bei is not None else "trailing CPI",
            "ts_imputed": imputed}


# ---------------------------------------------------------------------------------------
# One origin
# ---------------------------------------------------------------------------------------
def run_origin(d: pd.DataFrame, pos: int, inp: Dict, cfg: Config) -> Dict:
    macro, fund = inp["macro"], inp["fund"]
    expected = expected_index_return(fund, macro, cfg)
    T = max(HORIZONS)
    vol = vol_term_structure(macro, MarketSnapshot(asof=datetime.now(timezone.utc)), cfg, T)
    annual_cost = cfg.expense_ratio_default + (cfg.leverage_target - 1) * (macro.rf_3m + cfg.swap_spread)
    kw = dict(spot=1.0, sigma_annual=vol.daily, leverage=cfg.leverage_target, daily_cost=annual_cost / 252.0,
              tracking_sd_daily=TRACKING_SD, rf_annual=macro.rf_3m, horizons=HORIZONS, n_paths=cfg.n_paths,
              dof=cfg.t_dof, skew_gamma=cfg.skew_gamma, max_daily_move=cfg.max_daily_move, seed=cfg.seed,
              drift_sd_annual=cfg.drift_uncertainty_sd, sv_persistence=cfg.sv_persistence,
              sv_logvol_sd=cfg.sv_logvol_sd, sv_leverage=cfg.sv_leverage)
    sim = simulate(mu_annual=np.full(T, expected.final), **kw)
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
           "label_const7": rating7.label, "score_const7": rating7.score}
    fund_level = d["fund"].values
    for h in HORIZONS:
        if pos + h >= len(d):
            continue
        start = fund_level[pos]
        window = fund_level[pos + 1: pos + h + 1] / start
        y = float(np.log(window[-1]))
        x, x7 = np.log(sim.terminal[h]), np.log(sim7.terminal[h])
        q5, q25, q75, q95 = np.percentile(x, [5, 25, 75, 95])
        m, s = naive_leveraged_lognormal(macro.vix, macro.rf_3m, annual_cost, h, cfg.leverage_target)
        tbill = float(np.prod(1.0 + d["tbill_ret"].values[pos + 1: pos + h + 1]) - 1.0)
        rec.update({
            f"ret_{h}": float(window[-1] - 1.0), f"tbill_{h}": tbill, f"excess_{h}": float(window[-1] - 1.0 - tbill),
            f"sp_{h}": float(d["tr"].values[pos + h] / d["tr"].values[pos] - 1.0),
            f"pit_{h}": float(np.mean(x <= y)), f"in90_{h}": bool(q5 <= y <= q95), f"in50_{h}": bool(q25 <= y <= q75),
            f"crps_model_{h}": crps_sample(x, y), f"crps_const7_{h}": crps_sample(x7, y),
            f"crps_naive_{h}": crps_normal(m, s, y),
            f"pred_dd20_{h}": float(np.mean(sim.path_min[h] <= 0.8)), f"real_dd20_{h}": float(window.min() <= 0.8),
        })
    return rec


# ---------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------
def _ci(lo: float, hi: float, fmt: str = "{:+.1%}") -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[n/a]"
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def _group_diff(label: str):
    def stat(a: np.ndarray) -> float:
        is_l = a[:, 0] > 0.5
        if is_l.sum() == 0 or (~is_l).sum() == 0:
            return float("nan")
        return float(a[is_l, 1].mean() - a[~is_l, 1].mean())
    return stat


def rating_section(o: pd.DataFrame) -> List[str]:
    out = []
    h = RATING_H
    ev = o.dropna(subset=[f"excess_{h}"])
    block = math.ceil(h / 21)
    out.append(f"Forward 6-month SPXL return by the rating given ({len(ev)} month-ends with a full 6 months after; "
               f"consecutive windows overlap, about {effective_n(ev['pos'], h):.0f} independent)")
    rows = []
    for lab in LABELS:
        g = ev[ev["label"] == lab]
        if len(g) == 0:
            rows.append({"rating": lab, "months": 0, "share": "0%"})
            continue
        diff_lo, diff_hi = block_bootstrap(np.column_stack([(ev["label"] == lab).astype(float), ev[f"excess_{h}"]]),
                                           block, _group_diff(lab))
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
        a = np.column_stack([e["score"], e[f"excess_{hh}"]])
        lo, hi = block_bootstrap(a, math.ceil(hh / 21), spearman)
        a7 = np.column_stack([e["score_const7"], e[f"excess_{hh}"]])
        rows.append({"horizon": f"{hh // 21}M", "n": len(e), "independent": f"{effective_n(e['pos'], hh):.0f}",
                     "model score": f"{spearman(a):+.2f} {_ci(lo, hi, '{:+.2f}')}",
                     "score with constant 7% drift": f"{spearman(a7):+.2f}"})
    out.append(pd.DataFrame(rows).to_string(index=False))

    out.append("\nForward 6-month SPXL excess return by score quintile")
    q = pd.qcut(ev["score"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
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


def calibration_section(o: pd.DataFrame) -> List[str]:
    out = ["Calibration of the full model (targets: mean PIT 0.50, 90% band 90%, 50% band 50%) [90% CI, overlap-aware]"]
    rows = []
    for h in HORIZONS:
        e = o.dropna(subset=[f"pit_{h}"])
        block = max(1, math.ceil(h / 21))
        pit_ci = block_bootstrap(e[f"pit_{h}"].values, block)
        c90_ci = block_bootstrap(e[f"in90_{h}"].astype(float).values, block)
        rec = {"horizon": {5: "1W", 10: "2W", 21: "1M", 63: "3M", 126: "6M"}[h], "n": len(e),
               "independent": f"{effective_n(e['pos'], h):.0f}",
               "mean PIT": f"{e[f'pit_{h}'].mean():.2f} {_ci(*pit_ci, '{:.2f}')}",
               "in 90% band": f"{e[f'in90_{h}'].mean():.0%} {_ci(*c90_ci, '{:.0%}')}",
               "in 50% band": f"{e[f'in50_{h}'].mean():.0%}",
               "below 5% / above 95%": f"{(e[f'pit_{h}'] < 0.05).mean():.0%} / {(e[f'pit_{h}'] > 0.95).mean():.0%}"}
        if h >= 21:
            rec["P(-20% dip) pred/real"] = f"{e[f'pred_dd20_{h}'].mean():.0%} / {e[f'real_dd20_{h}'].mean():.0%}"
        rows.append(rec)
    out.append(pd.DataFrame(rows).fillna("").to_string(index=False))
    return out


def skill_section(o: pd.DataFrame) -> List[str]:
    out = ["Skill: CRPS of the SPXL log return (lower = better forecast); skill = 1 - CRPS(model)/CRPS(benchmark), "
           "positive = the model is better [90% CI]"]
    rows = []
    for h in HORIZONS:
        e = o.dropna(subset=[f"crps_model_{h}"])
        block = max(1, math.ceil(h / 21))
        rec = {"horizon": {5: "1W", 10: "2W", 21: "1M", 63: "3M", 126: "6M"}[h], "n": len(e),
               "CRPS model": f"{e[f'crps_model_{h}'].mean():.4f}"}
        for bench, label in (("naive", "vs naive lognormal"), ("const7", "vs constant 7% drift")):
            a = np.column_stack([e[f"crps_model_{h}"], e[f"crps_{bench}_{h}"]])
            sk = 1.0 - a[:, 0].mean() / a[:, 1].mean()
            lo, hi = block_bootstrap(a, block, lambda x: 1.0 - x[:, 0].mean() / x[:, 1].mean())
            rec[label] = f"{sk:+.1%} {_ci(lo, hi)}"
        rows.append(rec)
    out.append(pd.DataFrame(rows).to_string(index=False))
    for era, sel in (("1990-2007", o["date"] < "2008-01-01"), ("2008-2026", o["date"] >= "2008-01-01")):
        e = o[sel].dropna(subset=[f"crps_model_{RATING_H}"])
        if len(e):
            out.append(f"  6M skill {era}: vs naive {1 - e[f'crps_model_{RATING_H}'].mean() / e[f'crps_naive_{RATING_H}'].mean():+.1%}, "
                       f"vs constant 7% {1 - e[f'crps_model_{RATING_H}'].mean() / e[f'crps_const7_{RATING_H}'].mean():+.1%} "
                       f"({len(e)} months)")
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
    val = validate_synthetic(d)
    positions = month_end_positions(d.index, args.start)
    print(f"{len(positions)} month-ends {d.index[positions[0]].date()} .. {d.index[positions[-1]].date()}, "
          f"{args.paths} paths, term structure before 2008: {args.term_structure}", flush=True)
    rows, t0 = [], time.time()
    for i, pos in enumerate(positions):
        inp = inputs_at(d, pos, inputs, cfg, args.term_structure, fits)
        if inp is None:
            continue
        rows.append(run_origin(d, pos, inp, cfg))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(positions)} ({time.time() - t0:.0f}s)", flush=True)
    o = pd.DataFrame(rows)
    o.to_csv(out_dir / "rating_backtest.csv", index=False)

    lines = [f"SPXLcast rating backtest, run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"{len(o)} month-ends {o['date'].iloc[0]:%Y-%m} .. {o['date'].iloc[-1]:%Y-%m}; {args.paths} paths per "
             f"simulation; earnings lagged {EARNINGS_LAG_MONTHS} months; inflation: breakeven from 2003, trailing CPI before",
             f"VIX term structure before 2008: {args.term_structure}"
             + (f" (log fit on {fits['vix6m'][3]} days: R2 {fits['vix3m'][2]:.2f} for VIX3M, {fits['vix6m'][2]:.2f} for VIX6M)"
                if args.term_structure == "impute" else ""),
             f"Synthetic 3x fund vs SPXL {SPXL_FROM:%Y}-now ({val['days']} days): daily correlation {val['corr']:.4f}, "
             f"growth {val['syn_growth']:+.1%}/yr vs {val['spxl_growth']:+.1%}/yr (gap {val['gap_per_year']:+.2%}/yr), "
             f"tracking {val['tracking_sd_annual']:.1%}/yr",
             f"Labels given: " + ", ".join(f"{lab} {(o['label'] == lab).mean():.0%}" for lab in LABELS)
             + f"; with a constant 7% drift: " + ", ".join(f"{lab} {(o['label_const7'] == lab).mean():.0%}" for lab in LABELS),
             ""]
    for section in (rating_section(o), [""], threshold_section(o), [""], strategy_section(o, d), [""],
                    calibration_section(o), [""], skill_section(o)):
        lines.extend(section)
    report = "\n".join(lines)
    (out_dir / "rating_backtest.txt").write_text(report + "\n", encoding="utf-8")
    plot(o, cfg, out_dir / "rating_backtest.png")
    print(report)
    print(f"\nwrote {out_dir / 'rating_backtest.csv'}, rating_backtest.txt, rating_backtest.png "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--start", default="1990-01")
    ap.add_argument("--term-structure", choices=("impute", "flat"), default="impute")
    ap.add_argument("--refresh", action="store_true", help="re-download the input data")
    ap.add_argument("--out", default=str(ROOT / "output" / "backtest"))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    run(ap.parse_args())
