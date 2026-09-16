"""Historical calibration backtest of the SPXLcast simulation engine.

At every month-end t in the sample the model inputs are built from data available at t (spot,
VIX / VIX3M / VIX6M term structure, 3-month bill), the simulation is run with the project's own
``simulate`` and ``vol_term_structure``, and the realised outcome over the next 21 / 63 / 126
trading days is scored:

* PIT = percentile of the realised price in the simulated distribution (uniform if calibrated)
* coverage of the 5-95% and 25-75% bands
* predicted vs realised frequency of touching -10% / -20% / +10% / +20% from spot

The fundamentals-based drift history is not reproducible (no historical P/E or breakeven series),
so a constant drift is used and stated. The S&P returned ~15%/yr over the sample, so a mean PIT
above 0.5 is expected for both SPY and SPXL; what matters for the SPXL model is dispersion
(coverage close to nominal, variance ratio near 1) and touch frequencies.

Usage:  py scripts/backtest_calibration.py [--drift 0.07] [--paths 5000] [--years 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spxlcast.config import Config                                   # noqa: E402
from spxlcast.data import MarketSnapshot                             # noqa: E402
from spxlcast.fundamentals import MacroState, discount_to_bey, vol_term_structure  # noqa: E402
from spxlcast.montecarlo import simulate                             # noqa: E402

HORIZONS = [21, 63, 126]
TOUCH = {"dn10": 0.90, "dn20": 0.80, "up10": 1.10, "up20": 1.20}
TICKERS = ["SPXL", "SPY", "^VIX", "^VIX3M", "^VIX6M", "^IRX"]

# name -> (vrp haircuts, dof, skew gamma, swap spread)
VARIANTS = {
    "previous (vrp 2 flat, symmetric, spread 0.40%)": (2.0, 4.0, 1.0, 0.004),
    "current  (vrp 2/4/6, skew 0.9, spread 0.75%)": ((2.0, 4.0, 6.0), 4.0, 0.9, 0.0075),
    "current without skew": ((2.0, 4.0, 6.0), 4.0, 1.0, 0.0075),
}


def load_closes(years: int, cache: Path) -> pd.DataFrame:
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        import yfinance as yf
        raw = yf.download(TICKERS, period=f"{years}y", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        df = pd.DataFrame({t: raw[t]["Close"] for t in TICKERS})
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df.to_pickle(cache)
    df = df.sort_index().dropna(subset=["SPXL", "SPY"])
    for c in ["^VIX", "^VIX3M", "^VIX6M", "^IRX"]:
        df[c] = df[c].ffill()
    return df.dropna()


def month_ends(idx: pd.DatetimeIndex, max_h: int):
    s = pd.Series(np.arange(len(idx)), index=idx)
    pos = s.groupby([idx.year, idx.month]).max().values
    return [int(p) for p in pos if p + max_h < len(idx)]


def macro_at(row: pd.Series) -> MacroState:
    rf = discount_to_bey(float(row["^IRX"]) / 100.0)
    return MacroState(rf_3m=rf, y2=None, y5=None, y10=rf, y30=None, curve_10y_3m=0.0, breakeven_10y=None,
                      real_10y=None, sofr=None, hy_oas=None, unemployment=None, unemployment_sahm_gap=None,
                      cpi_yoy=None, vix=float(row["^VIX"]), vix3m=float(row["^VIX3M"]), vix6m=float(row["^VIX6M"]),
                      vvix=None, skew=None, dxy=None, oil=None, gold=None)


def run(drift: float, n_paths: int, years: int, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_closes(years, out_dir / f"closes_{years}y.pkl")
    idx = df.index
    positions = month_ends(idx, max(HORIZONS))
    print(f"sample {idx[0].date()} .. {idx[-1].date()}, {len(positions)} month-end origins, drift {drift:.1%}, "
          f"{n_paths} paths")
    rows = []
    t0 = time.time()
    for vname, (vrp, dof, gamma, spread) in VARIANTS.items():
        cfg = Config(use_fred=False, use_news=False, vrp_vol_points=vrp, t_dof=dof, skew_gamma=gamma, swap_spread=spread)
        for pos in positions:
            row = df.iloc[pos]
            macro = macro_at(row)
            snap = MarketSnapshot(asof=datetime.now(timezone.utc),
                                  prices={"SPY": pd.DataFrame({"Close": df["SPY"].iloc[: pos + 1]})})
            vol = vol_term_structure(macro, snap, cfg, max(HORIZONS))
            cost = cfg.expense_ratio_default + 2.0 * (macro.rf_3m + spread)
            mu = np.full(max(HORIZONS), drift)
            sims = {
                "SPXL": simulate(float(row["SPXL"]), mu, vol.daily, 3.0, cost / 252.0, 0.0, macro.rf_3m, HORIZONS,
                                 n_paths=n_paths, dof=dof, skew_gamma=gamma, seed=42),
                "SPY": simulate(float(row["SPY"]), mu, vol.daily, 1.0, 0.0, 0.0, macro.rf_3m, HORIZONS,
                                n_paths=n_paths, dof=dof, skew_gamma=gamma, seed=42),
            }
            for asset, sim in sims.items():
                for h in HORIZONS:
                    path = df[asset].iloc[pos + 1: pos + h + 1].values
                    realised, pmin, pmax = float(path[-1]), float(path.min()), float(path.max())
                    term = sim.terminal[h]
                    rec = {"variant": vname, "asset": asset, "horizon": h, "date": idx[pos],
                           "pit": sim.percentile_of_price(realised, h) / 100.0,
                           "logret_realised": np.log(realised / sim.spot),
                           "model_median_logret": np.log(np.median(term) / sim.spot),
                           "model_sd_logret": float(np.std(np.log(term / sim.spot)))}
                    for name, lvl in TOUCH.items():
                        level = lvl * sim.spot
                        if lvl < 1:
                            rec[f"pred_{name}"], rec[f"real_{name}"] = sim.prob_touch_below(level, h), float(pmin <= level)
                        else:
                            rec[f"pred_{name}"], rec[f"real_{name}"] = sim.prob_touch_above(level, h), float(pmax >= level)
                    rows.append(rec)
        print(f"  {vname}: done ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "observations.csv", index=False)

    summ = []
    for (vname, asset, h), g in res.groupby(["variant", "asset", "horizon"], sort=False):
        pit = g["pit"].values
        rec = {"variant": vname, "asset": asset, "h": h, "n": len(pit), "mean_pit": pit.mean(),
               "ks": stats.kstest(pit, "uniform").statistic,
               "cov_5_95": np.mean((pit >= 0.05) & (pit <= 0.95)), "cov_25_75": np.mean((pit >= 0.25) & (pit <= 0.75)),
               "below_5": np.mean(pit < 0.05), "above_95": np.mean(pit > 0.95),
               "var_ratio": np.var(g["logret_realised"] - g["model_median_logret"]) / np.mean(g["model_sd_logret"] ** 2)}
        for name in TOUCH:
            rec[f"{name} pred/real"] = f"{g[f'pred_{name}'].mean():.2f}/{g[f'real_{name}'].mean():.2f}"
        summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
    print("\nNominal: cov_5_95 0.90, cov_25_75 0.50, below_5/above_95 0.05, var_ratio 1.0 (realised / model variance)")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=0.07)
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "output" / "backtest"))
    a = ap.parse_args()
    run(a.drift, a.paths, a.years, Path(a.out))
