"""Does the expected-return model predict forward S&P 500 returns? A 150-year test on Shiller's data.

The engine backtest (scripts/backtest_calibration.py) could only use a constant drift because the
live model's inputs (P/E, dividend yield, breakeven inflation) have no history in Yahoo. Shiller's
monthly dataset has them back to 1871, so here the model's drift is rebuilt month by month exactly
as spxlcast.fundamentals.expected_index_return would (earnings-yield model, dividend-growth model,
50/50 blend, capped valuation term; no regime penalties, whose inputs are not in the file) and
compared with the realised nominal total return over the next 1, 5 and 10 years.

Comparators: a constant drift, earnings yield alone, and 1/CAPE, each plus expected inflation.
Expected inflation is proxied by the trailing 10-year CPI growth (the 10-year breakeven the live
model uses only exists from 2003).

Usage:  py scripts/backtest_drift.py [--xls output/backtest/ie_data.xls]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spxlcast.config import Config   # noqa: E402

# Shiller's data moved to shillerdata.com (the Yale copy stopped updating in 2023). The download
# link carries a version stamp, so it is read from the page; the Yale URL is the last resort.
PAGE = "https://shillerdata.com/"
URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def shiller_url() -> str:
    import re
    import requests
    try:
        html = requests.get(PAGE, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
        m = re.search(r"(?:https?:)?//[^\"'\s]+/ie_data\.xls[^\"'\s]*", html)
        if m:
            return m.group(0) if m.group(0).startswith("http") else "https:" + m.group(0)
    except requests.RequestException:
        pass
    return URL


def load_shiller(path: Path) -> pd.DataFrame:
    if not path.exists():
        import requests
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(shiller_url(), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = pd.read_excel(path, sheet_name="Data", header=None, skiprows=8)
    df = raw.iloc[:, [0, 1, 2, 3, 4, 6, 9, 12]].copy()
    df.columns = ["date", "P", "D", "E", "CPI", "GS10", "real_tr", "CAPE"]
    df = df[pd.to_numeric(df["date"], errors="coerce").notna()].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    year = np.floor(df["date"]).astype(int)
    month = np.round((df["date"] - year) * 100).astype(int).clip(1, 12)
    df.index = pd.to_datetime({"year": year, "month": month, "day": 1})
    df = df.drop(columns="date").dropna(subset=["P", "D", "E", "CPI", "real_tr"])
    df["nominal_tr"] = df["real_tr"] * df["CPI"] / df["CPI"].iloc[-1]
    return df


def model_drift(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ep"] = df["E"] / df["P"]
    out["dy"] = df["D"] / df["P"]
    out["infl"] = (df["CPI"] / df["CPI"].shift(120)) ** (1 / 10) - 1      # trailing 10y CPI growth
    out["cape_yield"] = 1.0 / df["CAPE"]
    m_ey = out["ep"] + out["infl"]
    m_dg = out["dy"] + cfg.long_run_real_eps_growth + out["infl"]
    base = 0.5 * (m_ey + m_dg)
    valuation = ((out["ep"] - cfg.neutral_earnings_yield) * cfg.valuation_sensitivity).clip(
        -cfg.valuation_adj_cap, cfg.valuation_adj_cap)
    out["model"] = (base + valuation).clip(cfg.drift_floor, cfg.drift_cap)
    out["model_no_valuation"] = base.clip(cfg.drift_floor, cfg.drift_cap)
    # adaptive neutral: E/P relative to its own trailing 20-year mean (a regime-aware valuation term)
    neutral_adaptive = out["ep"].rolling(240, min_periods=120).mean()
    val_adaptive = ((out["ep"] - neutral_adaptive) * cfg.valuation_sensitivity).clip(
        -cfg.valuation_adj_cap, cfg.valuation_adj_cap)
    out["model_adaptive_valuation"] = (base + val_adaptive).clip(cfg.drift_floor, cfg.drift_cap)
    out["ep_plus_infl"] = m_ey
    out["dy_growth"] = m_dg
    out["cape_plus_infl"] = out["cape_yield"] + out["infl"]
    return out


def forward_returns(df: pd.DataFrame, years: int) -> pd.Series:
    n = 12 * years
    return (df["nominal_tr"].shift(-n) / df["nominal_tr"]) ** (1 / years) - 1


def evaluate(pred: pd.Series, real: pd.Series) -> dict:
    m = pd.concat([pred, real], axis=1).dropna()
    if len(m) < 24:
        return {"n": len(m)}
    err = m.iloc[:, 0] - m.iloc[:, 1]
    return {"n": int(len(m)), "corr": float(m.corr().iloc[0, 1]), "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "mean_pred": float(m.iloc[:, 0].mean()), "mean_real": float(m.iloc[:, 1].mean())}


def run(xls: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    df = load_shiller(xls)
    d = model_drift(df, cfg)
    d = d.dropna(subset=["infl"])            # needs 10 years of CPI history
    print(f"Shiller sample {d.index[0]:%Y-%m} .. {d.index[-1]:%Y-%m}, {len(d)} months")
    constant = pd.Series(0.07, index=d.index)
    candidates = {"model (blend + valuation)": d["model"], "blend without valuation": d["model_no_valuation"],
                  "blend + adaptive valuation (20y)": d["model_adaptive_valuation"],
                  "E/P + inflation": d["ep_plus_infl"], "D/P + 3% + inflation": d["dy_growth"],
                  "1/CAPE + inflation": d["cape_plus_infl"], "constant 7%": constant}
    rows = []
    for years in (1, 5, 10):
        real = forward_returns(df, years).reindex(d.index)
        for name, pred in candidates.items():
            rows.append({"horizon_years": years, "predictor": name, **evaluate(pred, real)})
        # by era for the model only
        for era, (a, b) in {"1881-1945": ("1881", "1945"), "1946-1989": ("1946", "1989"), "1990-2023": ("1990", "2023")}.items():
            sl = slice(a, b)
            rows.append({"horizon_years": years, "predictor": f"model, {era}", **evaluate(d["model"][sl], real[sl])})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "drift_backtest.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\nPredicted vs realised forward nominal total return (annualised). corr = information, "
          "rmse/bias = calibration.")
    print(res.to_string(index=False))

    recent = d.loc["2016-09":"2023-09", "model"]
    print(f"\nModel drift over 2016-09..2023-09: mean {recent.mean():.2%} (min {recent.min():.2%}, max {recent.max():.2%}); "
          f"realised S&P total return over that window was about 12-15%/yr, i.e. the engine backtest's upward PIT "
          f"bias is what this drift would have produced too.")
    latest = d.iloc[-1]
    print(f"Latest Shiller month {d.index[-1]:%Y-%m}: E/P {latest['ep']:.2%}, D/P {latest['dy']:.2%}, "
          f"infl proxy {latest['infl']:.2%}, model drift {latest['model']:.2%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    real10 = forward_returns(df, 10).reindex(d.index)
    axes[0].plot(d.index, d["model"], label="model drift", lw=1.2)
    axes[0].plot(d.index, real10, label="realised next-10y return", lw=1.0, alpha=0.8)
    axes[0].set_title("Model drift vs realised 10-year nominal total return")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    m = pd.concat([d["model"], real10], axis=1).dropna()
    axes[1].scatter(m.iloc[:, 0], m.iloc[:, 1], s=5, alpha=0.4)
    lim = [min(m.min()), max(m.max())]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    axes[1].set_xlabel("model drift"); axes[1].set_ylabel("realised 10y return")
    axes[1].set_title("10-year horizon"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drift_backtest.png", dpi=120)
    print(f"\nwrote {out_dir / 'drift_backtest.csv'} and drift_backtest.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", default=str(ROOT / "output" / "backtest" / "ie_data.xls"))
    ap.add_argument("--out", default=str(ROOT / "output" / "backtest"))
    a = ap.parse_args()
    run(Path(a.xls), Path(a.out))
