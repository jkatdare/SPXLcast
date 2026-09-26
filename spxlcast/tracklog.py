"""Live track record: append each run's forecast to a CSV, and later score it against what
actually happened.

Every row stores the inputs that drove the forecast and the predicted quantiles of the SPXL price
at each horizon. ``score_log`` then looks up the realised price ``h`` sessions after each row's
spot date and reports, per horizon, where realised outcomes fell in the predicted distribution
(PIT), how often the 5-95% and 25-75% bands covered them, and whether the predicted chances of
touching -20% / +20% matched reality. It also measures whether the news score had any relation to
the next two weeks of returns, which is the calibration the sentiment channel currently lacks.

Each row also records the model version, the build (git commit) and any data problems of that run
(``data_flags``, empty when clean), so results can be scored per model version and runs on bad
data can be told apart.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MODEL_VERSION, Config
from .env import build_id

LOG_QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
DEFAULT_LOG = os.path.join("logs", "forecast_log.csv")


def _h(h: int, name: str) -> str:
    return f"h{h}_{name}"


def forecast_row(fc) -> Dict[str, object]:
    """Flatten a Forecast into one CSV row."""
    sim = fc.sim
    dq = fc.data_quality()
    row: Dict[str, object] = {
        "run_at": fc.snap.asof.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spot_date": fc.spot_date,
        "spot_status": fc.spot_status,
        "model_version": MODEL_VERSION,
        "build": build_id(),
        "data_flags": ";".join(dq["flags"]),
        "fred_series": dq["fred_series"],
        "news_fetched": "" if dq["news_fetched"] is None else dq["news_fetched"],
        "spot": round(fc.spot, 4),
        "rating": fc.rating.label,
        "score": round(fc.rating.score, 4),
        "score_se": round(fc.rating.score_se, 4),
        "conviction": fc.rating.conviction,
        "index_drift": round(fc.expected.final, 5),
        "rf_3m": round(fc.macro.rf_3m, 5),
        "annual_cost": round(fc.etf.annual_cost, 5),
        "vix": fc.macro.vix,
        "sentiment_score": round(fc.sentiment.score, 4) if fc.sentiment else "",
        "sentiment_n": fc.sentiment.n_used if fc.sentiment else "",
        "horizons": " ".join(str(h) for h in sim.horizons),
    }
    for h in sim.horizons:
        row[_h(h, "vol")] = round(fc.vol.total_vol(h), 5)
        q = sim.quantiles(h, LOG_QUANTILES)
        for lvl in LOG_QUANTILES:
            row[_h(h, f"q{lvl:02d}")] = round(q[float(lvl)], 4)
        s = sim.summary(h)
        row[_h(h, "p_up")] = round(s["p_positive"], 4)
        row[_h(h, "p_beat_rf")] = round(s["p_beat_rf"], 4)
        row[_h(h, "p_dd20")] = round(s["p_drawdown_20"], 4)
        row[_h(h, "p_up20")] = round(s["p_up_20"], 4)
        row[_h(h, "rf_growth")] = round(sim.rf_growth[h], 6)
    return row


def append_log(fc, path: str = DEFAULT_LOG) -> str:
    row = forecast_row(fc)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fieldnames = list(row.keys())
    if exists:
        with open(path, newline="", encoding="utf-8") as fh:
            existing = next(csv.reader(fh), None) or []
        # keep the file's column order; append any new columns at the end
        fieldnames = existing + [c for c in fieldnames if c not in existing]
        if fieldnames != existing:
            _rewrite_with_columns(path, fieldnames)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def _rewrite_with_columns(path: str, fieldnames: List[str]) -> None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in fieldnames:
        if c not in df.columns:
            df[c] = ""
    df[fieldnames].to_csv(path, index=False)


def load_log(path: str = DEFAULT_LOG) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    # text columns that pandas would otherwise mangle ("1.10" -> 1.1, a hex commit -> a float)
    df = pd.read_csv(path, dtype={"model_version": str, "build": str, "data_flags": str})
    if df.empty:
        return df
    df["run_at"] = pd.to_datetime(df["run_at"], utc=True)
    df["spot_date"] = pd.to_datetime(df["spot_date"])
    # one forecast per spot date: prefer a close over an intraday quote, then the latest run
    df["_close"] = (df["spot_status"] == "close").astype(int)
    df = df.sort_values(["spot_date", "_close", "run_at"]).groupby("spot_date", as_index=False).tail(1)
    return df.drop(columns="_close").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
@dataclass
class HorizonScore:
    horizon: int
    n: int
    mean_pit: float
    cov_5_95: float
    cov_25_75: float
    frac_below_5: float
    frac_above_95: float
    mean_realised_return: float
    mean_predicted_median_return: float
    pred_dd20: float
    real_dd20: float
    pred_up20: float
    real_up20: float


@dataclass
class ScoreReport:
    n_rows: int
    n_scoreable: int
    first_date: Optional[str]
    last_date: Optional[str]
    horizons: List[HorizonScore] = field(default_factory=list)
    by_rating: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sentiment_corr: Optional[float] = None
    sentiment_n: int = 0
    notes: List[str] = field(default_factory=list)
    versions: Dict[str, int] = field(default_factory=dict)   # rows per model version in the whole log
    model_version: Optional[str] = None                      # the version scored, when filtered


def _pit(realised: float, quantiles: Dict[int, float]) -> float:
    """Piecewise-linear CDF (in log price) through the stored quantiles."""
    lv = np.array(sorted(quantiles))
    qv = np.log(np.array([quantiles[int(k)] for k in lv]))
    x = np.log(realised)
    if x <= qv[0]:
        return lv[0] / 200.0
    if x >= qv[-1]:
        return 1.0 - (100 - lv[-1]) / 200.0
    return float(np.interp(x, qv, lv) / 100.0)


def score_log(path: str = DEFAULT_LOG, closes: Optional[pd.Series] = None, cfg: Optional[Config] = None,
              model_version: Optional[str] = None) -> ScoreReport:
    """Score the logged forecasts; ``model_version`` restricts it to rows from that version."""
    cfg = cfg or Config()
    df = load_log(path)
    report = ScoreReport(n_rows=int(len(df)), n_scoreable=0, first_date=None, last_date=None,
                         model_version=model_version)
    if df.empty:
        report.notes.append(f"no rows in {path}")
        return report
    if "model_version" in df:
        versions = df["model_version"].fillna("unknown").astype(str)
        report.versions = {str(k): int(v) for k, v in versions.value_counts().sort_index().items()}
        if model_version is not None:
            df = df[versions == model_version].reset_index(drop=True)
            report.n_rows = int(len(df))
            if df.empty:
                report.notes.append(f"no rows from model version {model_version} "
                                    f"(the log has {', '.join(report.versions)})")
                return report
        elif len(report.versions) > 1:
            report.notes.append(f"rows from {len(report.versions)} model versions are pooled "
                                f"({', '.join(report.versions)}); use --model-version to score one")
    report.first_date = str(df["spot_date"].min().date())
    report.last_date = str(df["spot_date"].max().date())
    if closes is None:
        from .cache import Cache
        from .data import fetch_prices
        prices, _ = fetch_prices([cfg.etf], cfg.history_period, Cache(cfg.cache_dir), cfg.price_ttl_hours, required=(cfg.etf,))
        closes = prices.get(cfg.etf, pd.DataFrame()).get("Close")
    if closes is None or len(closes) == 0:
        report.notes.append("no price history available to score against")
        return report
    closes = closes.dropna()
    idx = closes.index

    horizons = sorted({int(h) for hs in df["horizons"].astype(str) for h in hs.split()})
    scored_any = set()
    for h in horizons:
        recs = []
        for _, row in df.iterrows():
            qcols = {lvl: row.get(_h(h, f"q{lvl:02d}")) for lvl in LOG_QUANTILES}
            if any(pd.isna(v) for v in qcols.values()):
                continue
            pos = int(idx.searchsorted(row["spot_date"], side="right")) - 1
            if pos < 0 or idx[pos] != row["spot_date"] or pos + h >= len(idx):
                continue
            path_vals = closes.iloc[pos + 1: pos + h + 1].values
            realised = float(path_vals[-1])
            spot = float(row["spot"])
            recs.append({
                "pit": _pit(realised, {int(k): float(v) for k, v in qcols.items()}),
                "in_5_95": float(qcols[5]) <= realised <= float(qcols[95]),
                "in_25_75": float(qcols[25]) <= realised <= float(qcols[75]),
                "below_5": realised < float(qcols[5]),
                "above_95": realised > float(qcols[95]),
                "ret": realised / spot - 1.0,
                "pred_med": float(qcols[50]) / spot - 1.0,
                "pred_dd20": float(row.get(_h(h, "p_dd20"), np.nan)),
                "real_dd20": float(path_vals.min() <= 0.8 * spot),
                "pred_up20": float(row.get(_h(h, "p_up20"), np.nan)),
                "real_up20": float(path_vals.max() >= 1.2 * spot),
                "rating": row["rating"],
            })
            scored_any.add(row["spot_date"])
        if not recs:
            continue
        g = pd.DataFrame(recs)
        report.horizons.append(HorizonScore(
            horizon=h, n=len(g), mean_pit=float(g["pit"].mean()),
            cov_5_95=float(g["in_5_95"].mean()), cov_25_75=float(g["in_25_75"].mean()),
            frac_below_5=float(g["below_5"].mean()), frac_above_95=float(g["above_95"].mean()),
            mean_realised_return=float(g["ret"].mean()), mean_predicted_median_return=float(g["pred_med"].mean()),
            pred_dd20=float(g["pred_dd20"].mean()), real_dd20=float(g["real_dd20"].mean()),
            pred_up20=float(g["pred_up20"].mean()), real_up20=float(g["real_up20"].mean()),
        ))
        if h == cfg.rating_horizon:
            for label, gg in g.groupby("rating"):
                report.by_rating[str(label)] = {"n": int(len(gg)), "mean_return": float(gg["ret"].mean()),
                                                "p_positive": float((gg["ret"] > 0).mean())}
    report.n_scoreable = len(scored_any)

    # News sentiment vs the next 10 sessions of SPXL return.
    if "sentiment_score" in df:
        pairs = []
        for _, row in df.iterrows():
            s = row.get("sentiment_score")
            if pd.isna(s) or s == "":
                continue
            pos = int(idx.searchsorted(row["spot_date"], side="right")) - 1
            if pos < 0 or idx[pos] != row["spot_date"] or pos + 10 >= len(idx):
                continue
            pairs.append((float(s), float(closes.iloc[pos + 10] / closes.iloc[pos] - 1.0)))
        if len(pairs) >= 10:
            a = np.array(pairs)
            report.sentiment_corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        report.sentiment_n = len(pairs)
    if report.n_scoreable == 0:
        report.notes.append("nothing to score yet: the shortest horizon has not elapsed since the first logged date")
    return report
