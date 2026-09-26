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

Scores are reported honestly for overlapping windows: forecasts a day apart share most of their
outcome window, so each horizon reports the number of independent outcomes the rows amount to and
90% intervals from a block bootstrap (only once there are at least three). The whole distribution
is also graded with CRPS against a naive benchmark (a lognormal at the raw VIX with a T-bill drift,
see ``evaluation.naive_leveraged_lognormal``). With the run archive (``archive_dir``) the fine
103-point percentile grid of each run is used instead of the nine logged quantiles.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MODEL_VERSION, Config
from .env import build_id
from .evaluation import (block_bootstrap, crps_quantiles, effective_n, naive_leveraged_lognormal,
                         normal_quantiles, pit_from_quantiles)

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
    n_eff: float = float("nan")                  # independent outcomes the overlapping rows amount to
    mean_pit_ci: Tuple[float, float] = (float("nan"), float("nan"))     # 90% block-bootstrap intervals
    cov_5_95_ci: Tuple[float, float] = (float("nan"), float("nan"))
    crps_model: float = float("nan")             # mean CRPS of the log return (lower = better)
    crps_naive: float = float("nan")
    crps_skill: float = float("nan")             # 1 - model / naive: positive = better than the benchmark
    crps_skill_ci: Tuple[float, float] = (float("nan"), float("nan"))
    n_fine: int = 0                              # rows scored on the archived 103-point grid


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
    lv = sorted(quantiles)
    return pit_from_quantiles(lv, np.log([quantiles[int(k)] for k in lv]), float(np.log(realised)))


def _archived_grids(archive_dir: str, run_at) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Horizon -> (percent levels, prices) of the run's fine terminal-price grid, from the archive."""
    t = pd.Timestamp(run_at)
    t = t.tz_convert("UTC") if t.tzinfo is not None else t
    path = os.path.join(archive_dir, "runs", t.strftime("%Y-%m-%d"), t.strftime("%H%M%SZ") + ".json.gz")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    grids = {}
    for h, f in ((rec.get("forecast") or {}).get("forecast") or {}).items():
        g = f.get("grid")
        if g and g.get("terminal"):
            grids[int(h)] = (np.asarray(g["percentiles"], dtype=float), np.asarray(g["terminal"], dtype=float))
    return grids


def score_log(path: str = DEFAULT_LOG, closes: Optional[pd.Series] = None, cfg: Optional[Config] = None,
              model_version: Optional[str] = None, archive_dir: Optional[str] = None) -> ScoreReport:
    """Score the logged forecasts; ``model_version`` restricts it to rows from that version and
    ``archive_dir`` (the run archive) supplies each run's fine percentile grid when present."""
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
    grid_cache: Dict[str, Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]] = {}
    nan2 = (float("nan"), float("nan"))
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
            grid = None
            if archive_dir:
                key = str(row["run_at"])
                if key not in grid_cache:
                    grid_cache[key] = _archived_grids(archive_dir, row["run_at"])
                grid = (grid_cache[key] or {}).get(h)
            if grid is not None:
                levels, prices = grid
            else:
                levels = np.asarray(LOG_QUANTILES, dtype=float)
                prices = np.asarray([float(qcols[lvl]) for lvl in LOG_QUANTILES])
            qv = np.log(prices / spot)                 # everything in log return from the spot
            y = math.log(realised / spot)
            q5, q25, q50, q75, q95 = np.interp([5, 25, 50, 75, 95], levels, qv)

            crps_naive = float("nan")
            inputs = [row.get("vix"), row.get("rf_3m"), row.get("annual_cost")]
            if all(v is not None and np.isfinite(float(v)) for v in inputs):
                m, s = naive_leveraged_lognormal(*(float(v) for v in inputs), h, cfg.leverage_target)
                crps_naive = crps_quantiles(levels, normal_quantiles(m, s, levels), y)   # same grid: comparable
            recs.append({
                "pos": pos,
                "pit": pit_from_quantiles(levels, qv, y),
                "in_5_95": q5 <= y <= q95,
                "in_25_75": q25 <= y <= q75,
                "below_5": y < q5,
                "above_95": y > q95,
                "ret": realised / spot - 1.0,
                "pred_med": math.exp(q50) - 1.0,
                "pred_dd20": float(row.get(_h(h, "p_dd20"), np.nan)),
                "real_dd20": float(path_vals.min() <= 0.8 * spot),
                "pred_up20": float(row.get(_h(h, "p_up20"), np.nan)),
                "real_up20": float(path_vals.max() >= 1.2 * spot),
                "rating": row["rating"],
                "crps_model": crps_quantiles(levels, qv, y),
                "crps_naive": crps_naive,
                "fine": grid is not None,
            })
            scored_any.add(row["spot_date"])
        if not recs:
            continue
        g = pd.DataFrame(recs).sort_values("pos").reset_index(drop=True)
        n_eff = effective_n(g["pos"].values, h)
        enough = n_eff >= 3                    # intervals from fewer independent outcomes mean nothing
        # a bootstrap block spans one outcome window: h sessions, in rows at the log's own spacing
        spacing = float(np.median(np.diff(g["pos"].values))) if len(g) > 1 else 1.0
        block = max(1, math.ceil(h / max(spacing, 1.0)))
        both = g.dropna(subset=["crps_naive"])
        crps_m = float(both["crps_model"].mean()) if len(both) else float("nan")
        crps_n = float(both["crps_naive"].mean()) if len(both) else float("nan")
        report.horizons.append(HorizonScore(
            horizon=h, n=len(g), mean_pit=float(g["pit"].mean()),
            cov_5_95=float(g["in_5_95"].mean()), cov_25_75=float(g["in_25_75"].mean()),
            frac_below_5=float(g["below_5"].mean()), frac_above_95=float(g["above_95"].mean()),
            mean_realised_return=float(g["ret"].mean()), mean_predicted_median_return=float(g["pred_med"].mean()),
            pred_dd20=float(g["pred_dd20"].mean()), real_dd20=float(g["real_dd20"].mean()),
            pred_up20=float(g["pred_up20"].mean()), real_up20=float(g["real_up20"].mean()),
            n_eff=n_eff,
            mean_pit_ci=block_bootstrap(g["pit"].values, block) if enough else nan2,
            cov_5_95_ci=block_bootstrap(g["in_5_95"].astype(float).values, block) if enough else nan2,
            crps_model=crps_m, crps_naive=crps_n,
            crps_skill=1.0 - crps_m / crps_n if crps_n > 0 else float("nan"),
            crps_skill_ci=(block_bootstrap(both[["crps_model", "crps_naive"]].values, block,
                                           lambda a: 1.0 - a[:, 0].mean() / a[:, 1].mean())
                           if enough and len(both) else nan2),
            n_fine=int(g["fine"].sum()),
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
            if a[:, 0].std() > 0 and a[:, 1].std() > 0:      # a constant score has no correlation
                report.sentiment_corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        report.sentiment_n = len(pairs)
    if report.n_scoreable == 0:
        report.notes.append("nothing to score yet: the shortest horizon has not elapsed since the first logged date")
    return report
