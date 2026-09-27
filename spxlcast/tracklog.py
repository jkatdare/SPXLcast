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
90% intervals on that many observations (see ``evaluation.mean_interval``), shown from
``MIN_INDEPENDENT`` independent outcomes on. The whole distribution is also graded with CRPS
against a naive benchmark (a lognormal at the raw VIX with a T-bill drift, see
``evaluation.naive_leveraged_lognormal``). With the run archive (``archive_dir``) the fine
103-point percentile grid of each run is used instead of the nine logged quantiles.

Yahoo back-adjusts its closes for every later distribution and split, while a row's spot and
quantiles are as traded, so each row is scored on the ratio of later closes to the close on its
own date, which leaves a closed window's score unchanged by anything paid or split afterwards.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MODEL_VERSION, Config, nyse_session
from .env import build_id
from .evaluation import (crps_quantiles, effective_n, mean_interval, naive_leveraged_lognormal, normal_quantiles,
                         pit_from_quantiles, proportion_interval, skill_interval)

LOG_QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
DEFAULT_LOG = os.path.join("logs", "forecast_log.csv")
# Intervals are shown from three independent outcomes: below it they are too wide to read (the
# coverage the 90% intervals reach is in the ``evaluation`` docstring).
MIN_INDEPENDENT = 3
# columns read as text; every other column is numeric, and a malformed cell becomes NaN
TEXT_COLUMNS = ("run_at", "spot_date", "spot_status", "model_version", "build", "data_flags", "rating",
                "conviction", "horizons")
# the price download failed: `score` exits non-zero, so the job keeps the last good track record
NO_PRICES = "no price history available to score against"


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
    fieldnames = list(row.keys())
    existing: List[str] = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        _set_aside_torn_tail(path)
        # utf-8-sig: a BOM left by a hand edit is not part of the first column's name
        with open(path, newline="", encoding="utf-8-sig") as fh:
            existing = _header(csv.reader(fh))
    if existing:
        # keep the file's column order; append any new columns at the end
        fieldnames = existing + [c for c in fieldnames if c not in existing]
        if fieldnames != existing:
            _rewrite_with_columns(path, fieldnames)
    with open(path, "a" if existing else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerow(row)
    return path


def _header(rows) -> List[str]:
    """The first row with any content: a hand edit can leave blank lines above the header, and a file
    of blank lines only holds no track record."""
    return next((r for r in rows if any(c.strip() for c in r)), [])


def _set_aside_torn_tail(path: str) -> None:
    """A write cut short leaves the last row without its line end, and the next row would be glued
    onto it. A row with all its fields only gets its line end; a partial one is moved to
    ``path + ".partial"``, so a cut-off number is never scored."""
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return
        fh.seek(0)
        data = fh.read()
    if data.endswith(b"\r"):        # a whole row: a CRLF cut after its CR, or a log with CR-only line ends
        with open(path, "ab") as fh:
            fh.write(b"\n")
        return
    cut = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1

    def width(line: bytes) -> int:
        return len(next(csv.reader([line.decode("utf-8-sig", "replace")]), []))

    header = _header(csv.reader(io.StringIO(data.decode("utf-8-sig", "replace"), newline="")))
    if cut and width(data[cut:]) == len(header):
        with open(path, "ab") as fh:
            fh.write(b"\r\n")
        return
    with open(path + ".partial", "ab") as fh:
        fh.write(data[cut:] + b"\n")
    with open(path, "r+b") as fh:
        fh.truncate(cut)


def _rewrite_with_columns(path: str, fieldnames: List[str]) -> None:
    """Rewrite the log under a wider header. The new file is written aside and swapped in, so an
    interrupted write never truncates the track record; a row with more fields than the header
    (damage) is moved to ``path + ".partial"`` rather than dropped or misaligned."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        # strict: a stray quote (a hand edit) fails the run rather than merging every later row into one cell
        header, *rows = [r for r in csv.reader(fh, strict=True) if any(c.strip() for c in r)] or [[]]
    col = {c: i for i, c in enumerate(header)}
    bad = [r for r in rows if len(r) > len(header)]
    tmp = f"{path}.{os.urandom(6).hex()}.tmp"      # unique: an overlapping run never publishes this one's
    try:
        with open(tmp, "x", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(fieldnames)
            for r in rows:
                if r and len(r) <= len(header):
                    writer.writerow([r[col[c]] if c in col and col[c] < len(r) else "" for c in fieldnames])
        if bad:
            with open(path + ".partial", "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(bad)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_log(path: str = DEFAULT_LOG) -> pd.DataFrame:
    try:
        # a line with more fields than the header (damage) is skipped with a warning, not fatal
        df = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, encoding="utf-8-sig", on_bad_lines="warn")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if df.empty or not {"run_at", "spot_date", "spot_status"} <= set(df.columns):
        return pd.DataFrame()
    num = [c for c in df.columns if c not in TEXT_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])]
    if num:
        df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.copy()      # pandas 3 reads one block per column, and adding a column to that warns
    # the format the log writes, stated: inferring it from the first row turns other ISO forms into NaT
    df["run_at"] = pd.to_datetime(df["run_at"], utc=True, errors="coerce", format="ISO8601")
    df["spot_date"] = pd.to_datetime(df["spot_date"], errors="coerce")
    df = df.dropna(subset=["spot_date"])
    # One forecast per spot date: a run made on that New York day, a close before an intraday
    # quote, then the latest. A run on a later day (before the next open, or on an exchange
    # holiday) logs the same close but has seen the news since, so it stands in only when nothing
    # ran on the day itself, and then the earliest one.
    same_day = (df["run_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
                == df["spot_date"])
    order = df["run_at"].rank(method="first")
    df["_same"] = same_day.astype(int)
    df["_close"] = (df["spot_status"] == "close").astype(int)
    df["_order"] = np.where(same_day, order, -order)
    df = (df.sort_values(["spot_date", "_same", "_close", "_order"], na_position="first")
          .groupby("spot_date", as_index=False).tail(1))
    return df.drop(columns=["_same", "_close", "_order"]).reset_index(drop=True)


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
    mean_pit_ci: Tuple[float, float] = (float("nan"), float("nan"))     # 90% intervals on n_eff
    cov_5_95_ci: Tuple[float, float] = (float("nan"), float("nan"))
    crps_model: float = float("nan")             # mean CRPS of the log return (lower = better)
    crps_naive: float = float("nan")
    crps_skill: float = float("nan")             # 1 - model / naive: positive = better than the benchmark
    crps_skill_ci: Tuple[float, float] = (float("nan"), float("nan"))
    n_fine: int = 0                              # rows scored on the archived 103-point grid
    # rows with the benchmark's inputs (logged VIX, T-bill, cost): both CRPS columns and the skill
    # are means over these rows, so they stay comparable; with none, CRPS model covers all n rows
    n_crps: int = 0


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
    """Horizon -> (percent levels, prices) of the run's fine terminal-price grid, from the archive.
    A missing or damaged run gives None and a malformed grid is left out, so that row falls back to
    the nine logged quantiles."""
    if pd.isna(run_at):
        return None
    t = pd.Timestamp(run_at)
    t = t.tz_convert("UTC") if t.tzinfo is not None else t
    path = os.path.join(archive_dir, "runs", t.strftime("%Y-%m-%d"), t.strftime("%H%M%SZ") + ".json.gz")
    if os.path.exists(path[:-len(".json.gz")] + "-1.json.gz"):
        return None             # two runs in that second: the log's run_at cannot tell which is this row's
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, EOFError, ValueError, RecursionError, zlib.error):
        return None
    fc = rec.get("forecast") if isinstance(rec, dict) else None
    fc = fc.get("forecast") if isinstance(fc, dict) else None
    grids = {}
    for h, f in (fc.items() if isinstance(fc, dict) else ()):
        try:
            g = f.get("grid") or {}
            levels = np.asarray(g.get("percentiles") or [], dtype=float)
            prices = np.asarray(g.get("terminal") or [], dtype=float)
            h = int(h)
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        # percent levels inside (0, 100) that span the 5-95 band the coverage is scored on
        if (levels.ndim == 1 and len(levels) >= 2 and levels.shape == prices.shape
                and np.isfinite(levels).all() and np.isfinite(prices).all() and (prices > 0).all()
                and 0 < levels[0] <= 5 and 95 <= levels[-1] < 100
                and (np.diff(levels) > 0).all() and (np.diff(prices) >= 0).all()):
            grids[h] = (levels, prices)
    return grids


def _price_basis(df: pd.DataFrame, closes: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Each row's session index in ``closes`` (-1 when its spot date is not there) and the factor
    that puts ``closes`` in the row's own, as-traded price units. A close row's factor is its spot
    over the close on its date. An intraday row's spot is not that close, so it takes the factor of
    the nearest close row (the factor only changes on ex-dates), the later one of two as near, which
    is right when the row's own date is an ex-date; or 1 when the log has none."""
    idx = closes.index
    dates = df["spot_date"].to_numpy()
    pos = idx.searchsorted(dates, side="right") - 1
    found = (pos >= 0) & (np.asarray(idx[np.maximum(pos, 0)]) == dates)
    spot = df["spot"].to_numpy(dtype=float)
    found &= np.isfinite(spot) & (spot > 0)
    pos = np.where(found, pos, -1)
    intraday = (df["spot_status"] == "intraday").to_numpy()
    basis = np.full(len(df), np.nan)
    anchor, live = found & ~intraday, found & intraday
    basis[anchor] = spot[anchor] / closes.to_numpy(dtype=float)[pos[anchor]]
    if anchor.any() and live.any():
        d = pos[anchor][None, :] - pos[live][:, None]
        nearest = (2 * np.abs(d) + (d < 0)).argmin(axis=1)
        basis[live] = basis[anchor][nearest]
    else:
        basis[live] = 1.0
    return pos, basis


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
    # a log without the column (never written by this tool) holds untagged rows, as blanks do
    versions = (df["model_version"].fillna("unknown").astype(str) if "model_version" in df
                else pd.Series("unknown", index=df.index))
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
        from .data import fetch_prices, session_state
        ny_date, is_open = session_state()     # asked on both sides of the download: one across the bell is partial
        prices, _ = fetch_prices([cfg.etf], cfg.history_period, Cache(cfg.cache_dir), cfg.price_ttl_hours, required=(cfg.etf,))
        closes = prices.get(cfg.etf, pd.DataFrame()).get("Close")
        is_open = is_open or session_state()[1]
        if closes is not None and is_open and len(closes) and str(closes.index[-1].date()) == ny_date:
            closes = closes.iloc[:-1]          # today's bar is the live, partial session, not a close
    if closes is not None:
        closes = closes.dropna()
    if closes is None or len(closes) == 0:
        report.notes.append(NO_PRICES)
        return report
    idx = closes.index
    px = closes.to_numpy(dtype=float)
    positions, basis = _price_basis(df, closes)
    # an intraday quote this far from its own day's close (in the row's units) means a split the
    # log cannot place, or bad data: such a row is not scored
    live = (df["spot_status"] == "intraday").to_numpy() & np.isfinite(basis)
    move = np.zeros(len(df))
    move[live] = np.log(px[positions[live]] * basis[live] / df["spot"].to_numpy(dtype=float)[live])
    implausible = np.abs(move) > math.log(1.5)
    if implausible.any():
        report.notes.append(f"{int(implausible.sum())} intraday row(s) not scored: the quote and that day's close "
                            f"in the price history differ by more than 1.5x (a split, or bad data)")
        basis[implausible] = np.nan

    horizons = sorted({int(h) for hs in df["horizons"].fillna("").astype(str) for h in hs.split() if h.isdigit()})
    scored_any = set()
    pending = []                            # (sessions still to go, horizon, spot date) of unresolved rows
    grid_cache: Dict[str, Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]] = {}
    nan2 = (float("nan"), float("nan"))
    for h in horizons:
        recs = []
        for i, (_, row) in enumerate(df.iterrows()):
            qcols = {lvl: row.get(_h(h, f"q{lvl:02d}")) for lvl in LOG_QUANTILES}
            if any(pd.isna(v) or v <= 0 for v in qcols.values()):
                continue
            pos, k = int(positions[i]), float(basis[i])
            if pos < 0 or not np.isfinite(k):
                if row["spot_date"] > idx[-1]:       # a run after the last close in the history (today's)
                    ahead = sum(nyse_session(d.date()) is not None
                                for d in pd.date_range(idx[-1] + pd.Timedelta(days=1), row["spot_date"]))
                    pending.append((h + ahead, h, row["spot_date"]))
                continue
            if pos + h >= len(idx):
                pending.append((pos + h - (len(idx) - 1), h, row["spot_date"]))
                continue
            # the closes after the row's date relative to the close on it, in the row's own units
            path_vals = k * px[pos + 1: pos + h + 1]
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
        enough = n_eff >= MIN_INDEPENDENT
        # model and benchmark CRPS are averaged over the same rows, so the columns and the skill agree
        both = g.dropna(subset=["crps_naive"])
        n_eff_both = effective_n(both["pos"].values, h)
        crps_m = float((both if len(both) else g)["crps_model"].mean())
        crps_n = float(both["crps_naive"].mean()) if len(both) else float("nan")
        pit_lo, pit_hi = mean_interval(g["pit"].values, n_eff) if enough else nan2
        report.horizons.append(HorizonScore(
            horizon=h, n=len(g), mean_pit=float(g["pit"].mean()),
            cov_5_95=float(g["in_5_95"].mean()), cov_25_75=float(g["in_25_75"].mean()),
            frac_below_5=float(g["below_5"].mean()), frac_above_95=float(g["above_95"].mean()),
            mean_realised_return=float(g["ret"].mean()), mean_predicted_median_return=float(g["pred_med"].mean()),
            pred_dd20=float(g["pred_dd20"].mean()), real_dd20=float(g["real_dd20"].mean()),
            pred_up20=float(g["pred_up20"].mean()), real_up20=float(g["real_up20"].mean()),
            n_eff=n_eff,
            mean_pit_ci=(float(np.clip(pit_lo, 0.0, 1.0)), float(np.clip(pit_hi, 0.0, 1.0))),
            cov_5_95_ci=proportion_interval(float(g["in_5_95"].mean()), n_eff) if enough else nan2,
            crps_model=crps_m, crps_naive=crps_n,
            crps_skill=1.0 - crps_m / crps_n if crps_n > 0 else float("nan"),
            crps_skill_ci=(skill_interval(both["crps_model"].values, both["crps_naive"].values, n_eff_both)
                           if n_eff_both >= MIN_INDEPENDENT else nan2),
            n_fine=int(g["fine"].sum()), n_crps=int(len(both)),
        ))
        if h == cfg.rating_horizon:
            for label, gg in g.groupby("rating"):
                report.by_rating[str(label)] = {"n": int(len(gg)), "mean_return": float(gg["ret"].mean()),
                                                "p_positive": float((gg["ret"] > 0).mean())}
    report.n_scoreable = len(scored_any)

    # News sentiment vs the next 10 sessions of SPXL return.
    if "sentiment_score" in df:
        pairs = []
        for i, s in enumerate(df["sentiment_score"]):
            pos = int(positions[i])
            if pd.isna(s) or pos < 0 or pos + 10 >= len(idx):
                continue
            pairs.append((float(s), float(px[pos + 10] / px[pos] - 1.0)))
        if len(pairs) >= 10:
            a = np.array(pairs)
            if a[:, 0].std() > 0 and a[:, 1].std() > 0:      # a constant score has no correlation
                report.sentiment_corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        report.sentiment_n = len(pairs)
    if report.n_scoreable == 0:
        note = "nothing to score yet: no logged forecast has reached its horizon"
        if pending:
            from .report import horizon_label
            left, h, date = min(pending)
            note += (f"; the first, the {horizon_label(h)} forecast from {date.date()}, resolves "
                     + ("at the next close" if left == 1 else f"in {left} sessions"))
        report.notes.append(note)
    return report
