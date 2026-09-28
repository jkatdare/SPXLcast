"""Golden-file regression test: MODEL_VERSION must change whenever the model's numbers change.

A fixed synthetic market (no network) goes through the real pipeline: drift, vol curve, ETF
calibration, news tilt, simulation, assessment, the retired rating and the track-record row. The key
numbers are compared with ``tests/golden/forecast_golden.json``, which records the MODEL_VERSION it
was made with. Regenerate it with ``python tests/test_golden.py``: that refuses to overwrite changed
numbers under an unchanged MODEL_VERSION (``--force`` when only this test's own market changed).
"""
import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # also when run as a script

import spxlcast.pipeline as pipeline  # noqa: E402
from spxlcast.assess import assessment_to_dict  # noqa: E402
from spxlcast.config import FRED_SERIES, MODEL_VERSION, Config  # noqa: E402
from spxlcast.data import MarketSnapshot, NewsItem  # noqa: E402
from spxlcast.tracklog import LOG_QUANTILES, forecast_row  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "forecast_golden.json"
REGENERATE = "python tests/test_golden.py"
# A value matches within RTOL of its size or ATOL outright (returns and rates near zero). The numbers
# are identical on Windows under Python 3.12 / pandas 3 and 3.13 / pandas 2, with or without AVX. Moving
# every exp, power, sqrt and random draw in the engine by up to 4 units in the last digit, as another
# CPU or C maths library may, moved prices by at most 1.2e-12 of their size and every other value by
# at most 5.2e-12 outright. Changing one engine setting by 1e-7 (financing spread, stochastic vol,
# tails, skew, drift uncertainty, earnings growth) moves 57 to 146 of the values past these limits.
RTOL, ATOL = 1e-9, 1e-9
N_PATHS = 5000
HORIZONS = (5, 10, 21, 63, 126, 252)     # fixed here: extra horizons do not change the others' numbers
ASOF = datetime(2026, 9, 25, 21, 40, tzinfo=timezone.utc)   # a Friday, after the close
LAST = pd.Timestamp("2026-09-25")
ROW_TEXT = ("run_at", "build", "model_version")   # track-record columns that are not model output


# ---- the synthetic market -------------------------------------------------------------------
def _frame(values, idx):
    return pd.DataFrame({"Close": np.asarray(values, dtype=float)}, index=idx)


def golden_market(n=600) -> MarketSnapshot:
    """Two-plus years of prices, all nine FRED series and a week of headlines (two with a summary, one
    off-topic for the feed it came from, so the news-scoring settings reach the numbers). Every regime
    penalty fires, below their joint cap, so each one reaches the drift. Built with + and x only (no
    exp or power), so the inputs are the same to the last bit on every platform."""
    cal = pd.bdate_range(end=LAST, periods=n)
    rng = np.random.default_rng(20260925)
    r = 0.0004 + 0.011 * rng.standard_normal(n)
    r[0] = 0.0
    r_etf = 3 * r - 0.0001 + 0.0004 * rng.standard_normal(n)
    r_etf[[200, 400]] += (0.03, -0.025)                         # two close-vs-NAV dislocation days
    vix = np.empty(n)                                            # mean-reverting around 18
    vix[0] = 18.0
    for t, e in enumerate(0.6 * rng.standard_normal(n - 1), start=1):
        vix[t] = 18.0 + 0.97 * (vix[t - 1] - 18.0) + e
    snap = MarketSnapshot(asof=ASOF, max_stale_sessions=5)
    snap.prices = {"SPY": _frame(450.0 * np.cumprod(1 + r), cal), "SPXL": _frame(100.0 * np.cumprod(1 + r_etf), cal),
                   "^GSPC": _frame(4500.0 * np.cumprod(1 + r - 0.00005), cal),
                   "^IRX": _frame(np.full(n, 4.0), cal), "^TNX": _frame(np.full(n, 4.0), cal),
                   "^VIX": _frame(vix, cal), "^VIX3M": _frame(vix + 2.8, cal), "^VIX6M": _frame(vix + 4.9, cal)}
    snap.calendar = cal
    snap.fetched_at = {"prices": ASOF}
    snap.infos = {"SPXL": {"netExpenseRatio": 0.87}, "SPY": {"trailingPE": 27.5, "yield": 0.0115}}
    daily = pd.bdate_range(end=LAST - pd.Timedelta(days=1), periods=400)     # FRED is a day behind
    levels = {"DGS3MO": 4.05, "DGS2": 3.8, "DGS10": 3.95, "DFII10": 1.6, "T10YIE": 2.35, "SOFR": 4.1,
              "BAMLH0A0HYM2": 5.1}                             # curve inverted by 0.1pp, HY spread 0.1pp over 5%
    snap.fred = {k: pd.Series(v, index=daily) for k, v in levels.items()}
    months = pd.date_range(end="2026-08-01", periods=60, freq="MS")
    snap.fred["CPIAUCSL"] = pd.Series(300.0 * np.cumprod(np.r_[1.0, np.full(59, 1.0031)]), index=months)  # 3.8% a year
    snap.fred["UNRATE"] = pd.Series(np.r_[np.full(54, 4.1), [4.2, 4.4, 4.6, 4.7, 4.8, 4.8]], index=months)  # Sahm
    assert set(snap.fred) == set(FRED_SERIES)
    snap.holdings = pd.DataFrame({"symbol": ["NVDA", "MSFT", "AAPL"], "weight": [0.075, 0.065, 0.06],
                                  "name": ["NVIDIA Corp", "Microsoft Corp", "Apple Inc"]})
    headlines = (("SPY", "Stocks rally as Fed signals rate cuts ahead", 0.3),
                 ("^TNX", "Treasury yields jump after hot inflation report", 1.2),
                 ("SPXL", "Wall Street's fear gauge spikes as tariffs rattle markets", 2.5),
                 ("AAPL", "Apple beats earnings estimates on strong iPhone demand", 0.8),
                 ("NVDA", "Nvidia shares slide after new export curbs", 3.1),
                 ("MSFT", "Microsoft opens a new office in Dublin", 4.0),
                 ("MSFT", "Airline shares soar on strong holiday travel demand", 1.5))   # off-topic for its feed
    summaries = {1: "Investors worry the Fed will wait longer.", 3: "Services revenue hit a record high."}
    snap.news = [NewsItem(t, title, summaries.get(i, ""), ASOF - timedelta(days=age), "synthetic",
                          f"https://example.com/{i}") for i, (t, title, age) in enumerate(headlines)]
    return snap


def golden_forecast(**overrides):
    """The forecast on the synthetic market with the default Config (all engine settings as shipped)."""
    orig = pipeline.load_market
    pipeline.load_market = lambda cfg: golden_market()
    try:
        return pipeline.run_forecast(Config(n_paths=N_PATHS, horizons=HORIZONS, **overrides))
    finally:
        pipeline.load_market = orig


def _horizon(sim, h) -> dict:
    """Summary statistics (less their copies of the quantiles), end-price and path-minimum quantiles."""
    out = {k: x for k, x in sim.summary(h).items() if not k.startswith("q")}
    out["quantile_prices"] = {f"q{int(q):02d}": x for q, x in sim.quantiles(h, LOG_QUANTILES).items()}
    out["path_min_quantile_prices"] = {f"q{int(q):02d}": x for q, x in sim.path_min_quantiles(h).items()}
    return out


def golden_outputs(fc) -> dict:
    """The numbers the golden file pins, as plain JSON types."""
    e, v, etf, sim = fc.expected, fc.vol, fc.etf, fc.sim
    cal = etf.calibration.__dict__ if etf.calibration is not None else None
    r = fc.rating
    out = {
        "drift": {"earnings_yield_model": e.earnings_yield_model, "dividend_growth_model": e.dividend_growth_model,
                  "base": e.base, "adjustments": e.adjustments, "final": e.final},
        "vol": {"pillars": {str(h): s for h, s in v.pillars.items()}, "one_year_vol": v.one_year_vol,
                "total_vol": {str(h): v.total_vol(h) for h in sim.horizons}},
        "etf": {"leverage": etf.leverage, "expense_ratio": etf.expense_ratio, "financing_rate": etf.financing_rate,
                "tracking_sd_daily": etf.tracking_sd_daily, "calibration": cal},
        "sentiment": {"score": fc.sentiment.score, "drift_adjustment": fc.sentiment.drift_adjustment,
                      "n_articles": fc.sentiment.n_articles, "n_used": fc.sentiment.n_used},
        "assessment": {k: x for k, x in assessment_to_dict(fc.assessment).items() if k != "notes"},
        "rating": {"label": r.label, "conviction": r.conviction, "score": r.score, "score_se": r.score_se,
                   "edge_annual": r.edge_annual, "sharpe_annual": r.sharpe_annual, "p_beat_rf": r.p_beat_rf},
        "forecast": {str(h): _horizon(sim, h) for h in sim.horizons},
        "track_record_row": {k: x for k, x in forecast_row(fc).items() if k not in ROW_TEXT},
    }
    return json.loads(json.dumps(out, allow_nan=False))


# ---- comparison -----------------------------------------------------------------------------
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(want, got, path="", changed=None, missing=None, new=None):
    """(changed, missing, new): paths whose value differs beyond RTOL / ATOL, paths only in the
    golden file, paths only in the current outputs."""
    changed = [] if changed is None else changed
    missing = [] if missing is None else missing
    new = [] if new is None else new
    if isinstance(want, dict) and isinstance(got, dict):
        for k, w in want.items():
            p = f"{path}.{k}" if path else k
            if k in got:
                compare(w, got[k], p, changed, missing, new)
            else:
                missing.append(p)
        new.extend(f"{path}.{k}" if path else k for k in got if k not in want)
    elif _is_number(want) and _is_number(got):
        if not math.isclose(want, got, rel_tol=RTOL, abs_tol=ATOL):
            changed.append(f"{path}: {want!r} in the golden file, {got!r} now")
    elif want != got:
        changed.append(f"{path}: {want!r} in the golden file, {got!r} now")
    return changed, missing, new


def _listing(paths, n=12) -> str:
    return "\n  " + "\n  ".join(paths[:n]) + (f"\n  ... and {len(paths) - n} more" if len(paths) > n else "")


# ---- tests ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def outputs():
    return golden_outputs(golden_forecast())


def test_forecast_matches_the_golden_file(outputs):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if golden["model_version"] != MODEL_VERSION:
        pytest.fail(f"MODEL_VERSION is {MODEL_VERSION} but the golden file was made with {golden['model_version']}: "
                    f"regenerate the golden file ({REGENERATE})")
    changed, missing, new = compare(golden["outputs"], outputs)
    if changed:
        pytest.fail(f"model outputs changed: bump MODEL_VERSION in spxlcast/config.py and regenerate the golden file "
                    f"({REGENERATE}). {len(changed)} values differ:" + _listing(changed))
    if missing or new:
        pytest.fail(f"the golden file lists other outputs than the forecast now gives (no number changed): regenerate "
                    f"the golden file ({REGENERATE})." + _listing([f"not produced: {p}" for p in missing]
                                                                  + [f"new: {p}" for p in new]))


def test_a_small_model_change_is_caught(outputs):
    """A hundredth of a basis point more financing spread fails the comparison, in many places."""
    other = golden_outputs(golden_forecast(swap_spread=Config().swap_spread + 1e-6))
    changed, missing, new = compare(outputs, other)
    assert len(changed) > 100 and not missing and not new
    assert any(p.startswith("forecast.126.quantile_prices.q50:") for p in changed)
    assert compare(outputs, outputs) == ([], [], [])


# ---- regeneration ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate tests/golden/forecast_golden.json.")
    ap.add_argument("--force", action="store_true", help="overwrite changed numbers without a MODEL_VERSION bump "
                                                          "(only when this test's own market changed)")
    args = ap.parse_args(argv)
    outputs = golden_outputs(golden_forecast())
    if GOLDEN.exists() and not args.force:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        changed = compare(golden["outputs"], outputs)[0]
        if golden["model_version"] == MODEL_VERSION and changed:
            print(f"model outputs changed but MODEL_VERSION is still {MODEL_VERSION}: bump it in spxlcast/config.py "
                  f"first (or pass --force if only this test's market changed). {len(changed)} values differ:"
                  + _listing(changed), file=sys.stderr)
            return 1
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    doc = {"model_version": MODEL_VERSION, "n_paths": N_PATHS, "horizons": list(HORIZONS), "outputs": outputs}
    GOLDEN.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {GOLDEN.relative_to(ROOT)} for MODEL_VERSION {MODEL_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
