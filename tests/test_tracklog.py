import csv
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spxlcast.config import MODEL_VERSION, Config
from spxlcast.montecarlo import simulate
from spxlcast.tracklog import LOG_QUANTILES, _pit, append_log, forecast_row, load_log, score_log


def _fake_forecast(spot=200.0, spot_date="2026-06-01", status="close", sentiment=0.1, flags=(),
                   asof=datetime(2026, 6, 1, 21, 0, tzinfo=timezone.utc)):
    T = 126
    sim = simulate(spot=spot, mu_annual=np.full(T, 0.07), sigma_annual=np.full(T, 0.16), leverage=3.0,
                   daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[21, 63, 126],
                   n_paths=4000, seed=1)
    return SimpleNamespace(
        snap=SimpleNamespace(asof=asof),
        spot_date=spot_date, spot_status=status, spot=spot,
        rating=SimpleNamespace(label="HOLD", score=-0.1, score_se=0.01, conviction="Low"),
        expected=SimpleNamespace(final=0.061), macro=SimpleNamespace(rf_3m=0.04, vix=17.0),
        etf=SimpleNamespace(annual_cost=0.104), sentiment=SimpleNamespace(score=sentiment, n_used=50),
        sim=sim, vol=SimpleNamespace(total_vol=lambda h: 0.16),
        data_quality=lambda: {"flags": list(flags), "fred_series": 9, "news_fetched": 120},
    )


def test_row_append_and_load(tmp_path):
    path = str(tmp_path / "log.csv")
    fc = _fake_forecast()
    row = forecast_row(fc)
    assert row["spot"] == 200.0 and row["h126_q50"] > 0 and row["model_version"]
    append_log(fc, path)
    append_log(_fake_forecast(status="intraday", spot=201.0), path)     # same date, intraday: ignored by load_log
    append_log(_fake_forecast(spot_date="2026-06-02", spot=202.0), path)
    with open(path, newline="") as fh:
        assert len(list(csv.reader(fh))) == 4      # header + 3 rows
    df = load_log(path)
    assert len(df) == 2                            # one row per spot date, the close preferred
    assert float(df.loc[df["spot_date"] == "2026-06-01", "spot"].iloc[0]) == 200.0


def test_pit_interpolates_between_quantiles():
    q = {lvl: 100.0 * (1 + (lvl - 50) / 100.0) for lvl in LOG_QUANTILES}   # linear price ladder
    assert abs(_pit(q[50], q) - 0.5) < 1e-9
    assert _pit(q[1] * 0.5, q) < 0.01
    assert _pit(q[99] * 1.5, q) > 0.99
    assert 0.25 < _pit(q[25] * 1.001, q) < 0.5


def test_score_log_against_synthetic_history(tmp_path):
    path = str(tmp_path / "log.csv")
    dates = pd.bdate_range("2026-01-02", periods=200)
    rng = np.random.default_rng(0)
    closes = pd.Series(200.0 * np.cumprod(1 + rng.normal(0.0005, 0.03, len(dates))), index=dates)
    for i in range(0, 60, 5):                       # forecasts on 12 dates, all with 126 sessions of follow-up
        fc = _fake_forecast(spot=float(closes.iloc[i]), spot_date=str(dates[i].date()), sentiment=rng.normal())
        append_log(fc, path)
    rep = score_log(path, closes=closes, cfg=Config())
    assert rep.n_rows == 12 and rep.n_scoreable == 12
    hs = {h.horizon: h for h in rep.horizons}
    assert set(hs) == {21, 63, 126}
    for h in hs.values():
        assert h.n == 12 and 0.0 <= h.mean_pit <= 1.0 and 0.0 <= h.cov_5_95 <= 1.0
        assert h.n_eff <= h.n and h.crps_model > 0 and h.crps_naive > 0 and np.isfinite(h.crps_skill)
        assert h.n_fine == 0                                    # no archive: the 9 logged quantiles
    # forecasts 5 sessions apart: 1M windows overlap, 6M windows almost entirely
    assert hs[21].n_eff == pytest.approx((55 + 21) / 21)
    assert all(np.isfinite(hs[21].mean_pit_ci)) and all(np.isfinite(hs[21].crps_skill_ci))
    assert hs[126].n_eff < 3 and all(np.isnan(hs[126].mean_pit_ci))   # too few independent outcomes
    assert "HOLD" in rep.by_rating and rep.by_rating["HOLD"]["n"] == 12
    assert rep.sentiment_n == 12 and rep.sentiment_corr is not None


def test_score_log_with_nothing_elapsed(tmp_path):
    path = str(tmp_path / "log.csv")
    dates = pd.bdate_range("2026-01-02", periods=10)
    closes = pd.Series(np.full(10, 200.0), index=dates)
    append_log(_fake_forecast(spot_date=str(dates[-1].date())), path)
    rep = score_log(path, closes=closes, cfg=Config())
    assert rep.n_rows == 1 and rep.n_scoreable == 0 and rep.horizons == []
    assert any("nothing to score" in n for n in rep.notes)


def test_rows_carry_version_build_and_data_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("SPXLCAST_BUILD", "0123456789abcdef0123")
    from spxlcast.env import build_id
    build_id.cache_clear()
    try:
        row = forecast_row(_fake_forecast(flags=("fred:none", "vix6m:missing")))
        assert row["model_version"] == MODEL_VERSION and row["build"] == "0123456789ab"
        assert row["data_flags"] == "fred:none;vix6m:missing" and row["fred_series"] == 9 and row["news_fetched"] == 120
        path = str(tmp_path / "log.csv")
        append_log(_fake_forecast(), path)
        df = load_log(path)
        assert df["build"].iloc[0] == "0123456789ab"            # read back as text, not a number
    finally:
        build_id.cache_clear()


def test_score_log_splits_by_model_version(tmp_path):
    path = str(tmp_path / "log.csv")
    dates = pd.bdate_range("2026-01-02", periods=200)
    closes = pd.Series(np.full(len(dates), 200.0), index=dates)
    for i in range(0, 30, 5):
        append_log(_fake_forecast(spot_date=str(dates[i].date())), path)
    df = pd.read_csv(path, dtype=str)
    df.loc[df.index[:2], "model_version"] = "0.1.0"          # two rows from an older model
    df.to_csv(path, index=False)
    pooled = score_log(path, closes=closes, cfg=Config())
    assert pooled.versions == {"0.1.0": 2, MODEL_VERSION: 4} and pooled.n_scoreable == 6
    assert any("pooled" in n for n in pooled.notes)
    only = score_log(path, closes=closes, cfg=Config(), model_version=MODEL_VERSION)
    assert only.n_rows == 4 and only.n_scoreable == 4 and only.model_version == MODEL_VERSION
    missing = score_log(path, closes=closes, cfg=Config(), model_version="9.9.9")
    assert missing.n_scoreable == 0 and any("no rows from model version 9.9.9" in n for n in missing.notes)


def test_score_log_uses_the_archived_fine_grid(tmp_path):
    from spxlcast.archive import write_run
    from spxlcast.live import GRID_PERCENTILES
    path, archive = str(tmp_path / "log.csv"), str(tmp_path / "archive")
    dates = pd.bdate_range("2026-01-02", periods=200)
    rng = np.random.default_rng(4)
    closes = pd.Series(200.0 * np.cumprod(1 + rng.normal(0.0005, 0.03, len(dates))), index=dates)
    for i in range(0, 60, 5):
        asof = datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc) + pd.Timedelta(days=i)
        fc = _fake_forecast(spot=float(closes.iloc[i]), spot_date=str(dates[i].date()), asof=asof)
        append_log(fc, path)
        grids = {str(h): {"grid": {"percentiles": list(GRID_PERCENTILES),
                                   "terminal": np.percentile(fc.sim.terminal[h], GRID_PERCENTILES).tolist()}}
                 for h in fc.sim.horizons}
        write_run(archive, {"run_at": asof.isoformat(), "forecast": {"forecast": grids}})
    coarse = {h.horizon: h for h in score_log(path, closes=closes, cfg=Config()).horizons}
    fine = {h.horizon: h for h in score_log(path, closes=closes, cfg=Config(), archive_dir=archive).horizons}
    for h in (21, 63, 126):
        assert fine[h].n_fine == fine[h].n == 12 and coarse[h].n_fine == 0
        assert fine[h].mean_pit == pytest.approx(coarse[h].mean_pit, abs=0.03)    # same distribution, finer grid
        assert fine[h].crps_model > coarse[h].crps_model * 0.97          # the coarse grid misses the tails
