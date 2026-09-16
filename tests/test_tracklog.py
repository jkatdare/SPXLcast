import csv
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from spxlcast.config import Config
from spxlcast.montecarlo import simulate
from spxlcast.tracklog import LOG_QUANTILES, _pit, append_log, forecast_row, load_log, score_log


def _fake_forecast(spot=200.0, spot_date="2026-06-01", status="close", sentiment=0.1):
    T = 126
    sim = simulate(spot=spot, mu_annual=np.full(T, 0.07), sigma_annual=np.full(T, 0.16), leverage=3.0,
                   daily_cost=0.10 / 252, tracking_sd_daily=0.0, rf_annual=0.04, horizons=[21, 63, 126],
                   n_paths=4000, seed=1)
    return SimpleNamespace(
        snap=SimpleNamespace(asof=datetime(2026, 6, 1, 21, 0, tzinfo=timezone.utc)),
        spot_date=spot_date, spot_status=status, spot=spot,
        rating=SimpleNamespace(label="HOLD", score=-0.1, score_se=0.01, conviction="Low"),
        expected=SimpleNamespace(final=0.061), macro=SimpleNamespace(rf_3m=0.04, vix=17.0),
        etf=SimpleNamespace(annual_cost=0.104), sentiment=SimpleNamespace(score=sentiment, n_used=50),
        sim=sim, vol=SimpleNamespace(total_vol=lambda h: 0.16),
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
