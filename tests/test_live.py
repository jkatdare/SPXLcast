import csv
import json
import threading
from datetime import datetime, timezone

import numpy as np
import pytest

from spxlcast import live
from spxlcast.live import GRID_PERCENTILES, refresh_once, reprice, run_loop


def _base(spot=100.0, seed=0):
    """A stored forecast in the shape forecast_to_dict writes, from a lognormal toy sample."""
    rng = np.random.default_rng(seed)
    pcts = list(GRID_PERCENTILES)
    forecast = {}
    for h, vol in ((21, 0.1), (126, 0.3)):
        term = spot * np.exp(rng.normal(0.0, vol, 20000))
        lo = np.minimum(term, spot * np.exp(-np.abs(rng.normal(0.0, vol, 20000))))
        hi = np.maximum(term, spot * np.exp(np.abs(rng.normal(0.0, vol, 20000))))
        forecast[str(h)] = {
            "horizon": h, "median_return": 0.0, "mean_return": vol ** 2 / 2, "p_positive": 0.5, "p_beat_rf": 0.49,
            "p_drawdown_20": float(np.mean(lo <= 0.8 * spot)), "p_up_20": float(np.mean(hi >= 1.2 * spot)),
            "quantile_prices": {f"{q:.1f}": float(np.percentile(term, q)) for q in (5, 25, 50, 75, 95)},
            "grid": {"percentiles": pcts, "terminal": np.percentile(term, pcts).tolist(),
                     "path_min": np.percentile(lo, pcts).tolist(), "path_max": np.percentile(hi, pcts).tolist()},
        }
    return {
        "run_at": "2026-09-22T17:40:00+00:00", "etf": "SPXL", "spot": spot, "spot_status": "intraday",
        "rating": {"label": "HOLD", "conviction": "Low", "score": 0.14, "score_se": 0.02, "horizon_days": 126},
        "forecast": forecast,
        "price_lookup": {"80.0": [], "90.0": []},
        "limit_ladder": {"126": [{"p_fill": 0.5, "price": 85.0, "vs_spot": -0.15}]},
    }


def test_reprice_scales_prices_and_keeps_returns():
    base = _base()
    live_ = reprice(base, 110.0, now=datetime(2026, 9, 22, 18, 31, tzinfo=timezone.utc))
    assert live_["session_open"] is True and live_["asof_ny"].startswith("2026-09-22 14:31")
    assert live_["change_vs_base"] == pytest.approx(0.10)
    h = live_["horizons"]["126"]
    assert h["quantile_prices"]["50.0"] == pytest.approx(base["forecast"]["126"]["quantile_prices"]["50.0"] * 1.1)
    assert h["p_drawdown_20"] == base["forecast"]["126"]["p_drawdown_20"]      # a return statistic: unchanged
    assert live_["rating"]["label"] == "HOLD" and live_["rating"]["score"] == 0.14
    assert live_["limit_ladder"]["126"][0]["price"] == pytest.approx(93.5)


def test_reprice_moves_fixed_levels_the_right_way():
    base = _base()
    at_base = reprice(base, 100.0)["horizons"]["126"]["price_lookup"]
    higher = reprice(base, 120.0)["horizons"]["126"]["price_lookup"]
    lvl = {r["price"]: r for r in at_base}
    lvl_hi = {r["price"]: r for r in higher}
    # $90 is 10% below a $100 spot: its percentile should agree with the sample the grid came from
    grid = base["forecast"]["126"]["grid"]
    expect = float(np.interp(90.0, grid["terminal"], grid["percentiles"]))
    assert lvl[90.0]["percentile"] == pytest.approx(expect, abs=0.5)
    assert 0.0 < lvl[90.0]["p_touch_below"] < 1.0 and lvl[90.0]["p_touch_above"] == 1.0   # already above 90
    # after a rally the same dollar level is further below spot: lower percentile, harder to dip to
    assert lvl_hi[90.0]["percentile"] < lvl[90.0]["percentile"]
    assert lvl_hi[90.0]["p_touch_below"] < lvl[90.0]["p_touch_below"]
    assert lvl_hi[90.0]["p_end_above"] == pytest.approx(1.0 - lvl_hi[90.0]["percentile"] / 100.0)


def test_reprice_rejects_bad_spot():
    with pytest.raises(ValueError):
        reprice(_base(), 0.0)
    with pytest.raises(ValueError):
        reprice(_base(), float("nan"))


def test_refresh_once_writes_live_json_and_spot_log(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "forecast.json").write_text(json.dumps(_base()), encoding="utf-8")
    now = datetime(2026, 9, 22, 15, 1, tzinfo=timezone.utc)
    out = refresh_once(str(tmp_path), fetch=lambda t: 105.0, now=now)
    assert out is not None and out["spot"] == 105.0
    written = json.loads((tmp_path / "output" / "live.json").read_text(encoding="utf-8"))
    assert written["spot"] == 105.0 and written["base_spot"] == 100.0
    rows = list(csv.DictReader(open(tmp_path / "logs" / "spot_log.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["spot"] == "105.0000" and rows[0]["session"] == "open"
    # a failed quote leaves the previous files alone and returns None
    assert refresh_once(str(tmp_path), fetch=lambda t: None, now=now) is None
    assert len(list(csv.DictReader(open(tmp_path / "logs" / "spot_log.csv", encoding="utf-8")))) == 1


def test_refresh_once_without_base(tmp_path):
    assert refresh_once(str(tmp_path), fetch=lambda t: 1.0) is None


def test_loop_only_fetches_while_the_session_is_open(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "forecast.json").write_text(json.dumps(_base()), encoding="utf-8")
    calls = []
    stop = threading.Event()

    def fetch(t):
        calls.append(t)
        stop.set()
        return 101.0

    import pandas as pd
    monkeypatch.setattr(live, "ny_now", lambda: pd.Timestamp("2026-09-22 03:00", tz="America/New_York"))
    t = threading.Thread(target=run_loop, args=(str(tmp_path), 0.01, stop, fetch, 0.01), daemon=True)
    t.start()
    t.join(0.2)          # several idle ticks go by
    stop.set()
    t.join(1.0)
    assert calls == [] and not t.is_alive()     # closed: never fetched

    stop = threading.Event()
    monkeypatch.setattr(live, "ny_now", lambda: pd.Timestamp("2026-09-22 10:00", tz="America/New_York"))
    run_loop(str(tmp_path), interval=0.01, stop=stop, fetch=fetch, idle_interval=0.01)
    assert calls == ["SPXL"] and (tmp_path / "output" / "live.json").exists()
