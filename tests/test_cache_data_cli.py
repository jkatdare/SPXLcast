import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from spxlcast.cache import Cache
from spxlcast.cli import build_parser, config_from_args
from spxlcast.data import MarketSnapshot, _parse_news, session_state
from spxlcast.pipeline import _clean
from spxlcast.report import horizon_label, ordinal


# ---- cache -----------------------------------------------------------------------------
def test_cache_roundtrip_and_empty_results_not_stored(tmp_path):
    c = Cache(str(tmp_path))
    c.put("k", {"a": 1})
    assert c.get("k", 1.0) == {"a": 1}
    assert c.mtime("k") is not None
    for empty in ({}, [], None, pd.DataFrame()):
        c.put("empty", empty)
        assert c.get("empty", 1.0) is None
    calls = []

    def fetch():
        calls.append(1)
        return {}
    assert c.get_or_fetch("e2", 1.0, fetch) == {}
    assert c.get_or_fetch("e2", 1.0, fetch) == {}
    assert len(calls) == 2                       # an empty result is fetched again, never served
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]


def test_cache_refresh_writes_but_does_not_read(tmp_path):
    c = Cache(str(tmp_path), read_enabled=False)
    c.put("k", [1, 2])
    assert c.get("k", 1.0) is None
    assert Cache(str(tmp_path)).get("k", 1.0) == [1, 2]


def test_cache_ttl_and_prune(tmp_path):
    c = Cache(str(tmp_path))
    c.put("k", [1])
    path = c._path("k")
    old = os.path.getmtime(path) - 10 * 3600
    os.utime(path, (old, old))
    assert c.get("k", 1.0) is None
    assert c.get("k", 24.0) == [1]
    c.put("archive:x", [2])
    apath = c._path("archive:x")
    os.utime(apath, (old, old))
    assert c.prune(max_age_hours=5) == 1          # the archive entry survives pruning
    assert c.get("k", 24.0) is None
    assert c.get("archive:x", 24.0) == [2]


# ---- data ------------------------------------------------------------------------------
def _snap_with_calendar(n=30):
    snap = MarketSnapshot(asof=datetime.now(timezone.utc))
    cal = pd.bdate_range("2026-06-01", periods=n)
    snap.prices["SPY"] = pd.DataFrame({"Close": np.linspace(100, 130, n)}, index=cal)
    snap.calendar = cal
    return snap, cal


def test_change_and_diff_align_to_calendar_and_flag_gaps():
    snap, cal = _snap_with_calendar()
    assert abs(snap.change("SPY", 1) - (130 / (130 - 30 / 29) - 1)) < 1e-9
    # a series with a 12-session hole: the 5-day change spans the hole => n/a, the 1-day change is fine
    keep = list(range(0, 10)) + list(range(22, 30))
    gappy = pd.DataFrame({"Close": np.arange(30, dtype=float)[keep]}, index=cal[keep])
    snap.prices["^VIX3M"] = gappy
    assert snap.diff("^VIX3M", 1) == 1.0
    assert snap.diff("^VIX3M", 5) == 5.0             # reference session 24 exists
    assert snap.diff("^VIX3M", 10) is None           # reference session 19 is inside the hole
    assert snap.diff("^VIX3M", 20) == 20.0           # reference session 9 exists again
    # a series that stopped 10 sessions ago is stale for the latest value too
    stale = pd.DataFrame({"Close": np.arange(20, dtype=float)}, index=cal[:20])
    snap.prices["^SKEW"] = stale
    assert snap.fresh_last("^SKEW") is None
    assert snap.change("^SKEW", 1) is None
    # a series lagging the calendar by one session (published late): changes are anchored to its own
    # latest observation, so the 1D change is the real previous-session change, not zero
    lagging = pd.DataFrame({"Close": np.arange(29, dtype=float) * 2.0}, index=cal[:29])
    snap.prices["^VVIX"] = lagging
    assert snap.fresh_last("^VVIX") == 56.0
    assert snap.diff("^VVIX", 1) == 2.0
    assert snap.diff("^VVIX", 21) == 42.0
    assert snap.diff("^VVIX", 29) is None          # not enough history


def test_mixed_timezone_inputs_do_not_crash():
    snap, cal = _snap_with_calendar()
    aware = pd.DataFrame({"Close": np.arange(30, dtype=float)}, index=cal.tz_localize("America/New_York"))
    snap.prices["^VIX"] = aware
    assert snap.fresh_last("^VIX") == 29.0
    assert snap.diff("^VIX", 1) == 1.0
    snap.calendar = cal.tz_localize("UTC")
    naive = pd.DataFrame({"Close": np.arange(30, dtype=float)}, index=cal)
    snap.prices["^SKEW"] = naive
    assert snap.diff("^SKEW", 5) == 5.0


def test_session_state():
    assert session_state(pd.Timestamp("2026-09-15 12:00", tz="America/New_York")) == ("2026-09-15", True)
    assert session_state(pd.Timestamp("2026-09-15 16:30", tz="America/New_York")) == ("2026-09-15", False)
    assert session_state(pd.Timestamp("2026-09-13 12:00", tz="America/New_York"))[1] is False   # Sunday


def test_parse_news_formats_and_undated_items():
    raw = [
        {"id": "1", "content": {"title": "Stocks fall", "summary": "s", "pubDate": "2026-09-15T15:30:49Z",
                                "provider": {"displayName": "Wire"}, "canonicalUrl": {"url": "http://x"}}},
        {"title": "Legacy item", "publisher": "Old", "link": "http://y", "providerPublishTime": 1_700_000_000},
        {"content": {"title": "Undated item"}},
        {"title": "Millisecond epoch", "providerPublishTime": 1_700_000_000_000},   # dropped, never crashes
    ]
    items = _parse_news("SPY", raw)
    assert [i.title for i in items] == ["Stocks fall", "Legacy item"]
    assert items[0].provider == "Wire" and items[0].url == "http://x"
    assert items[1].published.tzinfo is not None


# ---- pipeline JSON cleaning -----------------------------------------------------------
def test_clean_makes_json_strict():
    obj = {"a": np.float64(1.5), "b": float("nan"), "c": np.array([1.0, np.inf]), "d": [np.int64(3), {"e": -np.inf}]}
    cleaned = _clean(obj)
    assert cleaned == {"a": 1.5, "b": None, "c": [1.0, None], "d": [3, {"e": None}]}
    json.dumps(cleaned, allow_nan=False)


# ---- cli --------------------------------------------------------------------------------
def test_cli_rejects_bad_horizons_and_paths():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--horizons", "0", "21"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--paths", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--rating-horizon", "-5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--index-drift", "-1.5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--vol", "-0.1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--skew", "0"])


def test_cli_config_mapping_adds_rating_horizon():
    args = build_parser().parse_args(["price", "190", "--horizons", "10", "42", "--rating-horizon", "63",
                                      "--no-fred", "--eps-growth", "0.07", "--swap-spread", "0.005", "--skew", "1.0"])
    cfg = config_from_args(args)
    assert cfg.horizons == (10, 42, 63) and cfg.rating_horizon == 63
    assert cfg.use_fred is False and cfg.override_eps_growth == 0.07
    assert cfg.swap_spread == 0.005 and cfg.skew_gamma == 1.0
    assert args.prices == [190.0]


def test_top_level_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-h"])
    out = capsys.readouterr().out
    for cmd in ("forecast", "price", "metrics", "news", "calibrate"):
        assert cmd in out


# ---- report helpers ---------------------------------------------------------------------
def test_ordinal_and_horizon_label():
    assert [ordinal(x) for x in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 63, 100, 101)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "63rd", "100th", "101st"]
    assert horizon_label(21) == "1M" and horizon_label(126) == "6M" and horizon_label(252) == "1Y"
    assert horizon_label(10) == "2W" and horizon_label(300) == "300d" and horizon_label(504) == "2Y"


def test_build_id_prefers_the_image_build_and_falls_back_to_git(monkeypatch):
    from spxlcast.env import build_id
    build_id.cache_clear()
    try:
        monkeypatch.setenv("SPXLCAST_BUILD", "f00dfeedbeefcafe1234")
        assert build_id() == "f00dfeedbeef"
        build_id.cache_clear()
        monkeypatch.setenv("SPXLCAST_BUILD", "unknown")        # an image built without the argument
        b = build_id()
        assert b == "unknown" or len(b.split("-")[0]) == 12    # this checkout's commit when git is available
    finally:
        build_id.cache_clear()
