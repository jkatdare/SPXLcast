import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from spxlcast.archive import append_news, load_run, news_records, replay, story_key, write_run
from spxlcast.config import Config
from spxlcast.data import NewsItem
from spxlcast.montecarlo import simulate
from spxlcast.pipeline import _clean
from spxlcast.sentiment import analyze_news

NOW = datetime(2026, 9, 28, 14, 40, 5, tzinfo=timezone.utc)


def _sim_inputs():
    T = 21
    rng = np.random.default_rng(3)
    return dict(spot=287.43, mu_annual=np.full(T, 0.0625) + rng.normal(0, 1e-3, T),
                sigma_annual=0.14 + rng.normal(0, 1e-2, T), leverage=2.97, daily_cost=0.106 / 252,
                tracking_sd_daily=0.0008, rf_annual=0.041, horizons=[5, 10, 21], n_paths=500, dof=4.0,
                skew_gamma=0.9, max_daily_move=0.2, seed=42, drift_sd_annual=0.02, sv_persistence=0.97,
                sv_logvol_sd=0.35, sv_leverage=-0.5)


def test_run_record_round_trips_and_replays_exactly(tmp_path):
    inputs = _sim_inputs()
    original = simulate(**inputs)
    record = _clean({"run_at": NOW.isoformat(), "sim_inputs": inputs, "rating_sim_inputs": None,
                     "forecast": {"spot": inputs["spot"]}})
    path = write_run(str(tmp_path), record)
    assert path.endswith(os.path.join("runs", "2026-09-28", "144005Z.json.gz"))
    loaded = load_run(path)
    assert loaded == json.loads(json.dumps(record))
    again = replay(loaded)
    for h in (5, 10, 21):       # bit-for-bit: JSON keeps every float exactly
        assert np.array_equal(again.terminal[h], original.terminal[h])
        assert np.array_equal(again.path_min[h], original.path_min[h])


def test_news_is_archived_once_across_runs_and_months(tmp_path):
    def rec(key, seen):
        return {"key": key, "first_seen": seen.isoformat(), "title": key}

    root = str(tmp_path)
    assert append_news(root, [rec("a", NOW), rec("b", NOW), rec("a", NOW)]) == 2   # duplicate in one batch
    assert append_news(root, [rec("a", NOW + timedelta(hours=1)), rec("c", NOW)]) == 1
    # a story first archived in September is not re-archived when still fresh in early October
    october = datetime(2026, 10, 2, 14, 40, tzinfo=timezone.utc)
    assert append_news(root, [rec("c", october), rec("d", october)]) == 1
    sept = (tmp_path / "news" / "2026-09.jsonl").read_text(encoding="utf-8").splitlines()
    octo = (tmp_path / "news" / "2026-10.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["key"] for x in sept] == ["a", "b", "c"]
    assert [json.loads(x)["key"] for x in octo] == ["d"]
    # a torn last line (crash mid-write) does not break the next run
    with open(tmp_path / "news" / "2026-10.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"key": "e", "tit')
    assert append_news(root, [rec("d", october), rec("f", october)]) == 1


def test_news_records_keep_text_score_and_relevance():
    items = [
        NewsItem("SPY", "Markets surge on strong growth and upgrades", "Stocks rallied.", NOW - timedelta(hours=1),
                 "Reuters", "https://example.com/a#top"),
        NewsItem("^GSPC", "Markets surge on strong growth and upgrades", "", NOW - timedelta(hours=2), "Reuters",
                 "https://example.com/a"),
        NewsItem("NVDA", "Nvidia downgraded on weak demand warning", "", NOW - timedelta(hours=3), "AP", ""),
    ]
    res = analyze_news(items, Config(), holdings_weights={"NVDA": 0.08}, now=NOW,
                       holdings_names={"NVDA": "NVIDIA Corp"})
    recs = {r["title"]: r for r in news_records(res, NOW.isoformat())}
    assert len(recs) == 2                       # one per unique story, as the model scored them
    surge = recs["Markets surge on strong growth and upgrades"]
    assert surge["key"] == "https://example.com/a" and surge["feeds"] == ["SPY", "^GSPC"]
    assert surge["market_wide"] and surge["relevance"] == 1.0 and surge["score"] > 0
    nvda = recs["Nvidia downgraded on weak demand warning"]
    assert nvda["score"] < 0 and 0 < nvda["relevance"] < 1
    assert nvda["key"] == story_key("", nvda["title"], nvda["published"])   # no URL: title + date
