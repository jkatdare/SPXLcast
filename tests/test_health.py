from datetime import datetime, timedelta, timezone

from spxlcast.health import check, parse_cron, render, scheduled_runs, session_open

UTC = timezone.utc
TUE = datetime(2026, 9, 29, tzinfo=UTC)            # a Tuesday (New York on daylight time)


def at(h, m=0, day=TUE):
    return day.replace(hour=h, minute=m)


def rows_for(slots, spot_date="2026-09-29", flags=""):
    return [{"run_at": (s + timedelta(seconds=25)).strftime("%Y-%m-%dT%H:%M:%SZ"), "spot_date": spot_date,
             "spot_status": "intraday", "data_flags": flags} for s in slots]


def runs_until(now):
    return scheduled_runs(now - timedelta(days=4), now)


def healthy(now, **overrides):
    rows = overrides.pop("rows", rows_for(runs_until(now)))
    last = max(r["run_at"] for r in rows) if rows else None
    args = dict(healthz_ok=True, rows=rows, live={"asof": (now - timedelta(minutes=1)).isoformat()},
                forecast={"run_at": last.replace("Z", ".500000+00:00") if last else None})
    args.update(overrides)
    return check(now, **args)


def test_cron_parsing_and_schedule():
    assert parse_cron("40 13-21 * * 1-5") == ([40], list(range(13, 22)), [1, 2, 3, 4, 5])
    assert parse_cron("0 9,12 * * 0,7")[2] == [0]           # 7 is Sunday too
    fri = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)
    slots = scheduled_runs(fri, fri + timedelta(days=3, hours=-6))     # Friday 20:00 -> Monday 14:00
    assert slots == [at(20, 40, fri), at(21, 40, fri), datetime(2026, 9, 28, 13, 40, tzinfo=UTC)]


def test_session_follows_new_york_daylight_time():
    assert session_open(at(13, 45))                                       # 9:45 EDT
    assert not session_open(datetime(2026, 12, 1, 13, 45, tzinfo=UTC))    # 8:45 EST
    assert session_open(datetime(2026, 12, 1, 20, 30, tzinfo=UTC))        # 15:30 EST
    assert not session_open(datetime(2026, 10, 3, 15, 0, tzinfo=UTC))     # Saturday


def test_healthy_site_passes():
    for now in (at(15, 5), at(17, 5), at(19, 5), at(22, 15)):
        assert healthy(now) == ([], [])


def test_missed_run_is_reported_by_the_next_check():
    now = at(15, 5)
    rows = rows_for([s for s in runs_until(now) if s != at(14, 40)])
    problems, _ = healthy(now, rows=rows)
    assert len(problems) == 1 and "2026-09-29 14:40 UTC" in problems[0] and "13:40" in problems[0]
    # outside the lookback window it is no longer re-reported
    later = at(19, 5)
    rows = rows_for([s for s in runs_until(later) if s != at(14, 40)])
    assert healthy(later, rows=rows) == ([], [])


def test_monday_morning_rechecks_friday_evening_only():
    mon = datetime(2026, 9, 28, 15, 5, tzinfo=UTC)
    fri = datetime(2026, 9, 25, tzinfo=UTC)
    rows = rows_for([s for s in runs_until(mon) if s.day == 28 or s >= at(19, 40, fri)], spot_date="2026-09-28")
    assert healthy(mon, rows=rows) == ([], [])       # Friday's earlier runs were judged on Friday
    rows = rows_for([s for s in runs_until(mon) if s != at(21, 40, fri)], spot_date="2026-09-28")
    problems, _ = healthy(mon, rows=rows)            # in case Friday's 22:15 check never ran
    assert len(problems) == 1 and "2026-09-25 21:40 UTC" in problems[0]


def test_a_run_still_in_progress_is_not_a_miss():
    now = at(14, 50)                           # the 14:40 run may still be running
    rows = rows_for([s for s in runs_until(now) if s != at(14, 40)])
    assert healthy(now, rows=rows) == ([], [])


def test_live_loop_must_be_fresh_in_session_and_run_until_the_close():
    stale = {"asof": (at(12, 0)).isoformat()}
    problems, _ = healthy(at(15, 5), live=stale)
    assert len(problems) == 1 and "live.json" in problems[0]
    assert "live.json" in healthy(at(17, 5), live=None)[0][0]
    # after the close: a loop that stopped during the day is still reported
    problems, _ = healthy(at(22, 15), live=stale)
    assert len(problems) == 1 and "before the last session close (2026-09-29 20:00 UTC)" in problems[0]
    assert healthy(at(22, 15), live={"asof": at(19, 59).isoformat()}) == ([], [])
    assert healthy(at(22, 15), live=None) == ([], [])
    sat = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)
    rows = rows_for(runs_until(sat), spot_date="2026-10-02")
    assert healthy(sat, rows=rows, live={"asof": "2026-10-02T19:59:10+00:00"}) == ([], [])
    assert len(healthy(sat, rows=rows, live={"asof": "2026-10-02T18:00:00+00:00"})[0]) == 1
    winter = datetime(2026, 12, 1, 22, 15, tzinfo=UTC)                  # the close is 21:00 UTC
    rows = rows_for(runs_until(winter), spot_date="2026-12-01")
    assert healthy(winter, rows=rows, live={"asof": "2026-12-01T20:59:00+00:00"}) == ([], [])
    assert len(healthy(winter, rows=rows, live={"asof": "2026-12-01T20:40:00+00:00"})[0]) == 1


def test_unfinished_run_and_stale_prices_and_down_page():
    now = at(17, 5)
    problems, _ = healthy(now, forecast={"run_at": at(15, 40).isoformat()})
    assert len(problems) == 1 and "did not finish" in problems[0]
    problems, _ = healthy(now, rows=rows_for(runs_until(now), spot_date="2026-09-23"))
    assert len(problems) == 1 and "stale" in problems[0]
    problems, _ = healthy(now, healthz_ok=False, errors=["could not fetch the forecast log: timeout"])
    assert len(problems) == 2


def test_data_flags_are_warnings_and_torn_rows_are_tolerated():
    now = at(19, 5)
    rows = rows_for(runs_until(now), flags="fred:none;vix6m:missing")
    problems, warnings = healthy(now, rows=rows)
    assert problems == [] and len(warnings) == 1 and "fred:none" in warnings[0]
    torn = rows_for(runs_until(now)) + [{"run_at": "2026-09-29T19:04:59Z", "spot_date": None}]
    problems, _ = healthy(now, rows=torn, forecast={"run_at": "2026-09-29T19:04:59+00:00"})
    assert problems == []
    assert "FAILED" in render("https://x", now, ["boom"], []) and "passed" in render("https://x", now, [], [])
