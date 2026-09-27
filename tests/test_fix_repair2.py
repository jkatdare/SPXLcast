"""Regression tests for the second repair round: problems the re-verification still found after the
first round of fixes (all offline)."""
import csv
import glob
import json
import logging
import os
import socket
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import requests

import spxlcast.data as data
from spxlcast import health
from spxlcast.config import Config
from spxlcast.fundamentals import build_macro
from spxlcast.live import refresh_once, settle_closed
from spxlcast.tracklog import append_log, load_log

from test_fix_model import _flags, _frame, _market
from test_fix_scorer import _after_close, _closes, _daily_log, _fc
from test_fix_web import _base, _write

UTC = timezone.utc
SECRET = "k" * 32


# ---- R1-19: urllib3's header-parse warning carries the request URL, FRED key included -------------
def test_a_fred_answer_urllib3_cannot_parse_logs_no_key_at_the_default_level(monkeypatch, caplog):
    body = json.dumps({"observations": [{"date": "2026-09-24", "value": "4.2"},
                                        {"date": "2026-09-25", "value": "4.3"}]}).encode()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def answer():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nthis line has no colon\r\n"
                     b"Connection: close\r\n\r\n" + body)
        conn.close()

    threading.Thread(target=answer, daemon=True).start()
    port, real = srv.getsockname()[1], requests.get
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(data.requests, "get", lambda url, *a, **k: real(
        url.replace("https://api.stlouisfed.org", f"http://127.0.0.1:{port}"), *a, **k))
    caplog.set_level(logging.WARNING)                                  # the CLI's default level
    try:
        assert len(data._fred_api("DGS10", 5.0, SECRET)) == 2
    finally:
        srv.close()
    assert SECRET not in caplog.text


# ---- R1-24: the column rewrite reads the log strictly; CR-only line ends are whole rows -----------
def test_a_stray_quote_fails_the_column_rewrite_and_leaves_the_log_alone(tmp_path):
    log = tmp_path / "log.csv"
    raw = _closes(n=45)
    _daily_log(str(log), raw, 40)
    lines = log.read_bytes().split(b"\r\n")
    lines[10] = lines[10].replace(b",HOLD,", b',"HOLD,', 1)           # a hand edit leaves an open quote
    log.write_bytes(b"\r\n".join(lines))
    before = log.read_bytes()
    d = raw.index[41]
    with pytest.raises(csv.Error):                                      # was: 29 rows merged into one cell
        append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(5, 21, 63)), str(log))
    assert log.read_bytes() == before and not glob.glob(str(log) + ".*")


@pytest.mark.parametrize("horizons", [(5, 21), (5, 21, 63)])
@pytest.mark.parametrize("torn", [False, True])
def test_a_log_with_cr_only_line_ends_keeps_its_track_record(tmp_path, horizons, torn):
    log, partial = tmp_path / "log.csv", tmp_path / "log.csv.partial"
    raw = _closes(n=45)
    _daily_log(str(log), raw, 40)
    text = log.read_bytes().replace(b"\r\n", b"\r")
    log.write_bytes(text[:-30] if torn else text)                     # torn: the last row cut short
    d = raw.index[41]
    append_log(_fc(raw[d], d.date(), _after_close(d), horizons=horizons), str(log))
    assert len(load_log(str(log))) == (40 if torn else 41)             # was: 1, the rest moved aside
    assert partial.exists() == torn
    if torn:
        assert partial.read_bytes().count(b"\n") == 1


# ---- R2-04: VIX3M/VIX6M without today's bar are moved with today's VIX ---------------------------
def _vol_market(shock, lag):
    snap = _market()
    cal = snap.calendar
    rng = np.random.default_rng(3)
    x = np.zeros(len(cal))
    for i in range(1, len(cal)):
        x[i] = 0.95 * x[i - 1] + 0.07 * rng.standard_normal()
    x[-1] = x[-2] + shock
    truth = {"^VIX3M": 19.0 * np.exp(0.7 * x + 0.01 * rng.standard_normal(len(cal))),
             "^VIX6M": 20.5 * np.exp(0.5 * x + 0.01 * rng.standard_normal(len(cal)))}
    snap.prices["^VIX"] = _frame(17.0 * np.exp(x), cal)
    for t, v in truth.items():
        snap.prices[t] = _frame(v[:len(cal) - lag], cal[:len(cal) - lag])
    return snap, {t: (v[-1], v[-1 - lag]) for t, v in truth.items()}


@pytest.mark.parametrize("shock", [0.4, -0.15])
def test_vol_pillars_without_todays_bar_move_with_the_vix(monkeypatch, shock):
    snap, truth = _vol_market(shock, lag=1)
    m = build_macro(snap, Config())
    for got, t in ((m.vix3m, "^VIX3M"), (m.vix6m, "^VIX6M")):
        now, stale = truth[t]
        assert abs(got - now) < 0.3 * abs(stale - now), t              # was: the stale close itself
    assert "moved with the VIX" in m.sources["vix_term"]
    assert _flags(monkeypatch, snap) == ["vix3m:stale", "vix6m:stale"]  # the inputs were still stale
    snap, truth = _vol_market(shock, lag=0)                               # all three from one session
    m = build_macro(snap, Config())
    assert (m.vix3m, m.vix6m) == (truth["^VIX3M"][0], truth["^VIX6M"][0]) and "vix_term" not in m.sources


# ---- R1-21: only a check made during the session closes the issue --------------------------------
@pytest.mark.parametrize("is_open", [True, False])
def test_main_tells_the_workflow_whether_it_saw_the_session(tmp_path, monkeypatch, is_open):
    monkeypatch.setattr(health, "gather", lambda base: (True, None, None, None, []))
    monkeypatch.setattr(health, "check", lambda *a, **k: ([], []))
    monkeypatch.setattr(health, "session_open", lambda now: is_open)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert health.main(["--url", "https://x"]) == 0
    assert out.read_text(encoding="utf-8") == f"live_checked={'true' if is_open else 'false'}\n"


def test_a_run_that_ends_after_the_bell_keeps_the_loops_last_quote_time(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "output", "forecast.json"), json.dumps(_base(spot=100.0, run_at="2026-09-29T18:40:05Z")))
    refresh_once(root, fetch=lambda t: 103.0, now=datetime(2026, 9, 29, 19, 59, 30, tzinfo=UTC))
    assert settle_closed(root)["session_open"] is False                    # 20:00: closed at the last quote
    # the 19:40 run lands at 20:05, after the bell
    _write(os.path.join(root, "output", "forecast.json"), json.dumps(_base(spot=101.0, run_at="2026-09-29T19:40:05Z")))
    live = settle_closed(root)
    assert (live["spot"], live["asof"], live["session_open"]) == (103.0, "2026-09-29T19:59:30+00:00", False)
    assert live["base_run_at"] == "2026-09-29T19:40:05Z" and settle_closed(root) is None
    # a check delayed to 20:30 UTC no longer reads this as a loop that stopped at 19:40
    assert health.check(datetime(2026, 9, 29, 20, 30, tzinfo=UTC), True, None, live, None) == ([], [])


# ---- R1-59: a quote that stops moving during the session is caught ------------------------------
def _ticks(root, quotes, start):
    for i, q in enumerate(quotes):
        refresh_once(root, fetch=lambda t, q=q: q, now=start + timedelta(minutes=i))
    with open(os.path.join(root, "output", "live.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_a_frozen_quote_is_a_problem_during_the_session(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "output", "forecast.json"), json.dumps(_base(spot=100.0, run_at="2026-09-29T13:40:05Z")))
    start = datetime(2026, 9, 29, 14, 0, tzinfo=UTC)                      # Tuesday 10:00 ET
    now = start + timedelta(minutes=36)
    live = _ticks(root, [103.0] * 36, start)
    assert live["spot_since"] == "2026-09-29T14:00:00+00:00"
    problems, _ = health.check(now, True, None, live, None)
    assert len(problems) == 1 and "has not changed since 2026-09-29 14:00 UTC" in problems[0]
    live = _ticks(root, [103.0, 103.01] * 18, start)                       # a quote that moves: fine
    assert health.check(now, True, None, live, None) == ([], [])
    # yesterday's closed block at the same price does not count as time the quote stood still
    closed = dict(live, spot=103.0, session_open=False, spot_since="2026-09-28T15:00:00+00:00")
    _write(os.path.join(root, "output", "live.json"), json.dumps(closed))
    assert _ticks(root, [103.0], start)["spot_since"] == "2026-09-29T14:00:00+00:00"
