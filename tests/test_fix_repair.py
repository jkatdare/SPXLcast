"""Regression tests for the repair round: problems the re-verification and the diff review found in
the first round of fixes (all offline)."""
import csv
import gzip
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import spxlcast.cli as cli
import spxlcast.data as data
from spxlcast import archive, health, live, serve, tracklog
from spxlcast.data import _bad_close, _finite, _merge_archive
from spxlcast.live import GRID_PERCENTILES, append_spot_log, refresh_once, settle_closed
from spxlcast.tracklog import _archived_grids, append_log, load_log, score_log

from test_fix_data import _bars
from test_fix_infra import FAKE_AZ, FAKE_GIT, FAKE_PYTHON, ROOT
from test_fix_model import _flags, _market, _patch_market, _run_cli
from test_fix_scorer import _after_close, _closes, _daily_log, _fc
from test_fix_web import FRI_1559, Site, _base, _live_doc, _write

UTC = timezone.utc


# ---- `python -m spxlcast` passes main()'s exit code on (daily.sh depends on it) ------------------
def test_python_m_spxlcast_exits_with_the_code_main_returns(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    log.write_text("run_at,spot_date,spot_status,spot\n2026-09-24T21:40:00Z,2026-09-24,close,100\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "fetch_prices", lambda *a, **k: ({}, None))     # the download came back empty
    monkeypatch.setattr(sys, "argv", ["spxlcast", "score", "--log-file", str(log)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("spxlcast", run_name="__main__")
    assert exc.value.code == 1


# ---- append_log: blank lines above the header (a data-loss regression of round 1) ---------------
def test_blank_lines_above_the_header_do_not_wipe_the_track_record(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes(n=30)
    _daily_log(path, raw, 20)
    data_ = open(path, "rb").read()
    for lead in (b"\r\n", b"\n", b"\r\n\r\n", b"  \r\n"):
        open(path, "wb").write(lead + data_)
        d = raw.index[25]
        append_log(_fc(raw[d], d.date(), _after_close(d)), path)
        assert len(load_log(path)) == 21, lead
    open(path, "wb").write(b"\r\n" + data_)                  # a rewrite under a wider header keeps every row
    d = raw.index[26]
    append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(5, 21, 63)), path)
    df = load_log(path)
    assert len(df) == 21 and df["h63_q50"].notna().sum() == 1 and df["h5_q50"].notna().all()
    assert not os.path.exists(path + ".partial")
    torn = b"\r\n" + data_ + data_.splitlines(keepends=True)[1].rstrip()     # a whole last row missing its end
    open(path, "wb").write(torn)
    append_log(_fc(raw[d], d.date(), _after_close(d)), path)
    assert not os.path.exists(path + ".partial") and len(pd.read_csv(path)) == 22
    open(path, "wb").write(b"  \r\n\r\n")                     # blank lines only: no track record to keep
    append_log(_fc(raw[d], d.date(), _after_close(d)), path)
    assert len(load_log(path)) == 1 and open(path).readline().startswith("run_at,")


def test_a_log_that_cannot_be_appended_still_writes_the_outputs_and_fails_the_run(monkeypatch, tmp_path, capsys):
    def full_disk(fc, path):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(cli, "append_log", full_disk)
    assert _run_cli(monkeypatch, tmp_path) == 1
    assert json.loads((tmp_path / "out" / "forecast.json").read_text(encoding="utf-8"))["etf"] == "SPXL"
    assert (tmp_path / "out" / "fan.png").read_bytes()[:4] == b"\x89PNG"
    out = capsys.readouterr().out
    assert "was not logged" in out and "No space left" in out and "appended to" not in out


def test_appended_to_line_prints_a_log_path_literally(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "render_all", lambda *a, **k: None)
    _patch_market(monkeypatch, _market())
    assert cli.main(["forecast", "--paths", "200", "--horizons", "5", "--rating-horizon", "5", "--no-news",
                     "--log-file", "[bold]log.csv"]) == 0
    assert "appended to [bold]log.csv" in capsys.readouterr().out


# ---- the scorer ---------------------------------------------------------------------------------
def test_todays_intraday_run_is_named_when_it_is_the_first_to_resolve(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes(n=40)
    old = raw.index[-8]
    append_log(_fc(raw[old], old.date(), _after_close(old), horizons=(21,)), path)    # 1M: 14 sessions to go
    today = raw.index[-1] + pd.offsets.BDay(1)
    append_log(_fc(raw.iloc[-1], today.date(), datetime(today.year, today.month, today.day, 15, tzinfo=UTC),
                   status="intraday"), path)
    rep = score_log(path, closes=raw)
    note = next(n for n in rep.notes if n.startswith("nothing to score yet"))
    assert f"the 1W forecast from {today.date()}, resolves in 6 sessions" in note


def test_a_download_across_the_bell_drops_the_partial_bar(tmp_path, monkeypatch):
    path = str(tmp_path / "log.csv")
    raw = _closes(n=30, start="2026-03-02")
    today = raw.index[-1]
    d = raw.index[-6]                                        # its 1W window ends today
    append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(5,)), path)
    monkeypatch.setattr(data, "fetch_prices", lambda *a, **k: ({"SPXL": pd.DataFrame({"Close": raw})}, None))
    cfg = tracklog.Config(cache_dir=str(tmp_path / "cache"))
    for states, scored in (([True, False], 0), ([False, True], 0), ([False, False], 1)):
        it = iter(states)
        monkeypatch.setattr(data, "session_state", lambda *a: (str(today.date()), next(it)))
        assert score_log(path, cfg=cfg).n_scoreable == scored, states


def test_run_at_in_another_iso_form_is_read(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes(n=30)
    _daily_log(path, raw, 3)
    df = pd.read_csv(path, dtype=str)
    df.loc[2, "run_at"] = "2026-01-06 21:40:00"               # a hand edit in the other ISO form
    df.to_csv(path, index=False)
    assert load_log(path)["run_at"].notna().all()


def _grid_record(run_at, terminal, percentiles=None):
    pcts = list(GRID_PERCENTILES) if percentiles is None else percentiles
    return {"run_at": run_at, "forecast": {"forecast": {"5": {"grid": {"percentiles": pcts, "terminal": terminal}}}}}


def test_two_runs_in_one_second_fall_back_to_the_logged_quantiles(tmp_path):
    root = str(tmp_path)
    stamp = "2026-09-25T20:40:07Z"
    archive.write_run(root, _grid_record(stamp, list(np.linspace(90, 110, 103))))
    assert 5 in _archived_grids(root, pd.Timestamp(stamp))
    archive.write_run(root, _grid_record(stamp, list(np.linspace(80, 120, 103))))
    assert _archived_grids(root, pd.Timestamp(stamp)) is None      # the row cannot tell which run is its


def test_malformed_grids_are_left_out(tmp_path):
    root = str(tmp_path)
    fractions = [p / 100 for p in GRID_PERCENTILES]              # levels as fractions: not percent levels
    archive.write_run(root, _grid_record("2026-09-25T20:40:07Z", list(np.linspace(90, 110, 103)), fractions))
    assert _archived_grids(root, pd.Timestamp("2026-09-25T20:40:07Z")) == {}
    archive.write_run(root, _grid_record("2026-09-25T20:40:08Z", [100.0, 101.0], [40.0, 60.0]))   # no 5-95 span
    assert _archived_grids(root, pd.Timestamp("2026-09-25T20:40:08Z")) == {}
    folder = os.path.join(root, "runs", "2026-09-25")
    with gzip.open(os.path.join(folder, "204009Z.json.gz"), "wt") as fh:      # an integer too big for a float
        fh.write('{"forecast": {"forecast": {"5": {"grid": {"percentiles": [1, ' + "9" * 400 + '], '
                 '"terminal": [1, 2]}}}}}')
    assert _archived_grids(root, pd.Timestamp("2026-09-25T20:40:09Z")) == {}
    with gzip.open(os.path.join(folder, "204010Z.json.gz"), "wt") as fh:      # nested past the recursion limit
        fh.write("[" * 100_000 + "]" * 100_000)
    assert _archived_grids(root, pd.Timestamp("2026-09-25T20:40:10Z")) is None


def test_a_failed_archive_write_leaves_no_empty_record(tmp_path, monkeypatch):
    def refuse(src, dst):
        raise OSError(5, "share gone")
    monkeypatch.setattr(archive.os, "replace", refuse)
    with pytest.raises(OSError):
        archive.write_run(str(tmp_path), _grid_record("2026-09-25T20:40:07Z", [1.0, 2.0]))
    assert os.listdir(tmp_path / "runs" / "2026-09-25") == []


def test_intraday_row_on_an_ex_date_takes_the_later_close_rows_basis():
    raw = pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2026-03-02", periods=3))
    adj = raw.copy()
    adj.iloc[:1] *= 0.99                                     # a distribution goes ex on the second day
    df = pd.DataFrame({"spot_date": raw.index, "spot": [100.0, 100.5, 102.0],
                       "spot_status": ["close", "intraday", "close"]})
    pos, basis = tracklog._price_basis(df, adj)
    assert basis[1] == pytest.approx(1.0)                    # the ex-date's own (unadjusted) basis


# ---- the rating backtest: a rating difference needs three independent windows each ----------------
def test_rating_difference_needs_three_independent_windows():
    sys.path.insert(0, str(ROOT / "scripts"))
    import backtest_rating as br
    y = pd.Series(np.linspace(-0.2, 0.3, 60))
    pos = pd.Series(np.arange(60) * 21)
    two_lone_months = pd.Series(np.isin(np.arange(60), [10, 40]))
    assert all(np.isnan(v) for v in br._diff_ci(y, two_lone_months, pos, 126))
    four = pd.Series(np.isin(np.arange(60), [5, 20, 35, 50]))
    assert all(np.isfinite(v) for v in br._diff_ci(y, four, pos, 126))
    o = pd.DataFrame({"date": pd.to_datetime(["2026-08-31"]), "pos": [0]})       # no outcome at any horizon
    assert "no month-end has an outcome" in br.calibration_section(o)[-1]
    assert "no month-end has an outcome" in br.skill_section(o)[-1]


def test_shiller_download_is_read_before_it_is_cached(tmp_path, monkeypatch):
    import requests
    sys.path.insert(0, str(ROOT / "scripts"))
    import backtest_drift as bd
    from test_fix_backtest import _Resp
    page = '<a href="http://[bad">x</a><a href="https://x.wsimg.com/ie_data.xls">d</a>'
    truncated = b"\xd0\xcf\x11\xe0" + b"\x00" * 600                  # OLE2 magic, then nothing readable

    def fake_get(url, *a, **k):
        return _Resp(page) if url == bd.PAGE else _Resp(content=truncated)
    monkeypatch.setattr(requests, "get", fake_get)
    assert bd.shiller_url() == "https://x.wsimg.com/ie_data.xls"   # a malformed href is skipped
    path = tmp_path / "ie_data.xls"
    with pytest.raises(RuntimeError, match="readable workbook"):
        bd.load_shiller(path)
    assert not path.exists()
    path.write_bytes(b"cached")                                     # --refresh never deletes the old copy first
    with pytest.raises(RuntimeError, match="readable workbook"):
        bd.load_shiller(path, refresh=True)
    assert path.read_bytes() == b"cached"


# ---- data ----------------------------------------------------------------------------------------
def test_an_archived_session_the_download_lacks_is_put_on_its_basis():
    days = pd.bdate_range(end="2026-09-23", periods=60)
    raw = _bars(100 * 1.001 ** np.arange(60), days)
    new = (raw * 0.99).iloc[10:].drop(days[30])                   # a distribution; Yahoo left out one session
    merged = _merge_archive(raw, new)
    assert np.allclose(merged["Close"], raw["Close"] * 0.99) and days[30] in merged.index
    exdate = raw.copy()                                           # an ex-date right after the hole: basis unknown
    exdate.loc[:days[30], ["Open", "High", "Low", "Close"]] *= 0.99
    assert days[30] not in _merge_archive(raw, exdate.iloc[10:].drop(days[30])).index
    same = raw.iloc[10:].drop(days[30])                           # no basis change: kept exactly as archived
    assert _merge_archive(raw, same).loc[days[30], "Close"] == raw.loc[days[30], "Close"]


def test_index_levels_cannot_print_zero_but_yields_and_futures_can():
    close = pd.Series([15.0, 0.0, -1.0])
    assert _bad_close("^VIX", close).tolist() == [False, True, True]
    assert _bad_close("^TNX", close).tolist() == [False, False, False]
    assert _bad_close("CL=F", close).tolist() == [False, False, False]
    assert _finite(10 ** 400) is None and _finite("31.5") == 31.5


def test_the_first_run_after_an_early_close_does_not_reuse_the_pre_open_download(tmp_path, monkeypatch):
    from spxlcast.cache import Cache
    calls = []

    def download(tickers, **kw):
        calls.append(1)
        return pd.concat({t: _bars([100.0, 101.0], pd.bdate_range(end="2026-11-26", periods=2)) for t in tickers}, axis=1)
    monkeypatch.setattr(data.yf, "download", download)
    cache = Cache(str(tmp_path))
    for clock in ("2026-11-27 08:40", "2026-11-27 13:40", "2026-11-27 14:40"):     # 13:00 close that day
        monkeypatch.setattr(data, "ny_now", lambda c=clock: pd.Timestamp(c, tz="America/New_York"))
        data.fetch_prices(["SPXL"], "5y", cache, ttl_hours=6.0)
    assert len(calls) == 2                                        # pre-open, then once after the close


def test_vol_pillars_are_checked_against_the_freshest_of_them(monkeypatch):
    # SPXL has no bar yet today while ^VIX does; VIX3M and VIX6M are yesterday's
    assert _flags(monkeypatch, _market(spxl_lag=1, vol_lag=1)) == ["vix3m:stale", "vix6m:stale"]


# ---- the live loop -------------------------------------------------------------------------------
def test_two_loops_that_find_the_spot_log_emptied_write_one_header(tmp_path, monkeypatch):
    root = str(tmp_path)
    path = tmp_path / "logs" / "spot_log.csv"
    _write(str(path), "")
    real = live.os.path.getsize
    monkeypatch.setattr(live.os.path, "getsize", lambda p: 0)     # both loops looked before either wrote
    append_spot_log(root, _live_doc(spot=103.0))
    append_spot_log(root, _live_doc(spot=104.0))
    monkeypatch.setattr(live.os.path, "getsize", real)
    text = path.read_text(encoding="utf-8")
    assert text.count("ts_utc") == 1
    assert pd.read_csv(path)["spot"].tolist() == [103.0, 104.0]


def test_a_row_torn_between_cr_and_lf_gets_no_blank_line(tmp_path):
    root = str(tmp_path)
    append_spot_log(root, _live_doc(spot=103.0))
    path = tmp_path / "logs" / "spot_log.csv"
    path.write_bytes(path.read_bytes()[:-1])
    append_spot_log(root, _live_doc(spot=104.0))
    assert [len(r) for r in csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline=""))] == [8, 8, 8]


def test_settle_keeps_the_last_quote_of_a_live_json_written_before_run_ids_were_normalised(tmp_path):
    root = str(tmp_path)
    base = _base(spot=100.0, run_at="2026-09-25T19:40:03.123456+00:00")
    _write(os.path.join(root, "output", "forecast.json"), json.dumps(base))
    doc = _live_doc(spot=103.0)
    doc["base_run_at"] = base["run_at"]                         # the old, raw form
    _write(os.path.join(root, "output", "live.json"), json.dumps(doc))
    assert settle_closed(root)["spot"] == 103.0


def test_a_run_during_a_quote_outage_replaces_the_previous_runs_block(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "output", "forecast.json"), json.dumps(_base(spot=100.0, label="HOLD")))
    refresh_once(root, fetch=lambda t: 103.0, now=FRI_1559)
    new = _base(spot=95.0, run_at="2026-09-28T13:40:05+00:00", label="SELL")
    _write(os.path.join(root, "output", "forecast.json"), json.dumps(new))
    got = refresh_once(root, fetch=lambda t: None)
    assert got["spot"] == 95.0 and got["rating"]["label"] == "SELL" and got["base_run_at"] == "2026-09-28T13:40:05Z"
    assert refresh_once(root, fetch=lambda t: None) is None      # nothing new: no rewrite every minute
    assert len(pd.read_csv(tmp_path / "logs" / "spot_log.csv")) == 1   # no quote, no price row


# ---- the status page -----------------------------------------------------------------------------
def test_page_follows_the_nyse_calendar():
    assert serve._session_hours(datetime(2026, 9, 28, 15, 0, tzinfo=UTC))          # Monday 11:00 ET
    assert not serve._session_hours(datetime(2026, 11, 26, 16, 0, tzinfo=UTC))     # Thanksgiving
    assert not serve._session_hours(datetime(2026, 11, 27, 18, 30, tzinfo=UTC))    # 13:30 ET, 13:00 close


def test_a_deeply_nested_live_json_drops_only_the_block(tmp_path):
    path = tmp_path / "live.json"
    path.write_text("[" * 100_000 + "]" * 100_000)
    assert serve.render_live(str(path)) == ""


@pytest.fixture
def site(tmp_path):
    _write(str(tmp_path / "output" / "report.txt"), "RATING HOLD")
    _write(str(tmp_path / "output" / "fan.png"), b"\x89PNG chart", "wb")
    s = Site(tmp_path)
    yield s
    s.server.shutdown()


def test_a_file_changed_within_the_last_seconds_gets_no_last_modified(site):
    png = site.path("output", "fan.png")
    status, headers, _ = site.request("/output/fan.png")
    assert status == 200 and "last-modified" not in headers and headers["etag"]
    os.utime(png, (os.path.getmtime(png) - 3600,) * 2)
    status, headers, _ = site.request("/output/fan.png")
    assert "last-modified" in headers
    assert site.request("/output/fan.png", headers={"If-Modified-Since": headers["last-modified"]})[0] == 304


def test_a_trailing_slash_or_an_error_never_carries_the_files_etag(site):
    etag = site.request("/output/report.txt")[1]["etag"]
    for path in ("/output/report.txt/", "/nope"):
        status, headers, _ = site.request(path, headers={"If-None-Match": etag})
        assert status == 404 and "etag" not in headers, path


# ---- health -------------------------------------------------------------------------------------
def test_cron_steps_and_weekday_ranges():
    assert health.parse_cron("40 13-21 * * 1/2")[2] == [1, 3, 5]       # no Sunday from '1/2'
    assert health.parse_cron("40 13 * * */2")[2] == [0, 2, 4, 6]
    assert health.parse_cron("40 13 * * 0-7")[2] == [0, 1, 2, 3, 4, 5, 6]
    for bad in ("5/ 13 * * 1-5", "40 13-21/ * * 1-5", "*/ 13 * * 1-5"):
        with pytest.raises(ValueError, match="empty step"):
            health.parse_cron(bad)


def test_every_copy_of_the_job_schedule_matches():
    doc = (ROOT / "infra" / "DEPLOY.md").read_text(encoding="utf-8")
    ps1 = (ROOT / "infra" / "deploy.ps1").read_text(encoding="utf-8")
    copies = re.findall(r'--cron-expression "(\d[^"]*)"', doc) + re.findall(r"cron (\d[^(]*?) \(UTC\)", ps1)
    assert copies and all(c.strip() == health.SCHEDULE for c in copies), copies


def test_a_spot_date_cut_short_is_not_read_as_a_stale_feed():
    now = datetime(2026, 9, 29, 17, 5, tzinfo=UTC)
    rows = [{"run_at": "2026-09-29T16:40:20Z", "spot_date": "2026-09-2", "data_flags": ""}]
    problems, _ = health.check(now, True, rows, {"asof": "2026-09-29T17:04:00+00:00"}, None,
                               lookback=timedelta(minutes=30))
    assert not any("stale" in p for p in problems)


# ---- infra ---------------------------------------------------------------------------------------
def test_the_dirty_check_covers_everything_the_image_copies():
    sources = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY "):
            sources += line.split()[1:-1]
    for text in ((ROOT / "infra" / "deploy.ps1").read_text(encoding="utf-8"),
                 (ROOT / "infra" / "DEPLOY.md").read_text(encoding="utf-8")):
        checked = re.search(r"status --porcelain --untracked-files=normal ([^)]+)\)", text).group(1).split()
        assert set(sources) <= set(checked), (sources, checked)


def test_dockerignore_also_drops_env_variants_and_data_folders_at_any_depth():
    from test_fix_infra import _dockerignored
    patterns = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    for path in (".env.local", "spxlcast/.env.production", "spxlcast/output/r.txt", "scripts/archive/a.jsonl",
                 "infra/logs/l.txt"):
        assert _dockerignored(path, patterns), path
    for path in ("spxlcast/output.py", "spxlcast/archive.py", "spxlcast/cli.py"):
        assert not _dockerignored(path, patterns), path


SLOW_PYTHON = FAKE_PYTHON.replace('[ "$FAKE_FORECAST" = ok ] && { echo "NEW REPORT"; exit 0; }',
                                  '[ "$FAKE_FORECAST" = ok ] && { echo "REPORT $RUN part 1"; sleep 2; '
                                  'echo "REPORT $RUN part 2"; exit 0; }')


def test_overlapping_daily_runs_each_publish_a_whole_report(tmp_path):
    dash = shutil.which("dash")
    if not dash:
        pytest.skip("needs dash (/bin/sh in the image)")
    app, data_dir, bindir = tmp_path / "app", tmp_path / "data", tmp_path / "bin"
    for d in (app, data_dir / "output", bindir):
        d.mkdir(parents=True, exist_ok=True)
    (bindir / "python").write_bytes(SLOW_PYTHON.encode())
    (bindir / "python").chmod(0o755)
    script = tmp_path / "daily.sh"
    script.write_bytes((ROOT / "infra/daily.sh").read_bytes().replace(b"/app", app.as_posix().encode()))
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ.get("PATH", ""),
               SPXLCAST_DATA=data_dir.as_posix(), FAKE_FORECAST="ok", FAKE_SCORE="ok")
    procs = [subprocess.Popen([dash, script.as_posix()], env=dict(env, RUN=r), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) for r in ("A", "B")]
    outs = [p.communicate(timeout=60)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    report = (data_dir / "output" / "report.txt").read_text()
    assert report in ("REPORT A part 1\nREPORT A part 2\n", "REPORT B part 1\nREPORT B part 2\n")
    assert sorted(p.name for p in (data_dir / "output").iterdir()) == ["report.txt", "score.txt"]
    assert not [p for p in data_dir.iterdir() if p.name.startswith(("report.txt.", "score.txt."))]


def test_deploy_refuses_a_template_placeholder_it_does_not_fill(tmp_path):
    ps = shutil.which("powershell")
    if sys.platform != "win32" or not ps:
        pytest.skip("needs Windows PowerShell")
    work, bindir, temp = tmp_path / "repo", tmp_path / "bin", tmp_path / "temp"
    for d in (work / "infra", bindir, temp):
        d.mkdir(parents=True)
    for f in ("deploy.ps1", "job.yaml", "web.yaml"):
        shutil.copy(ROOT / "infra" / f, work / "infra" / f)
    with open(work / "infra" / "job.yaml", "a", encoding="utf-8") as fh:
        fh.write("# extra: ${KEY2}\n")
    (bindir / "az.cmd").write_text(FAKE_AZ.replace("\n", "\r\n"))
    (bindir / "git.cmd").write_text(FAKE_GIT.replace("\n", "\r\n"))
    windir = os.environ.get("SystemRoot", r"C:\Windows")
    path = os.pathsep.join([str(bindir), rf"{windir}\System32", windir, rf"{windir}\System32\WindowsPowerShell\v1.0"])
    env = dict(os.environ, PATH=path, TEMP=str(temp), TMP=str(temp), FRED_API_KEY="fakefredkey0123456789")
    proc = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-File", r"infra\deploy.ps1"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=120)
    if "running scripts is disabled" in proc.stdout + proc.stderr:
        pytest.skip("PowerShell execution policy blocks local scripts")
    calls = (bindir / "calls.log").read_text().splitlines() if (bindir / "calls.log").exists() else []
    assert proc.returncode != 0 and "${KEY2}" in proc.stdout + proc.stderr
    assert not any(c.startswith(("containerapp job create", "containerapp create")) for c in calls)
    assert list(temp.iterdir()) == []
