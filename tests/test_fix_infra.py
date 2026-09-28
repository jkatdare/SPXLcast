"""Regression tests for the infra fixes: the image survives a CRLF checkout, daily.sh publishes the
report and score only when their step succeeds, deploy.ps1 works in Windows PowerShell 5.1, the
.dockerignore and placeholder checks, schedules kept in sync, and the health check's coverage and
robustness. Everything is offline: fake data, fake az/git/python, and 127.0.0.1 servers."""
import http.client
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from spxlcast import health
from spxlcast.health import check, gather, parse_cron, scheduled_runs
from spxlcast.serve import make_handler

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
TUE = datetime(2026, 9, 29, tzinfo=UTC)
PAGE = b"<!doctype html><html><body><h1>SPXLcast</h1>report</body></html>"


def rows_for(slots, spot_date=None):
    return [{"run_at": (s + timedelta(seconds=25)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "spot_date": spot_date or s.date().isoformat(), "data_flags": ""} for s in slots]


def fresh(now):
    return {"asof": (now - timedelta(minutes=1)).isoformat()}


def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------------------
# Dependencies and schedules (R1-60 drift, R1-20, R1-21)
# ---------------------------------------------------------------------------------------
def test_health_check_needs_only_the_standard_library():
    # the workflow installs nothing; health.py shares the NYSE calendar in spxlcast/config.py
    code = ("import sys, spxlcast.health; print(sorted({m.split('.')[0] for m in sys.modules} & "
            "{'numpy', 'pandas', 'yfinance', 'requests', 'scipy', 'rich', 'matplotlib'}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr


def test_schedules_agree_across_workflows_scripts_and_docs():
    deploy = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    assert re.search(r'CRON:\s*"([^"]+)"', deploy).group(1) == health.SCHEDULE
    ps1 = (ROOT / "infra/deploy.ps1").read_text(encoding="utf-8")
    assert re.search(r'\$Cron = "([^"]+)"', ps1).group(1) == health.SCHEDULE
    doc = (ROOT / "infra/DEPLOY.md").read_text(encoding="utf-8")
    assert re.search(r'CRON\s*=\s*"([^"]+)"', doc).group(1) == health.SCHEDULE
    hc = (ROOT / ".github/workflows/healthcheck.yml").read_text(encoding="utf-8")
    assert tuple(re.findall(r'-\s*cron:\s*"([^"]+)"', hc)) == health.CHECKS
    # only a check made during the session sees the live loop at work, so only its pass closes the issue (R1-21)
    assert "if: steps.check.outcome == 'success' && steps.check.outputs.live_checked == 'true'" in hc


def test_the_tests_before_a_deploy_run_on_the_python_the_image_ships():
    image = re.search(r"^FROM python:(\d+\.\d+\.\d+)-", (ROOT / "Dockerfile").read_text(encoding="utf-8"), re.M)
    deploy = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    assert image and re.search(r'python-version:\s*"([^"]+)"', deploy).group(1) == image.group(1)


def test_every_run_is_judged_by_two_checks_so_a_dropped_or_late_check_loses_nothing():
    mon, next_tue = datetime(2026, 9, 28, tzinfo=UTC), datetime(2026, 10, 6, tzinfo=UTC)
    week = scheduled_runs(mon, mon + timedelta(days=5))
    checks = sorted(t for expr in health.CHECKS for t in scheduled_runs(mon, next_tue, expr))
    all_runs = scheduled_runs(mon - timedelta(days=4), next_tue)

    def reporters(slot, times):
        rows = rows_for([s for s in all_runs if s != slot])
        stamp = slot.strftime("%Y-%m-%d %H:%M UTC")
        return [t for t in times if slot < t < slot + timedelta(days=4)
                and any(stamp in p for p in check(t, True, rows, fresh(t), None)[0])]

    for slot in week:
        assert len(reporters(slot, checks)) == 2, slot       # so dropping any one check still reports it
    fri_2040 = datetime(2026, 10, 2, 20, 40, tzinfo=UTC)
    no_fri_evening = [t for t in checks if t != datetime(2026, 10, 2, 22, 15, tzinfo=UTC)]
    assert reporters(fri_2040, no_fri_evening) == [datetime(2026, 10, 5, 15, 5, tzinfo=UTC)]
    # a 22:15 check that starts 70 minutes late still covers 19:40 (it used to miss it from +55 min)
    late = TUE.replace(hour=23, minute=25)
    assert reporters(TUE.replace(hour=19, minute=40), [late]) == [late]


# ---------------------------------------------------------------------------------------
# check(): R1-56, R1-58
# ---------------------------------------------------------------------------------------
def test_forecast_json_is_not_judged_while_the_newest_run_may_still_be_writing():
    now = TUE.replace(hour=15, minute=41, second=30)
    rows = rows_for(scheduled_runs(now - timedelta(days=4), now))       # 15:40 row just appended
    previous = {"run_at": "2026-09-29T14:40:25+00:00"}
    assert check(now, True, rows, fresh(now), previous) == ([], [])
    later = now + timedelta(minutes=15)                                  # still not written: it died
    problems, _ = check(later, True, rows, fresh(later), previous)
    assert len(problems) == 1 and "did not finish" in problems[0] and "15:40 UTC" in problems[0]


def test_stale_price_feed_counts_sessions():
    thu = datetime(2026, 10, 1, 19, 5, tzinfo=UTC)
    rows = rows_for(scheduled_runs(thu - timedelta(days=4), thu), spot_date="2026-09-28")   # Monday's close
    problems, _ = check(thu, True, rows, fresh(thu), None)
    assert len(problems) == 1 and "3 sessions ago" in problems[0]
    # Tuesday after the MLK holiday, winter, 14:40 run missing: the newest row is the 8:40 pre-open run
    # priced Friday; only the missed run is reported, not a stale feed
    tue = datetime(2026, 1, 20, 15, 5, tzinfo=UTC)
    missed = datetime(2026, 1, 20, 14, 40, tzinfo=UTC)
    rows = rows_for([s for s in scheduled_runs(tue - timedelta(days=4), tue) if s != missed], "2026-01-16")
    problems, _ = check(tue, True, rows, fresh(tue), None)
    assert len(problems) == 1 and "14:40 UTC" in problems[0]
    xmas = datetime(2026, 12, 25, 22, 15, tzinfo=UTC)                     # a Friday holiday
    rows = rows_for(scheduled_runs(xmas - timedelta(days=4), xmas), "2026-12-24")
    last_quote = {"asof": "2026-12-24T17:59:00+00:00"}                     # 12:59 ET, the eve's early close
    assert check(xmas, True, rows, last_quote, None) == ([], [])


def test_live_checks_follow_the_nyse_calendar_the_loop_uses():
    # Thanksgiving: the loop does not tick, so the in-session checks expect nothing new
    thanksgiving = datetime(2026, 11, 26, 17, 5, tzinfo=UTC)
    rows = rows_for(scheduled_runs(thanksgiving - timedelta(days=4), thanksgiving), "2026-11-25")
    assert check(thanksgiving, True, rows, {"asof": "2026-11-25T20:59:30+00:00"}, None) == ([], [])
    # the day after closes at 13:00 ET (18:00 UTC): the 19:05 check wants the loop to have lasted until then
    friday = datetime(2026, 11, 27, 19, 5, tzinfo=UTC)
    rows = rows_for(scheduled_runs(friday - timedelta(days=4), friday), "2026-11-27")
    assert check(friday, True, rows, {"asof": "2026-11-27T17:59:10+00:00"}, None) == ([], [])
    problems, _ = check(friday, True, rows, {"asof": "2026-11-27T16:30:00+00:00"}, None)
    assert len(problems) == 1 and "2026-11-27 18:00 UTC" in problems[0]


# ---------------------------------------------------------------------------------------
# gather(): R1-02, R1-22, R1-48, R1-57, R1-61 (fake transport)
# ---------------------------------------------------------------------------------------
LOG = ("run_at,spot_date,data_flags\n2026-09-25T20:41:10Z,2026-09-25,\n2026-09-25T21:41:15Z,2026-09-25,\n")


def fake_site(monkeypatch, overrides=None):
    pages = {"/healthz": b"ok", "/": PAGE, "/logs/forecast_log.csv": LOG.encode(),
             "/output/live.json": b'{"asof": "2026-09-25T19:59:00+00:00"}',
             "/output/forecast.json": b'{"run_at": "2026-09-25T21:41:15+00:00"}'}
    pages.update(overrides or {})

    def _get(url, timeout=60.0):
        body = pages.get(url[len("https://x"):])
        if isinstance(body, Exception):
            raise body
        if body is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return body
    monkeypatch.setattr(health, "_get", _get)
    return gather("https://x")


def test_gather_healthy_site(monkeypatch):
    healthz_ok, rows, live, forecast, errors = fake_site(monkeypatch)
    assert healthz_ok and errors == [] and len(rows) == 2 and live and forecast


def test_gather_requires_the_real_page(monkeypatch):
    errors = fake_site(monkeypatch, {"/": b"<html>Azure Container Apps - 404</html>"})[-1]
    assert len(errors) == 1 and "not the status page" in errors[0]
    errors = fake_site(monkeypatch, {"/": http.client.RemoteDisconnected("closed")})[-1]
    assert len(errors) == 1 and "did not load" in errors[0]


def test_gather_drops_a_torn_last_line_and_a_bom(monkeypatch):
    torn = ("\ufeff" + LOG + "2026-09-28T13:20:05Z,2026-09-2").encode("utf-8")    # spot_date cut mid-write
    _, rows, live, forecast, errors = fake_site(monkeypatch, {"/logs/forecast_log.csv": torn})
    assert errors == [] and [r["run_at"] for r in rows] == ["2026-09-25T20:41:10Z", "2026-09-25T21:41:15Z"]
    now = datetime(2026, 9, 28, 13, 25, tzinfo=UTC)                                  # Monday before the open
    assert check(now, True, rows, live, forecast, lookback=timedelta(hours=1)) == ([], [])


def test_gather_reports_non_object_json_instead_of_crashing(monkeypatch):
    for body in (b"[1, 2]", b"null", b'"text"', b"3", b"[" * 100_000 + b"]" * 100_000, b"{trunc"):
        _, _, live, forecast, errors = fake_site(monkeypatch, {"/output/forecast.json": body,
                                                               "/output/live.json": body})
        assert live == {} and forecast is None        # live.json is there but unreadable
        assert len(errors) == 1 and "forecast.json is not a JSON object" in errors[0]
        after_close = check(datetime(2026, 9, 29, 22, 15, tzinfo=UTC), True, None, live, None)[0]
        assert len(after_close) == 1 and "live.json is unreadable" in after_close[0]


def test_gather_reports_a_failed_score_step(monkeypatch):
    note = b"2026-09-29T15:41:12Z\nTraceback (most recent call last):\nKeyError: 'run_at'\n"
    errors = fake_site(monkeypatch, {"/output/score_error.txt": note})[-1]
    assert len(errors) == 1 and "2026-09-29T15:41:12Z" in errors[0] and "KeyError: 'run_at'" in errors[0]


# ---------------------------------------------------------------------------------------
# gather() against real sockets: R1-02 and R1-48 with the actual status page
# ---------------------------------------------------------------------------------------
def test_health_check_reads_the_real_status_page(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "output" / "report.txt").write_text("RATING HOLD", encoding="utf-8")
    (tmp_path / "output" / "score.txt").write_text("track record", encoding="utf-8")
    (tmp_path / "output" / "forecast.json").write_text('{"run_at": "2026-09-25T21:41:15+00:00"}', encoding="utf-8")
    (tmp_path / "logs" / "forecast_log.csv").write_text(LOG, encoding="utf-8")
    server, base = serve(make_handler(str(tmp_path)))
    try:
        healthz_ok, rows, _, forecast, errors = gather(base)
        assert healthz_ok and errors == [] and len(rows) == 2 and forecast     # PAGE_MARKER matches serve.py
        (tmp_path / "output" / "score_error.txt").write_text("2026-09-29T15:41:12Z\nKeyError: 'run_at'\n",
                                                            encoding="utf-8")
        errors = gather(base)[-1]
        assert len(errors) == 1 and "score step failed" in errors[0]          # serve.py must keep serving it
    finally:
        server.shutdown()
        server.server_close()


class _PageCrashes(BaseHTTPRequestHandler):
    """/healthz answers 'ok' but '/' closes the connection without a response (R1-02)."""

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path != "/":
            self.send_error(404)
        self.close_connection = True

    def log_message(self, *args):
        pass


def test_a_page_that_gives_no_response_fails_the_check_while_healthz_is_ok():
    server, base = serve(_PageCrashes)
    try:
        healthz_ok, _, _, _, errors = gather(base)
    finally:
        server.shutdown()
        server.server_close()
    assert healthz_ok and any("did not load" in e for e in errors)


# ---------------------------------------------------------------------------------------
# parse_cron and main(): R1-60, R1-63
# ---------------------------------------------------------------------------------------
def test_cron_parser_supports_steps_and_rejects_bad_fields():
    assert parse_cron("*/30 13-21/2 * * 1-5") == ([0, 30], [13, 15, 17, 19, 21], [1, 2, 3, 4, 5])
    assert parse_cron("5/20 9 * * *")[0] == [5, 25, 45]
    for bad in ("40 13-21 * * MON-FRI", "40 21-13 * * 1-5", "40 13-25 * * 1-5", "60 13 * * 1-5",
                "40 13-21 * * 5-1", "40 13 * * 1-9", "@hourly", "0 40 13-21 * * 1-5", "*/0 13 * * 1"):
        with pytest.raises(ValueError):
            parse_cron(bad)


def test_a_bad_schedule_is_rejected_before_any_fetch(monkeypatch, capsys):
    monkeypatch.setattr(health, "gather", lambda base: pytest.fail("fetched the site"))
    with pytest.raises(SystemExit) as exc:
        health.main(["--url", "https://x", "--schedule", "40 13-21 * * MON-FRI"])
    assert exc.value.code == 2 and "--schedule" in capsys.readouterr().err


def test_report_is_written_into_a_new_directory_and_a_write_failure_keeps_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "gather", lambda base: (True, None, None, None, []))
    monkeypatch.setattr(health, "check", lambda *a, **k: ([], []))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    out = tmp_path / "new" / "dir" / "report.md"
    assert health.main(["--url", "https://x", "--output", str(out)]) == 0
    assert "all checks passed" in out.read_text(encoding="utf-8")
    (tmp_path / "file").write_text("x")
    assert health.main(["--url", "https://x", "--output", str(tmp_path / "file" / "report.md")]) == 0


# ---------------------------------------------------------------------------------------
# Image and build context: R1-45, R1-49
# ---------------------------------------------------------------------------------------
def test_daily_sh_is_lf_in_every_checkout_and_the_image_strips_crs(tmp_path):
    attrs = subprocess.run(["git", "check-attr", "eol", "--", "infra/daily.sh", "Dockerfile"],
                           cwd=ROOT, capture_output=True, text=True)
    if attrs.returncode == 0:
        assert attrs.stdout.splitlines() == ["infra/daily.sh: eol: lf", "Dockerfile: eol: lf"]
    run = next(line for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
               if line.startswith("RUN ") and "daily.sh" in line)
    sed = re.search(r"sed -i '([^']+)' \./daily\.sh", run)
    assert sed and run.index("sed") < run.index("chmod")
    sed_bin, dash = shutil.which("sed"), shutil.which("dash")
    if not (sed_bin and dash):
        pytest.skip("needs sed and dash")
    copy = tmp_path / "daily.sh"
    copy.write_bytes((ROOT / "infra/daily.sh").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    subprocess.run([sed_bin, "-b", "-i", sed.group(1), copy.as_posix()], check=True)   # -b: Windows sed would drop CRs itself
    assert b"\r" not in copy.read_bytes()
    assert subprocess.run([dash, "-n", copy.as_posix()]).returncode == 0


def _dockerignored(path, patterns):
    """az acr build's reading of .dockerignore: each pattern is anchored at the context root; a path is
    excluded when it or a parent directory matches, the last matching pattern winning."""
    def rx(p):
        out = "".join(".*" if tok == "**" else "[^/]*" if tok == "*" else "[^/]" if tok == "?" else re.escape(tok)
                      for tok in re.split(r"(\*\*|\*|\?)", p))
        return re.compile(f"^{out}$")
    rules = [(p.startswith("!"), rx(p.lstrip("!"))) for p in patterns]
    parts = path.split("/")
    excluded = False
    for i in range(1, len(parts) + 1):
        prefix = "/".join(parts[:i])
        for negated, r in rules:
            if r.match(prefix):
                excluded = not negated
    return excluded


def test_dockerignore_excludes_secrets_and_junk_at_any_depth_but_keeps_what_the_image_copies():
    patterns = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    for path in (".env", "prod.env", "infra/.env", "infra/staging.env", ".claude/worktrees/wf/.env",
                 ".claude/settings.local.json", "spxlcast/__pycache__/cli.cpython-313.pyc", "spxlcast/stray.pyc",
                 "sub/.cache/prices.pkl", "spxlcast.egg-info/PKG-INFO", "output/report.txt", "tests/test_x.py"):
        assert _dockerignored(path, patterns), path
    for path in ("Dockerfile", "requirements.txt", "pyproject.toml", "spxlcast/cli.py", "scripts/backtest_rating.py",
                 "infra/daily.sh"):
        assert not _dockerignored(path, patterns), path


# ---------------------------------------------------------------------------------------
# daily.sh under dash with a fake python: R1-47, R1-48
# ---------------------------------------------------------------------------------------
FAKE_PYTHON = """#!/bin/sh
# fake `python -m spxlcast forecast|score ...`; FAKE_FORECAST / FAKE_SCORE = ok | fail
cp "$SPXLCAST_DATA/output/report.txt" "$SPXLCAST_DATA/seen_during_$3.txt" 2>/dev/null
if [ "$3" = forecast ]; then
    [ "$FAKE_FORECAST" = ok ] && { echo "NEW REPORT"; exit 0; }
    echo "Traceback (most recent call last):" >&2; echo "RuntimeError: No price history for SPXL" >&2; exit 1
fi
[ "$FAKE_SCORE" = ok ] && { echo "NEW SCORE"; exit 0; }
echo "Traceback (most recent call last):" >&2; echo "KeyError: 'run_at'" >&2; exit 1
"""


def run_daily(tmp_path, forecast="ok", score="ok"):
    dash = shutil.which("dash")
    if not dash:
        pytest.skip("needs dash (/bin/sh in the image)")
    app, data, bindir = tmp_path / "app", tmp_path / "data", tmp_path / "bin"
    for d in (app, data / "output", bindir):
        d.mkdir(parents=True, exist_ok=True)
    fake = bindir / "python"
    fake.write_bytes(FAKE_PYTHON.encode())
    fake.chmod(0o755)
    script = tmp_path / "daily.sh"
    script.write_bytes((ROOT / "infra/daily.sh").read_bytes().replace(b"/app", app.as_posix().encode()))
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ.get("PATH", ""),
               SPXLCAST_DATA=data.as_posix(), FAKE_FORECAST=forecast, FAKE_SCORE=score)
    return subprocess.run([dash, script.as_posix()], env=env, capture_output=True, text=True, timeout=60), data


def seed(tmp_path):
    out = tmp_path / "data" / "output"
    out.mkdir(parents=True)
    (out / "report.txt").write_text("OLD REPORT\n")
    (out / "score.txt").write_text("OLD SCORE\n")
    return out


def test_daily_run_replaces_the_report_only_when_done(tmp_path):
    out = seed(tmp_path)
    proc, data = run_daily(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (data / "seen_during_forecast.txt").read_text() == "OLD REPORT\n"     # not blanked mid-run
    assert (out / "report.txt").read_text() == "NEW REPORT\n" and (out / "score.txt").read_text() == "NEW SCORE\n"
    assert sorted(p.name for p in out.iterdir()) == ["report.txt", "score.txt"]


def test_a_failed_forecast_keeps_the_last_report_off_the_page(tmp_path):
    out = seed(tmp_path)
    proc, data = run_daily(tmp_path, forecast="fail")
    assert proc.returncode == 1 and "No price history" in proc.stdout
    assert (out / "report.txt").read_text() == "OLD REPORT\n" and (out / "score.txt").read_text() == "OLD SCORE\n"
    assert "No price history" in (data / "last_error.txt").read_text()
    assert not (data / "seen_during_score.txt").exists()                          # score did not run


def test_a_failed_score_keeps_the_last_track_record_and_is_reported(tmp_path, monkeypatch):
    out = seed(tmp_path)
    proc, _ = run_daily(tmp_path, score="fail")
    assert proc.returncode == 0 and "score step FAILED" in proc.stdout
    assert (out / "report.txt").read_text() == "NEW REPORT\n" and (out / "score.txt").read_text() == "OLD SCORE\n"
    note = (out / "score_error.txt").read_bytes()
    assert re.match(rb"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\r?\n", note) and b"KeyError" in note
    errors = fake_site(monkeypatch, {"/output/score_error.txt": note})[-1]
    assert len(errors) == 1 and "KeyError: 'run_at'" in errors[0]
    proc, _ = run_daily(tmp_path)                                                   # the next good run
    assert proc.returncode == 0 and not (out / "score_error.txt").exists()
    assert (out / "score.txt").read_text() == "NEW SCORE\n"


# ---------------------------------------------------------------------------------------
# deploy.ps1 and DEPLOY.md: R1-46, R1-50, R1-51
# ---------------------------------------------------------------------------------------
def _fill_keys(text):
    start = text.index("$fill = @{")
    return set(re.findall(r"(?:^|@\{|;)\s*([A-Z0-9_]+)\s*=", text[start:text.index("}\n", start)], re.MULTILINE))


def test_every_template_placeholder_is_filled_and_the_doc_check_passes_on_a_full_render():
    templates = {f: (ROOT / "infra" / f).read_text(encoding="utf-8") for f in ("job.yaml", "web.yaml")}
    used = {k for t in templates.values() for k in re.findall(r"\$\{([A-Z0-9_]+)\}", t)}
    ps1 = (ROOT / "infra/deploy.ps1").read_text(encoding="utf-8")
    doc = (ROOT / "infra/DEPLOY.md").read_text(encoding="utf-8")
    assert used and used <= _fill_keys(ps1) and used <= _fill_keys(doc)
    pattern = re.search(r"Select-String -Path \S+ -Pattern '([^']+)'", doc).group(1)
    for text in templates.values():
        rendered = re.sub(r"\$\{([A-Z0-9_]+)\}", "value", text)
        assert not re.search(pattern, rendered, re.IGNORECASE)          # Select-String ignores case
        assert re.search(pattern, rendered + "extra: ${NEW_KEY}\n", re.IGNORECASE)    # a key $fill lacks
        assert re.search(pattern, rendered + "extra: ${KEY2}\n", re.IGNORECASE)


FAKE_AZ = r"""@echo off
setlocal
echo(%*>>"%~dp0calls.log"
set "A=%*"
if not "%A:--query=%"=="%A%" goto query
if not "%A:job show=%"=="%A%" goto missing
if not "%A:containerapp show=%"=="%A%" goto missing
if not defined FAKE_FAIL_CREATE exit /b 0
if not "%A:job create=%"=="%A%" goto failcreate
exit /b 0
:missing
>&2 echo ERROR: (ResourceNotFound) The Resource was not found.
exit /b 1
:failcreate
>&2 echo ERROR: (InvalidParameterValue) bad spec
exit /b 1
:query
if not "%A:[0].name=%"=="%A%" exit /b 0
echo fakevalue123
exit /b 0
"""
FAKE_GIT = r"""@echo off
if "%1"=="rev-parse" (echo 0123456789ab& exit /b 0)
if "%1"=="status" if defined FAKE_DIRTY (echo  M spxlcast/cli.py& exit /b 0)
exit /b 0
"""


def run_deploy(tmp_path, **env_extra):
    ps = shutil.which("powershell")
    if sys.platform != "win32" or not ps:
        pytest.skip("needs Windows PowerShell")
    work, bindir, temp = tmp_path / "repo", tmp_path / "bin", tmp_path / "temp"
    for d in (work / "infra", bindir, temp):
        d.mkdir(parents=True)
    for f in ("deploy.ps1", "job.yaml", "web.yaml"):
        shutil.copy(ROOT / "infra" / f, work / "infra" / f)
    (bindir / "az.cmd").write_text(FAKE_AZ.replace("\n", "\r\n"))
    (bindir / "git.cmd").write_text(FAKE_GIT.replace("\n", "\r\n"))
    windir = os.environ.get("SystemRoot", r"C:\Windows")
    # only the fakes are reachable: the real az and git are not on this PATH
    path = os.pathsep.join([str(bindir), rf"{windir}\System32", windir, rf"{windir}\System32\WindowsPowerShell\v1.0"])
    env = dict(os.environ, PATH=path, TEMP=str(temp), TMP=str(temp), FRED_API_KEY="fakefredkey0123456789",
               **env_extra)
    proc = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-File", r"infra\deploy.ps1"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=120)
    if "running scripts is disabled" in proc.stdout + proc.stderr:
        pytest.skip("PowerShell execution policy blocks local scripts")
    calls = (bindir / "calls.log").read_text().splitlines() if (bindir / "calls.log").exists() else []
    return proc, calls, list(temp.iterdir())


def test_deploy_creates_new_apps_in_windows_powershell_and_stamps_a_clean_build(tmp_path):
    proc, calls, left = run_deploy(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert any(c.startswith("containerapp job create") for c in calls)
    assert any(c.startswith("containerapp create") for c in calls)
    assert any("SPXLCAST_BUILD=0123456789ab " in c for c in calls if c.startswith("acr build"))
    assert left == []                                                  # rendered specs (with secrets) removed


def test_deploy_stamps_a_dirty_build_and_removes_the_specs_when_it_fails(tmp_path):
    proc, calls, left = run_deploy(tmp_path, FAKE_DIRTY="1", FAKE_FAIL_CREATE="1")
    assert proc.returncode != 0
    assert any("SPXLCAST_BUILD=012345-dirty " in c for c in calls if c.startswith("acr build"))
    assert left == []
