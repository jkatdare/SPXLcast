"""External health check of the hosted site: is every scheduled run landing in the track record,
is the minute-by-minute live price fresh during the session, and is the page up?

Runs from a scheduled GitHub Action (.github/workflows/healthcheck.yml), outside Azure, so it
catches any break in the chain (job, file share, web app, DNS, certificate). Standard library only,
so the workflow needs no dependencies:

    python -m spxlcast.health --url https://spxlcast.com [--output problems.md]

Exit code 1 when something is wrong; data problems flagged by the latest run are warnings only.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from .config import nyse_session      # the NYSE calendar the live loop follows (standard library only too)

# The job's schedule in UTC. Keep in sync with CRON in .github/workflows/deploy.yml (a test checks).
SCHEDULE = "40 13-21 * * 1-5"
# When this check itself runs, in UTC. Keep in sync with .github/workflows/healthcheck.yml (a test checks).
CHECKS = ("5 15,17,19 * * 1-5", "15 22 * * 1-5")
NY = ZoneInfo("America/New_York")
RUN_GRACE = timedelta(minutes=15)       # a scheduled run should be in the log this long after its start
RUN_WINDOW = timedelta(minutes=20)      # a logged run counts for a slot if it started this soon after it
LIVE_MAX_AGE = timedelta(minutes=10)    # the live loop writes every minute of the session
SPOT_FROZEN_AGE = timedelta(minutes=30)  # SPXL's minute quotes repeated for at most 6 minutes in a row (Sep 2026)
SPOT_MAX_AGE_SESSIONS = 2               # a pre-open run prices the previous session; one more of slack
PAGE_MARKER = "<h1>SPXLcast</h1>"       # in every render of the status page (spxlcast/serve.py PAGE)
FETCH_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)


# ---------------------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------------------
def _field(spec: str, lo: int, hi: int, open_hi: Optional[int] = None) -> List[int]:
    """Values of one cron field; ``open_hi`` ends '*' and 'a/n' (6 for weekdays, where 7 is a
    second Sunday that only a range or a single value may name)."""
    values = set()
    for part in spec.split(","):
        rng, slash, step = part.partition("/")
        if slash and not step:
            raise ValueError(f"'{part}' has an empty step")
        if rng == "*":
            a, b = lo, hi if open_hi is None else open_hi
        elif "-" in rng:
            a, b = (int(x) for x in rng.split("-", 1))
        else:
            a = int(rng)
            b = (hi if open_hi is None else open_hi) if step else a
        n = int(step) if step else 1
        if not lo <= a <= b <= hi or n < 1:
            raise ValueError(f"'{part}' is reversed or outside {lo}-{hi}")
        values.update(range(a, b + 1, n))
    return sorted(values)


def parse_cron(expr: str) -> Tuple[List[int], List[int], List[int]]:
    """(minutes, hours, cron weekdays with 0 = Sunday) of a 'M H * * DOW' expression: numbers, '*',
    ranges, lists and steps (no names)."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"cron expression {expr!r} must have 5 fields")
    minute, hour, dom, month, dow = fields
    if dom != "*" or month != "*":
        raise ValueError("only '*' is supported for day-of-month and month")
    try:
        return _field(minute, 0, 59), _field(hour, 0, 23), sorted({d % 7 for d in _field(dow, 0, 7, open_hi=6)})
    except ValueError as exc:
        raise ValueError(f"cron expression {expr!r}: {exc}") from None


def scheduled_runs(start: datetime, end: datetime, expr: str = SCHEDULE) -> List[datetime]:
    """Scheduled start times in (start, end], UTC."""
    minutes, hours, dows = parse_cron(expr)
    out = []
    day = start.date()
    while day <= end.date():
        if (day.weekday() + 1) % 7 in dows:      # Python Monday = 0; cron Monday = 1, Sunday = 0
            for h in hours:
                for m in minutes:
                    slot = datetime(day.year, day.month, day.day, h, m, tzinfo=timezone.utc)
                    if start < slot <= end:
                        out.append(slot)
        day += timedelta(days=1)
    return out


def window_start(now: datetime) -> datetime:
    """Where a check at ``now`` starts looking for missed runs: the scheduled check before the previous
    one, less the grace. Every run is then judged by two checks, so a dropped or late check (GitHub
    delays and sometimes drops scheduled workflows) loses nothing; Monday's first check covers Friday
    evening."""
    checks = sorted(t for expr in CHECKS for t in scheduled_runs(now - timedelta(days=7), now, expr))
    return checks[-3] - RUN_GRACE


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def _utc(stamp: str) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(stamp).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _fmt(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _sessions_after(start: date, end: date) -> int:
    """NYSE sessions on the days d with start < d <= end."""
    return sum(nyse_session(start + timedelta(days=i)) is not None for i in range(1, (end - start).days + 1))


def session_open(now: datetime) -> bool:
    """Is the regular NYSE session open (holidays and 13:00 early closes included)? The live loop
    writes only then."""
    ny = now.astimezone(NY)
    hours = nyse_session(ny.date())
    return hours is not None and hours[0] <= ny.time() < hours[1]


def last_close(now: datetime) -> datetime:
    """The latest NYSE session close at or before ``now``."""
    day = now.astimezone(NY).date()
    while True:
        hours = nyse_session(day)
        if hours is not None:
            close = datetime.combine(day, hours[1], tzinfo=NY)
            if close <= now:
                return close
        day -= timedelta(days=1)


def check(now: datetime, healthz_ok: bool, rows: Optional[List[Dict[str, str]]],
          live: Optional[dict], forecast: Optional[dict], lookback: Optional[timedelta] = None,
          schedule: str = SCHEDULE, errors: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """(problems, warnings) for the state of the site at ``now``. Missed runs are looked for since
    ``now - lookback``, by default since window_start(now)."""
    problems: List[str] = list(errors)
    warnings: List[str] = []
    if not healthz_ok:
        problems.append("the status page's /healthz did not answer 'ok'")

    # Every scheduled run in the window must have logged a row.
    if rows is not None:
        runs = sorted(t for t in (_utc(r.get("run_at", "")) for r in rows) if t is not None)
        if not runs:
            problems.append("the forecast log has no rows")
        else:
            start = now - lookback if lookback is not None else window_start(now)
            for slot in scheduled_runs(start, now - RUN_GRACE, schedule):
                if not any(slot - timedelta(minutes=2) <= t <= slot + RUN_WINDOW for t in runs):
                    problems.append(f"no forecast logged for the run scheduled at {_fmt(slot)} "
                                    f"(latest logged run: {_fmt(runs[-1])})")
            newest = max(rows, key=lambda r: _utc(r.get("run_at", "")) or datetime.min.replace(tzinfo=timezone.utc))
            try:    # fromisoformat, unlike strptime, rejects a date cut short ("2026-09-2")
                spot_day = date.fromisoformat(newest["spot_date"])
            except (KeyError, TypeError, ValueError):
                spot_day = None
            if spot_day is not None:
                stale = _sessions_after(spot_day, now.astimezone(NY).date())
                if stale > SPOT_MAX_AGE_SESSIONS:
                    problems.append(f"the latest run priced SPXL as of {newest['spot_date']}, {stale} sessions ago: "
                                    f"the price feed looks stale")
            flags = (newest.get("data_flags") or "").strip()
            if flags:
                warnings.append(f"the latest run ({newest.get('run_at')}) flagged data problems: {flags}")
            # A run appends its row before it writes forecast.json: judge only runs that had time to finish.
            settled = [t for t in runs if t <= now - RUN_GRACE]
            if forecast is not None and settled:
                f_at = _utc(forecast.get("run_at", ""))
                if f_at is None or f_at < settled[-1] - timedelta(minutes=5):
                    problems.append(f"output/forecast.json (run {forecast.get('run_at')}) is older than the "
                                    f"latest logged run ({_fmt(settled[-1])}): that run did not finish its outputs")

    # The live loop must be fresh while the session is open, and must have run until the last close.
    asof = _utc(live.get("asof", "")) if live else None
    if session_open(now):
        if asof is None:
            problems.append("output/live.json is missing or unreadable during the session")
        elif now - asof > LIVE_MAX_AGE:
            problems.append(f"output/live.json is {int((now - asof).total_seconds() // 60)} minutes old "
                            f"during the session (the live loop should update it every minute)")
        else:
            since = _utc(live.get("spot_since", ""))
            if since is not None and now - since > SPOT_FROZEN_AGE:
                problems.append(f"the live SPXL quote has not changed since {_fmt(since)} "
                                f"({int((now - since).total_seconds() // 60)} minutes): "
                                f"the quote feed looks frozen")
    elif live is not None:
        close = last_close(now)
        if asof is None:
            problems.append("output/live.json is unreadable: it is not a JSON object with the time it was written")
        elif asof < close - LIVE_MAX_AGE:
            problems.append(f"output/live.json was last written at {_fmt(asof)}, before the last session close "
                            f"({_fmt(close)}): the live loop stopped during the session")
    return problems, warnings


# ---------------------------------------------------------------------------------------
# Fetching and reporting
# ---------------------------------------------------------------------------------------
def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "spxlcast-healthcheck", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def gather(base: str) -> Tuple[bool, Optional[List[Dict[str, str]]], Optional[dict], Optional[dict], List[str]]:
    errors: List[str] = []
    healthz_ok = False
    try:
        healthz_ok = _get(f"{base}/healthz").strip() == b"ok"
    except FETCH_ERRORS as exc:
        errors.append(f"could not reach {base}/healthz: {exc}")
    # /healthz is a constant; only the page itself shows that it renders.
    try:
        if PAGE_MARKER not in _get(f"{base}/").decode("utf-8", errors="replace"):
            errors.append(f"{base}/ answered but is not the status page")
    except FETCH_ERRORS as exc:
        errors.append(f"the status page {base}/ did not load: {exc}")
    rows = None
    try:
        text = _get(f"{base}/logs/forecast_log.csv").decode("utf-8-sig", errors="replace")
        if not text.endswith("\n"):     # a torn last line while the job appends: drop it
            text = text[:text.rfind("\n") + 1]
        rows = list(csv.DictReader(io.StringIO(text)))
    except FETCH_ERRORS as exc:
        errors.append(f"could not fetch the forecast log: {exc}")
    docs = {}
    for name in ("live", "forecast"):
        try:
            raw = _get(f"{base}/output/{name}.json")
        except FETCH_ERRORS as exc:
            docs[name] = None               # a missing live.json only matters in check()
            if name == "forecast":
                errors.append(f"could not fetch output/{name}.json: {exc}")
            continue
        try:
            doc = json.loads(raw)
        except (ValueError, RecursionError):
            doc = None
        if isinstance(doc, dict):
            docs[name] = doc
        elif name == "forecast":
            docs[name] = None
            errors.append("output/forecast.json is not a JSON object")
        else:
            docs[name] = {}                 # there but unreadable: check() reports it, in or out of the session
    # The job leaves this note (UTC time, then the tail of the output) when the score step failed; 404 otherwise.
    try:
        note = _get(f"{base}/output/score_error.txt").decode("utf-8", errors="replace").strip().splitlines()
    except FETCH_ERRORS:
        note = []
    if note:
        errors.append(f"the latest score step failed at {note[0]} (the page shows the previous track record): "
                      f"{note[-1][:200]}")
    return healthz_ok, rows, docs["live"], docs["forecast"], errors


def render(base: str, now: datetime, problems: List[str], warnings: List[str]) -> str:
    head = (f"Health check of {base} at {_fmt(now)}: "
            + ("**FAILED**" if problems else "all checks passed") + "\n")
    lines = [head]
    if problems:
        lines += ["Problems:", ""] + [f"- {p}" for p in problems] + [""]
    if warnings:
        lines += ["Warnings:", ""] + [f"- {w}" for w in warnings] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m spxlcast.health", description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("SPXLCAST_URL", "https://spxlcast.com"))
    ap.add_argument("--lookback-hours", type=float, default=None,
                    help="check the scheduled runs of the last N hours (default: since the scheduled check "
                         "before the previous one)")
    ap.add_argument("--schedule", default=SCHEDULE, help=f"the job's cron schedule in UTC (default '{SCHEDULE}')")
    ap.add_argument("--output", default=None, help="also write the report (markdown) to this file")
    a = ap.parse_args(argv)
    try:
        parse_cron(a.schedule)
    except ValueError as exc:
        ap.error(f"--schedule: {exc}")
    base = a.url.rstrip("/")
    now = datetime.now(timezone.utc)
    healthz_ok, rows, live, forecast, errors = gather(base)
    lookback = timedelta(hours=a.lookback_hours) if a.lookback_hours is not None else None
    problems, warnings = check(now, healthz_ok, rows, live, forecast, lookback, a.schedule, errors)
    text = render(base, now, problems, warnings)
    print(text)
    for target, mode in ((a.output, "w"), (os.environ.get("GITHUB_STEP_SUMMARY"), "a")):
        if not target:
            continue
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            with open(target, mode, encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError as exc:      # the checks' verdict stands; say why the report is missing
            print(f"warning: could not write the report to {target}: {exc}", file=sys.stderr)
    # Only a check made while the session is open sees the live loop at work: the workflow closes its
    # issue on such a pass only (not on a holiday, after an early close or delayed past the bell).
    if os.environ.get("GITHUB_OUTPUT"):
        try:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                fh.write(f"live_checked={'true' if session_open(now) else 'false'}\n")
        except OSError as exc:
            print(f"warning: could not write the step output: {exc}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
