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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

# The job's schedule in UTC. Keep in sync with CRON in .github/workflows/deploy.yml.
SCHEDULE = "40 13-21 * * 1-5"
NY = ZoneInfo("America/New_York")
RUN_GRACE = timedelta(minutes=15)       # a scheduled run should be in the log this long after its start
RUN_WINDOW = timedelta(minutes=20)      # a logged run counts for a slot if it started this soon after it
LIVE_MAX_AGE = timedelta(minutes=10)    # the live loop writes every minute of the session
SPOT_MAX_AGE_DAYS = 4                   # a long weekend plus a holiday


# ---------------------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------------------
def _field(spec: str, lo: int, hi: int) -> List[int]:
    values = set()
    for part in spec.split(","):
        if part == "*":
            values.update(range(lo, hi + 1))
        elif "-" in part:
            a, b = part.split("-", 1)
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(part))
    return sorted(values)


def parse_cron(expr: str) -> Tuple[List[int], List[int], List[int]]:
    """(minutes, hours, cron weekdays with 0 = Sunday) of a 'M H * * DOW' expression."""
    minute, hour, dom, month, dow = expr.split()
    if dom != "*" or month != "*":
        raise ValueError("only '*' is supported for day-of-month and month")
    return _field(minute, 0, 59), _field(hour, 0, 23), sorted({d % 7 for d in _field(dow, 0, 7)})


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


def session_open(now: datetime) -> bool:
    """Regular US session (weekday, 9:30-16:00 New York). Holidays count as open: the live loop
    keeps writing on them too, so they cannot cause a false alarm."""
    ny = now.astimezone(NY)
    return ny.weekday() < 5 and (ny.hour, ny.minute) >= (9, 30) and ny.hour < 16


def check(now: datetime, healthz_ok: bool, rows: Optional[List[Dict[str, str]]],
          live: Optional[dict], forecast: Optional[dict], lookback: timedelta = timedelta(hours=3.5),
          schedule: str = SCHEDULE, errors: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """(problems, warnings) for the state of the site at ``now``."""
    problems: List[str] = list(errors)
    warnings: List[str] = []
    if not healthz_ok:
        problems.append("the status page's /healthz did not answer 'ok'")

    # Every scheduled run in the lookback window must have logged a row.
    if rows is not None:
        runs = sorted(t for t in (_utc(r.get("run_at", "")) for r in rows) if t is not None)
        if not runs:
            problems.append("the forecast log has no rows")
        else:
            for slot in scheduled_runs(now - lookback, now - RUN_GRACE, schedule):
                if not any(slot - timedelta(minutes=2) <= t <= slot + RUN_WINDOW for t in runs):
                    problems.append(f"no forecast logged for the run scheduled at {_fmt(slot)} "
                                    f"(latest logged run: {_fmt(runs[-1])})")
            newest = max(rows, key=lambda r: _utc(r.get("run_at", "")) or datetime.min.replace(tzinfo=timezone.utc))
            try:
                spot_age = (now.astimezone(NY).date() - datetime.strptime(newest["spot_date"], "%Y-%m-%d").date()).days
            except (KeyError, TypeError, ValueError):   # a torn last line while the job appends
                spot_age = None
            if spot_age is not None and spot_age > SPOT_MAX_AGE_DAYS:
                problems.append(f"the latest run priced SPXL as of {newest['spot_date']}, {spot_age} days ago: "
                                f"the price feed looks stale")
            flags = (newest.get("data_flags") or "").strip()
            if flags:
                warnings.append(f"the latest run ({newest.get('run_at')}) flagged data problems: {flags}")
            if forecast is not None:
                f_at = _utc(forecast.get("run_at", ""))
                if f_at is None or f_at < runs[-1] - timedelta(minutes=5):
                    problems.append(f"output/forecast.json (run {forecast.get('run_at')}) is older than the "
                                    f"latest logged run ({_fmt(runs[-1])}): that run did not finish its outputs")

    # The live loop must be fresh while the session is open.
    if session_open(now):
        asof = _utc(live.get("asof", "")) if live else None
        if asof is None:
            problems.append("output/live.json is missing or unreadable during the session")
        elif now - asof > LIVE_MAX_AGE:
            problems.append(f"output/live.json is {int((now - asof).total_seconds() // 60)} minutes old "
                            f"during the session (the live loop should update it every minute)")
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
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        errors.append(f"could not reach {base}/healthz: {exc}")
    rows = None
    try:
        text = _get(f"{base}/logs/forecast_log.csv").decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        errors.append(f"could not fetch the forecast log: {exc}")
    docs = {}
    for name in ("live", "forecast"):
        try:
            docs[name] = json.loads(_get(f"{base}/output/{name}.json"))
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            docs[name] = None
            if name == "forecast":
                errors.append(f"could not fetch output/forecast.json: {exc}")
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
    ap.add_argument("--lookback-hours", type=float, default=3.5,
                    help="check the scheduled runs of the last N hours (default 3.5, the gap between checks)")
    ap.add_argument("--schedule", default=SCHEDULE, help=f"the job's cron schedule in UTC (default '{SCHEDULE}')")
    ap.add_argument("--output", default=None, help="also write the report (markdown) to this file")
    a = ap.parse_args(argv)
    base = a.url.rstrip("/")
    now = datetime.now(timezone.utc)
    healthz_ok, rows, live, forecast, errors = gather(base)
    problems, warnings = check(now, healthz_ok, rows, live, forecast, timedelta(hours=a.lookback_hours),
                               a.schedule, errors)
    text = render(base, now, problems, warnings)
    print(text)
    for target in (a.output, os.environ.get("GITHUB_STEP_SUMMARY")):
        if target:
            with open(target, "a" if target != a.output else "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
