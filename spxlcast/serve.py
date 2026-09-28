"""Tiny status page for the hosted run: shows the live minute-by-minute re-pricing, the latest full
report, fan chart, track-record score and offers the CSV/JSON for download. Standard library only (the
market calendar comes from config, which is too), read-only, no forms.

    python -m spxlcast serve --root /data --port 8000 --live

Expects the scheduled job to have written, under ``root``:
    output/report.txt   output/forecast.json   output/fan.png   output/score.txt
    logs/forecast_log.csv
and, with ``--live`` (or SPXLCAST_LIVE=1), keeps writing every minute of the session:
    output/live.json    logs/spot_log.csv
Only files inside ``output/`` and ``logs/`` are served; everything else under ``root`` is not.
"""
from __future__ import annotations

import argparse
import email.utils
import html
import json
import logging
import math
import os
import signal
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

from .config import nyse_session

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SPXLcast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<style>
body{{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0f1419;color:#e6e6e6;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px}} .meta{{color:#9aa;font-size:13px;margin-bottom:16px}}
pre{{background:#151b22;border:1px solid #2a323c;border-radius:6px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.35}}
img{{max-width:100%;border-radius:6px;border:1px solid #2a323c}} a{{color:#7cc4ff}}
.row{{display:flex;flex-wrap:wrap;gap:16px}} .row>div{{flex:1 1 480px}}
.live{{background:#151b22;border:1px solid #2a323c;border-radius:6px;padding:12px 16px;margin-bottom:16px}}
.live .spot{{font-size:28px;font-weight:bold}} .live .lvl{{color:#7cc4ff}}
.live table{{border-collapse:collapse;margin-top:8px;font-size:13px}} .live th,.live td{{padding:3px 10px;text-align:right;border-bottom:1px solid #2a323c}}
.live th:first-child,.live td:first-child{{text-align:left}} .dim{{color:#9aa}}
</style></head><body>
<h1>SPXLcast</h1>
<div class="meta">latest full run {stamp} &nbsp;|&nbsp; {links}</div>
{live}
<div class="row"><div><h2>Latest full forecast</h2><pre>{report}</pre></div>
<div><h2>Fan chart</h2>{chart}<h2>Track record</h2><pre>{score}</pre></div></div>
</body></html>"""

LINKS = (("logs", "forecast_log.csv"), ("logs", "spot_log.csv"), ("output", "forecast.json"),
         ("output", "live.json"), ("output", "report.txt"))

# Sent with every response. The ingress cannot add headers, so the app does.
SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; "
                                "base-uri 'none'; form-action 'none'"),
    ("Strict-Transport-Security", "max-age=31536000"),
)


def _read(path: str, limit: int = 200_000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return "(not available yet)"


def _horizon_label(days: int) -> str:
    if days % 21 == 0:
        m = days // 21
        return f"{m}M" if m % 12 else f"{m // 12}Y"
    if days < 21 and days % 5 == 0:
        return f"{days // 5}W"
    return f"{days}d"


def _ordinal(n: float) -> str:
    """Same as report.ordinal (this module stays standard-library only)."""
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fmt_ny(asof_ny: str) -> str:
    """'2026-09-22 14:31:05 EDT' -> '2:31 PM ET, 22 Sep'."""
    try:
        t = datetime.strptime(asof_ny[:19], "%Y-%m-%d %H:%M:%S")
        return t.strftime("%I:%M %p ET, %d %b").lstrip("0")
    except ValueError:
        return html.escape(asof_ny)


def _session_hours(now: datetime) -> bool:
    """Is the NYSE regular session open (holidays closed, early closes at 13:00)."""
    ny = now.astimezone(NY)
    hours = nyse_session(ny.date())
    return hours is not None and hours[0] <= ny.time() < hours[1]


def render_live(path: str, now: datetime | None = None) -> str:
    """The live block: latest quote, the distribution restated at it, and the fixed price checks.
    A live.json this cannot render drops the block, never the page."""
    try:
        with open(path, encoding="utf-8") as fh:
            live = json.load(fh)
    except (OSError, ValueError, RecursionError):
        return ""
    try:
        return _live_block(live, now or datetime.now(timezone.utc))
    except Exception as exc:  # noqa: BLE001 - whatever the file holds, the rest of the page must still load
        log.warning("cannot render %s: %r", path, exc)
        return ""


def _live_block(live: dict, now: datetime) -> str:
    try:
        asof = datetime.fromisoformat(live["asof"])
        age = (now - (asof if asof.tzinfo else asof.replace(tzinfo=timezone.utc))).total_seconds() / 60.0
    except (KeyError, TypeError, ValueError):
        age = float("nan")
    # the calendar, not the stored flag: a loop that stopped, or got no quote since the open, leaves a
    # flag from the last session, and the block must then read open and stale
    is_open = _session_hours(now)
    state = "market open" if is_open else "market closed"
    stale = f' <span class="dim">(stale: {age:.0f} min old)</span>' if is_open and age > 5 else ""
    spot = float(live["spot"])
    parts = [
        '<div class="live">',
        f'<div><span class="spot">{html.escape(str(live.get("etf") or "SPXL"))} {spot:,.2f}</span> '
        f'<span class="dim">as of {_fmt_ny(str(live["asof_ny"]))}, {state}{stale}</span></div>',
        f'<div>{_assessment_line(live.get("assessment"))}{live["change_vs_base"]:+.2%} since the full run '
        f'at {live["base_spot"]:,.2f}</div>',
    ]
    horizons = sorted(live.get("horizons", {}).values(), key=lambda x: x["horizon"])
    levels = sorted({row["price"] for h in horizons for row in h.get("price_lookup", [])})
    head = "".join(f"<th>{p:,.0f}: pctile / P(reach)</th>" for p in levels)
    rows = []
    for h in horizons:
        q = h["quantile_prices"]
        cells = "".join(_check_cell(row, spot) for row in h.get("price_lookup", []))
        rows.append(f'<tr><td>{_horizon_label(int(h["horizon"]))}</td><td>{q["5.0"]:,.0f}</td><td>{q["50.0"]:,.0f}</td>'
                    f'<td>{q["95.0"]:,.0f}</td><td>{h["p_positive"]:.0%}</td><td>{h["p_drawdown_20"]:.0%}</td>{cells}</tr>')
    parts.append('<table><tr><th>Horizon</th><th>Low end (5% below)</th><th>Typical</th><th>High end (5% above)</th>'
                 f'<th>P(up)</th><th>P(-20% dip)</th>{head}</tr>' + "".join(rows) + "</table>")
    parts.append('<div class="dim">Prices come from the last full run, rescaled to the live price in between runs (full '
                 'runs are hourly in the session); leverage cost, drawdown risk, P(up) and P(-20% dip) change only with a '
                 f'full run. The levels compare today with the months {_span(_period(live))}: low, normal and high (or '
                 'elevated) are the bottom, middle and top third of them. P(reach) is the chance the price dips to a level '
                 'below the quote, or rises to one above it, before the horizon. This is not a buy or sell signal.'
                 '</div></div>')
    return "".join(parts)


def _assessment_line(a) -> str:
    """'leverage cost normal · drawdown risk elevated', each with its number, then a separator; empty
    for a live.json written before the assessment existed (or one this cannot read)."""
    try:
        lev, dd = a["leverage"], a["drawdown"]
        levels = html.escape(str(lev["level"])), html.escape(str(dd["level"]))
    except (KeyError, TypeError, IndexError):
        return ""
    hurdle, p = lev.get("hurdle"), dd.get("p_dip20_3m")
    lc = (f' <span class="dim">(the S&amp;P 500 needs {hurdle:.1%}/yr for SPXL to break even over the long run)</span>'
          if isinstance(hurdle, (int, float)) else "")
    dip = (f' <span class="dim">({p:.0%} chance of a fall of 20% or more within 3 months)</span>'
           if isinstance(p, (int, float)) else "")
    return (f'leverage cost <b class="lvl">{levels[0]}</b>{lc} &middot; drawdown risk <b class="lvl">{levels[1]}</b>{dip}'
            ' &nbsp;|&nbsp; ')


def _period(live: dict):
    a = live.get("assessment")
    return a.get("period") if isinstance(a, dict) else None


def _span(period) -> str:
    """'1990-01..2026-08' -> 'from 1990 to 2026' (as report._span)."""
    try:
        a, b = str(period).split("..")
        return f"from {int(a[:4])} to {int(b[:4])}"
    except (TypeError, ValueError):
        return "of the backtest"


def _check_cell(row: dict, spot: float) -> str:
    """Percentile of the level, and the chance of reaching it from the side the quote is on."""
    dips = row["price"] <= spot
    p = row["p_touch_below"] if dips else row["p_touch_above"]
    return f'<td>{_ordinal(row["percentile"])} / {"dips" if dips else "rises"} {p:.0%}</td>'


def make_handler(root: str):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                          ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8"}
        _etag = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root, **kwargs)

        def version_string(self):
            return "spxlcast"

        def do_GET(self):  # noqa: N802
            self._route(body=True)

        def do_HEAD(self):  # noqa: N802
            self._route(body=False)

        def _route(self, body: bool):
            self._etag = None
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path in ("/", "/index.html"):
                try:
                    page = self._page()
                except Exception:  # noqa: BLE001 - answer 500 instead of dropping the connection
                    log.exception("status page failed")
                    self.send_error(500)
                    return
                self._send(page, "text/html; charset=utf-8", body)
                return
            if path == "/healthz":
                self._send(b"ok", "text/plain; charset=utf-8", body)
                return
            target = self._public_file()
            try:
                st = os.stat(target) if target else None
            except OSError:
                st = None
            if st is None:
                self.send_error(404)
                return
            # a strong validator: Last-Modified has whole seconds, so a copy fetched mid-write would pass it
            self._etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
            if self._etag in [t.strip() for t in self.headers.get("If-None-Match", "").split(",")]:
                self.send_response(304)
                self.end_headers()
                return
            if body:
                super().do_GET()
            else:
                super().do_HEAD()

        def _public_file(self) -> str | None:
            """The file a request names, if it lies inside output/ or logs/ once decoded and normalised
            (a directory is not a file, so nothing is ever listed)."""
            if self.path.split("?", 1)[0].split("#", 1)[0].endswith("/"):
                return None         # a directory request, even when the name before the slash is a file
            try:
                target = os.path.realpath(self.translate_path(self.path))
            except (OSError, ValueError):
                return None
            for d in ("output", "logs"):
                base = os.path.realpath(os.path.join(root, d))
                if target.startswith(base + os.sep) and os.path.isfile(target):
                    return target
            return None

        def _send(self, data: bytes, ctype: str, body: bool):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)

        def _page(self) -> bytes:
            out = os.path.join(root, "output")
            report = html.escape(_read(os.path.join(out, "report.txt")))
            score = html.escape(_read(os.path.join(out, "score.txt")))
            try:   # a new URL per chart, so no browser keeps showing the previous run's
                chart = f'<img src="/output/fan.png?v={os.stat(os.path.join(out, "fan.png")).st_mtime_ns}" alt="fan chart">'
            except OSError:
                chart = "<p>(no chart yet)</p>"
            stamp = "never"
            rp = os.path.join(out, "report.txt")
            if os.path.exists(rp):
                stamp = datetime.fromtimestamp(os.path.getmtime(rp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            links = " &nbsp;|&nbsp; ".join(
                f'<a href="/{d}/{n}">{n}</a>' if os.path.isfile(os.path.join(root, d, n))
                else f'<span class="dim">{n} (not yet)</span>' for d, n in LINKS)
            live_path = os.path.join(out, "live.json")
            live = render_live(live_path)
            refresh = ""
            if os.path.exists(live_path):   # the live loop runs: refresh every minute in the session, else every 5
                secs = 60 if _session_hours(datetime.now(timezone.utc)) else 300
                refresh = f'<meta http-equiv="refresh" content="{secs}">'
            return PAGE.format(stamp=stamp, links=links, report=report, score=score, chart=chart, live=live,
                               refresh=refresh).encode("utf-8")

        def list_directory(self, path):   # never reached (only files are routed here); refuse all the same
            self.send_error(404)
            return None

        def send_error(self, code, message=None, explain=None):
            self._etag = None               # an error is not the file whose validator was computed
            super().send_error(code, message, explain)

        def send_header(self, keyword, value):
            # Last-Modified has whole seconds. A copy served within a second of its file's last change
            # could be replaced in that same second, so it gets none: a client that revalidates by date
            # only (Chromium ignores the ETag on this HTTP/1.0 server) then refetches instead of a 304.
            if keyword.lower() == "last-modified":
                try:
                    if time.time() - email.utils.parsedate_to_datetime(value).timestamp() < 2.0:
                        return
                except (TypeError, ValueError):
                    pass
            super().send_header(keyword, value)

        def end_headers(self):
            for name, value in SECURITY_HEADERS:
                self.send_header(name, value)
            self.send_header("Cache-Control", "no-cache")   # revalidate: the files change with every run
            if self._etag:
                self.send_header("ETag", self._etag)
            super().end_headers()

        def log_request(self, code="-", size="-"):  # quieter container logs: skip the health probes
            if getattr(self, "path", "").split("?", 1)[0] != "/healthz":
                super().log_request(code, size)

    return Handler


def _interval(text: str) -> float:
    v = float(text)
    if not (math.isfinite(v) and 5.0 <= v <= 3600.0):
        raise argparse.ArgumentTypeError("must be between 5 and 3600 seconds")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spxlcast serve")
    ap.add_argument("--root", default=os.environ.get("SPXLCAST_DATA", "."), help="directory holding output/ and logs/")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--live", action="store_true", default=os.environ.get("SPXLCAST_LIVE", "0") == "1",
                    help="re-price the latest forecast at the live quote every minute of the session")
    ap.add_argument("--interval", type=_interval, default=60.0, help="seconds between live quotes (default 60)")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    stop = None
    if a.live:
        from .live import start_background
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        stop = start_background(root, a.interval)
    server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(root))

    def _terminate(signum, frame):
        # python runs as PID 1 in the container and would ignore SIGTERM: without this a replaced
        # revision keeps its live loop writing until it is killed
        if stop is not None:
            stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _terminate)
    except ValueError:   # not the main thread
        pass
    print(f"serving {a.root} on port {a.port}" + (" with the live loop" if a.live else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
