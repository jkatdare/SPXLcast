"""Tiny status page for the hosted run: shows the live minute-by-minute re-pricing, the latest full
report, fan chart, track-record score and offers the CSV/JSON for download. Standard library only,
read-only, no forms.

    python -m spxlcast serve --root /data --port 8000 --live

Expects the scheduled job to have written, under ``root``:
    output/report.txt   output/forecast.json   output/fan.png   output/score.txt
    logs/forecast_log.csv
and, with ``--live`` (or SPXLCAST_LIVE=1), keeps writing every minute of the session:
    output/live.json    logs/spot_log.csv
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SPXLcast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<style>
body{{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0f1419;color:#e6e6e6;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px}} .meta{{color:#9aa;font-size:13px;margin-bottom:16px}}
pre{{background:#151b22;border:1px solid #2a323c;border-radius:6px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.35}}
img{{max-width:100%;border-radius:6px;border:1px solid #2a323c}} a{{color:#7cc4ff}}
.row{{display:flex;flex-wrap:wrap;gap:16px}} .row>div{{flex:1 1 480px}}
.live{{background:#151b22;border:1px solid #2a323c;border-radius:6px;padding:12px 16px;margin-bottom:16px}}
.live .spot{{font-size:28px;font-weight:bold}} .live .BUY{{color:#5fd37a}} .live .HOLD{{color:#e6c34a}} .live .SELL{{color:#ff6b6b}}
.live table{{border-collapse:collapse;margin-top:8px;font-size:13px}} .live th,.live td{{padding:3px 10px;text-align:right;border-bottom:1px solid #2a323c}}
.live th:first-child,.live td:first-child{{text-align:left}} .dim{{color:#9aa}}
</style></head><body>
<h1>SPXLcast</h1>
<div class="meta">latest full run {stamp} &nbsp;|&nbsp; <a href="/logs/forecast_log.csv">forecast_log.csv</a> &nbsp;|&nbsp;
<a href="/logs/spot_log.csv">spot_log.csv</a> &nbsp;|&nbsp; <a href="/output/forecast.json">forecast.json</a> &nbsp;|&nbsp;
<a href="/output/live.json">live.json</a> &nbsp;|&nbsp; <a href="/output/report.txt">report.txt</a></div>
{live}
<div class="row"><div><h2>Latest full forecast</h2><pre>{report}</pre></div>
<div><h2>Fan chart</h2>{chart}<h2>Track record</h2><pre>{score}</pre></div></div>
</body></html>"""


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


def _fmt_ny(asof_ny: str) -> str:
    """'2026-09-22 14:31:05 EDT' -> '2:31 PM ET, 22 Sep'."""
    try:
        t = datetime.strptime(asof_ny[:19], "%Y-%m-%d %H:%M:%S")
        return t.strftime("%I:%M %p ET, %d %b").lstrip("0")
    except ValueError:
        return html.escape(asof_ny)


def render_live(path: str, now: datetime | None = None) -> str:
    """The live block: latest quote, the distribution restated at it, and the fixed price checks."""
    try:
        with open(path, encoding="utf-8") as fh:
            live = json.load(fh)
    except (OSError, ValueError):
        return ""
    now = now or datetime.now(timezone.utc)
    try:
        age = (now - datetime.fromisoformat(live["asof"])).total_seconds() / 60.0
    except (KeyError, ValueError):
        age = float("nan")
    r = live["rating"]
    label = html.escape(str(r["label"]))
    state = "market open" if live.get("session_open") else "market closed"
    stale = ""
    if live.get("session_open") and age == age and age > 5:
        stale = f' <span class="dim">(stale: {age:.0f} min old)</span>'
    parts = [
        '<div class="live">',
        f'<div><span class="spot">{html.escape(str(live.get("etf", "SPXL")))} {live["spot"]:,.2f}</span> '
        f'<span class="dim">as of {_fmt_ny(live["asof_ny"])}, {state}{stale}</span></div>',
        f'<div>rating <b class="{label}">{label}</b> ({html.escape(str(r["conviction"])).lower()}, score {r["score"]:+.2f}) '
        f'over {_horizon_label(int(r["horizon_days"]))} &nbsp;|&nbsp; {live["change_vs_base"]:+.2%} since the full run '
        f'at {live["base_spot"]:,.2f}</div>',
    ]
    horizons = sorted(live.get("horizons", {}).values(), key=lambda x: x["horizon"])
    levels = sorted({row["price"] for h in horizons for row in h.get("price_lookup", [])})
    head = "".join(f"<th>{p:,.0f}: pctile / P(dips)</th>" for p in levels)
    rows = []
    for h in horizons:
        q = h["quantile_prices"]
        cells = "".join(
            f'<td>{row["percentile"]:.0f}th / {row["p_touch_below"]:.0%}</td>'
            for row in h.get("price_lookup", []))
        rows.append(f'<tr><td>{_horizon_label(h["horizon"])}</td><td>{q["5.0"]:,.0f}</td><td>{q["50.0"]:,.0f}</td>'
                    f'<td>{q["95.0"]:,.0f}</td><td>{h["p_positive"]:.0%}</td><td>{h["p_drawdown_20"]:.0%}</td>{cells}</tr>')
    parts.append('<table><tr><th>Horizon</th><th>Bad case (5%)</th><th>Typical</th><th>Good case (95%)</th>'
                 f'<th>P(up)</th><th>P(-20% dip)</th>{head}</tr>' + "".join(rows) + "</table>")
    parts.append('<div class="dim">Restated at the live quote from the last full simulation: the return distribution '
                 'and the rating are recomputed hourly; prices scale with the quote in between.</div></div>')
    return "".join(parts)


def make_handler(root: str):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root, **kwargs)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                report = html.escape(_read(os.path.join(root, "output", "report.txt")))
                score = html.escape(_read(os.path.join(root, "output", "score.txt")))
                png = os.path.join(root, "output", "fan.png")
                chart = '<img src="/output/fan.png" alt="fan chart">' if os.path.exists(png) else "<p>(no chart yet)</p>"
                stamp = "never"
                rp = os.path.join(root, "output", "report.txt")
                if os.path.exists(rp):
                    stamp = datetime.fromtimestamp(os.path.getmtime(rp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                live = render_live(os.path.join(root, "output", "live.json"))
                refresh = '<meta http-equiv="refresh" content="60">' if live else ""
                body = PAGE.format(stamp=stamp, report=report, score=score, chart=chart, live=live,
                                   refresh=refresh).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if not (self.path.startswith("/output/") or self.path.startswith("/logs/")):
                self.send_error(404)
                return
            return super().do_GET()

        def log_message(self, fmt, *args):  # quieter container logs: skip health probes
            if not any("/healthz" in str(a) for a in args):
                super().log_message(fmt, *args)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spxlcast serve")
    ap.add_argument("--root", default=os.environ.get("SPXLCAST_DATA", "."), help="directory holding output/ and logs/")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--live", action="store_true", default=os.environ.get("SPXLCAST_LIVE", "0") == "1",
                    help="re-price the latest forecast at the live quote every minute of the session")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between live quotes (default 60)")
    a = ap.parse_args(argv)
    mimetypes.add_type("text/csv", ".csv")
    root = os.path.abspath(a.root)
    if a.live:
        import logging
        from .live import start_background
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        start_background(root, a.interval)
    server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(root))
    print(f"serving {a.root} on port {a.port}" + (" with the live loop" if a.live else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
