"""Tiny status page for the hosted daily run: shows the latest report, fan chart, track-record
score and offers the CSV/JSON for download. Standard library only, read-only, no forms.

    python -m spxlcast serve --root /data --port 8000

Expects the daily job to have written, under ``root``:
    output/report.txt   output/forecast.json   output/fan.png   output/score.txt
    logs/forecast_log.csv
"""
from __future__ import annotations

import argparse
import html
import mimetypes
import os
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SPXLcast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0f1419;color:#e6e6e6;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} .meta{{color:#9aa;font-size:13px;margin-bottom:16px}}
pre{{background:#151b22;border:1px solid #2a323c;border-radius:6px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.35}}
img{{max-width:100%;border-radius:6px;border:1px solid #2a323c}} a{{color:#7cc4ff}}
.row{{display:flex;flex-wrap:wrap;gap:16px}} .row>div{{flex:1 1 480px}}
</style></head><body>
<h1>SPXLcast</h1>
<div class="meta">latest run {stamp} &nbsp;|&nbsp; <a href="/logs/forecast_log.csv">forecast_log.csv</a> &nbsp;|&nbsp;
<a href="/output/forecast.json">forecast.json</a> &nbsp;|&nbsp; <a href="/output/report.txt">report.txt</a></div>
<div class="row"><div><h2>Latest forecast</h2><pre>{report}</pre></div>
<div><h2>Fan chart</h2>{chart}<h2>Track record</h2><pre>{score}</pre></div></div>
</body></html>"""


def _read(path: str, limit: int = 200_000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return "(not available yet)"


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
                body = PAGE.format(stamp=stamp, report=report, score=score, chart=chart).encode("utf-8")
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
    a = ap.parse_args(argv)
    mimetypes.add_type("text/csv", ".csv")
    server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(os.path.abspath(a.root)))
    print(f"serving {a.root} on port {a.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
