"""Regression tests for the status page (serve.py) and the live loop (live.py) fixes."""
import csv
import glob
import http.client
import json
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from spxlcast import live, serve
from spxlcast.live import (GRID_PERCENTILES, append_spot_log, refresh_once, reprice, run_loop, settle_closed,
                           write_live)
from spxlcast.serve import make_handler, render_live

UTC = timezone.utc
FRI_1559 = datetime(2026, 9, 25, 19, 59, 30, tzinfo=UTC)       # Friday 15:59:30 ET, in the session
SAT_NOON = datetime(2026, 9, 26, 16, 0, tzinfo=UTC)            # Saturday 12:00 ET


def _base(spot=100.0, run_at="2026-09-25T17:40:00.218019+00:00", levels=(80.0, 90.0), label="HOLD", seed=0):
    """A stored forecast in the shape forecast_to_dict writes. Like the simulator's, the path min/max
    samples hold a point mass exactly at the spot (paths that never trade below / above the start)."""
    rng = np.random.default_rng(seed)
    pcts = list(GRID_PERCENTILES)
    forecast = {}
    for h, vol in ((5, 0.05), (126, 0.3)):
        term = spot * np.exp(rng.normal(0.0, vol, 20000))
        lo = np.minimum(term, spot * np.exp(-np.abs(rng.normal(0.0, vol, 20000))))
        hi = np.maximum(term, spot * np.exp(np.abs(rng.normal(0.0, vol, 20000))))
        lo[:3000] = spot
        hi[:3000] = spot
        forecast[str(h)] = {
            "horizon": h, "median_return": 0.0, "mean_return": vol ** 2 / 2, "p_positive": 0.5, "p_beat_rf": 0.49,
            "p_drawdown_20": float(np.mean(lo <= 0.8 * spot)), "p_up_20": float(np.mean(hi >= 1.2 * spot)),
            "quantile_prices": {f"{q:.1f}": float(np.percentile(term, q)) for q in (5, 25, 50, 75, 95)},
            "grid": {"percentiles": pcts, "terminal": np.percentile(term, pcts).tolist(),
                     "path_min": np.percentile(lo, pcts).tolist(), "path_max": np.percentile(hi, pcts).tolist()},
        }
    return {
        "run_at": run_at, "etf": "SPXL", "spot": spot, "spot_status": "intraday",
        "rating": {"label": label, "conviction": "Low", "score": 0.14, "score_se": 0.02, "horizon_days": 126},
        "forecast": forecast,
        "price_lookup": {f"{p:.1f}": [] for p in levels},
        "limit_ladder": {"126": [{"p_fill": 0.5, "price": 85.0, "vs_spot": -0.15}]},
    }


def _write(path, text, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as fh:
        fh.write(text)


class Site:
    def __init__(self, root):
        self.root = str(root)
        self.handler = make_handler(self.root)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True).start()

    def request(self, path, method="GET", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        conn.request(method, path, headers=headers or {})
        r = conn.getresponse()
        body = r.read()
        conn.close()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, body

    def path(self, *parts):
        return os.path.join(self.root, *parts)


@pytest.fixture
def site(tmp_path):
    root = tmp_path
    _write(str(root / "output" / "report.txt"), "RATING HOLD ┌── “quoted”")
    _write(str(root / "output" / "score.txt"), "nothing to score yet")
    _write(str(root / "output" / "fan.png"), b"\x89PNG old chart", "wb")
    _write(str(root / "output" / "forecast.json"), json.dumps(_base()))
    _write(str(root / "logs" / "forecast_log.csv"), "run_at,spot\n2026-09-16,277.75\n")
    _write(str(root / ".env"), "FRED_API_KEY=SECRET_SENTINEL")
    _write(str(root / "archive" / "news" / "2026-09.jsonl"), '{"title": "HEADLINE_SENTINEL"}\n')
    _write(str(root / ".cache" / "fred_x.pkl"), b"PICKLE_SENTINEL", "wb")
    s = Site(root)
    yield s
    s.server.shutdown()


# ---------------------------------------------------------------------------------------
# serve.py: what is served, for every method (R1-01, R3-01, R1-52, R1-55, R1-04)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_nothing_outside_output_and_logs_is_served(site, method):
    for path in ("/output/../.env", "/output/%2e%2e/.env", "/logs/..%2f.env", "/output/..%2farchive/news/2026-09.jsonl",
                 "/logs/%2e%2e%2farchive/news/2026-09.jsonl", "/output/../.cache/fred_x.pkl", "/.env", "/%2e%65nv",
                 "/.env?x=1", "/archive/news/2026-09.jsonl", "/%61rchive/news/2026-09.jsonl", "/archive/", "/archive",
                 "/output/", "/logs/", "/output", "/output/%2e%2e/", "/output/%00x", "/nope"):
        status, headers, body = site.request(path, method)
        assert status == 404, (method, path, status)
        assert b"SENTINEL" not in body and "last-modified" not in headers


def test_head_answers_like_get(site):
    for path in ("/", "/index.html", "/?utm_source=x", "/healthz", "/output/report.txt", "/logs/forecast_log.csv"):
        g_status, g_headers, g_body = site.request(path)
        h_status, h_headers, h_body = site.request(path, "HEAD")
        assert g_status == h_status == 200, path
        assert h_headers["content-length"] == g_headers["content-length"] == str(len(g_body)) and h_body == b""


def test_query_strings_do_not_break_the_page(site):
    for path in ("/?utm_source=newsletter", "/index.html?fbclid=IwAR0abc", "/?gclid=x#top"):
        status, _, body = site.request(path)
        assert status == 200 and b"RATING HOLD" in body
    assert site.request("/healthz?probe=1")[2] == b"ok"
    assert site.request("/output/report.txt?v=1")[0] == 200


def test_access_log_hides_only_the_health_probe(site):
    lines = []
    site.handler.log_message = lambda self, fmt, *args: lines.append(fmt % args)
    site.request("/healthz")
    site.request("/output/report.txt?x=/healthz")
    site.request("/nope/healthz")
    assert not any('"GET /healthz ' in ln for ln in lines)
    assert any("report.txt?x=/healthz" in ln for ln in lines) and any("/nope/healthz" in ln for ln in lines)


# ---------------------------------------------------------------------------------------
# serve.py: headers (R2-02, R2-03, R1-53, R1-65)
# ---------------------------------------------------------------------------------------
def test_files_revalidate_and_a_same_second_rewrite_is_not_a_304(site):
    png = site.path("output", "fan.png")
    os.utime(png, (os.path.getmtime(png) - 3600,) * 2)       # written an hour ago
    status, headers, body = site.request("/output/fan.png")
    assert status == 200 and headers["cache-control"] == "no-cache" and headers["etag"]
    old_etag, old_lm = headers["etag"], headers["last-modified"]
    page = site.request("/")[2].decode("utf-8")
    assert '<img src="/output/fan.png?v=' in page
    assert site.request("/output/fan.png", headers={"If-None-Match": old_etag})[0] == 304

    # the run rewrites the chart within the same second: Last-Modified cannot tell, the ETag can
    st = os.stat(png)
    _write(png, b"\x89PNG the complete new chart", "wb")
    os.utime(png, ns=(st.st_atime_ns, st.st_mtime_ns - st.st_mtime_ns % 10 ** 9 + 999_000_000))
    status, headers, body = site.request("/output/fan.png", headers={"If-None-Match": old_etag,
                                                                     "If-Modified-Since": old_lm})
    assert status == 200 and body == b"\x89PNG the complete new chart" and headers["etag"] != old_etag
    assert site.request("/")[2].decode("utf-8") != page     # new chart URL on the page


def test_security_headers_and_no_version_banner(site):
    for path in ("/", "/output/report.txt", "/nope", "/healthz"):
        _, headers, _ = site.request(path)
        assert headers["server"] == "spxlcast", path
        assert headers["x-content-type-options"] == "nosniff" and headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert headers["referrer-policy"] == "no-referrer" and "max-age" in headers["strict-transport-security"]


def test_text_downloads_declare_utf8(site):
    status, headers, body = site.request("/output/report.txt")
    assert headers["content-type"] == "text/plain; charset=utf-8" and "┌".encode("utf-8") in body
    assert site.request("/logs/forecast_log.csv")[1]["content-type"] == "text/csv; charset=utf-8"


# ---------------------------------------------------------------------------------------
# serve.py: the page (R1-02, R1-03, R1-12, R1-66, R1-67)
# ---------------------------------------------------------------------------------------
def _live_doc(**over):
    doc = reprice(_base(), 103.0, now=FRI_1559)
    doc.update(over)
    return doc


def test_header_links_only_for_files_that_exist(tmp_path, site):
    empty = Site(tmp_path / "empty")
    try:
        status, _, body = empty.request("/")
        text = body.decode("utf-8")
        assert status == 200 and text.count("(not yet)") == 5 and 'href="/logs/' not in text
    finally:
        empty.server.shutdown()
    text = site.request("/")[2].decode("utf-8")
    assert '<a href="/logs/forecast_log.csv">' in text and "spot_log.csv (not yet)" in text
    assert "live.json (not yet)" in text and '<a href="/output/report.txt">' in text


def test_bad_live_json_drops_only_the_live_block(site):
    good = _live_doc()
    no_rating = {k: v for k, v in good.items() if k != "rating"}
    null_score = {**good, "rating": {**good["rating"], "score": None}}
    naive_asof = {**good, "asof": "2026-09-25T19:59:30"}
    h = next(iter(good["horizons"]))
    no_q5 = json.loads(json.dumps(good))
    del no_q5["horizons"][h]["quantile_prices"]["5.0"]
    for doc in ([], {}, "text", no_rating, null_score, no_q5, {**good, "horizons": [1, 2]}):
        _write(site.path("output", "live.json"), json.dumps(doc))
        assert render_live(site.path("output", "live.json"), now=SAT_NOON) == ""
        status, _, body = site.request("/")
        assert status == 200 and b"RATING HOLD" in body and b'class="live"' not in body
        assert b'http-equiv="refresh"' in body      # keeps refreshing, so it recovers when the loop rewrites it
    _write(site.path("output", "live.json"), json.dumps(naive_asof))    # read as UTC, like health.py does
    assert "SPXL 103.00" in render_live(site.path("output", "live.json"), now=SAT_NOON)


def test_page_error_is_a_500_not_a_dropped_connection(site, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("template bug")
    monkeypatch.setattr(serve, "_read", boom)
    assert site.request("/")[0] == 500


def test_live_block_follows_the_clock_after_the_close(tmp_path):
    path = str(tmp_path / "live.json")
    _write(path, json.dumps(_live_doc()))                   # the 15:59 tick: session_open is still True
    sat = render_live(path, now=SAT_NOON)
    assert "market closed" in sat and "stale" not in sat and "3:59 PM ET, 25 Sep" in sat
    fri = render_live(path, now=FRI_1559)
    assert "market open" in fri and "stale" not in fri
    _write(path, json.dumps(_live_doc(asof="2026-09-25T17:59:30+00:00")))    # the loop stopped 2 hours ago
    assert "market open" in render_live(path, now=FRI_1559) and "stale: 120 min old" in render_live(path, now=FRI_1559)
    thanksgiving = datetime(2026, 11, 26, 19, 0, tzinfo=UTC)       # 14:00 ET on an NYSE holiday
    assert "market closed" in render_live(path, now=thanksgiving) and "stale" not in render_live(path, now=thanksgiving)
    _write(path, json.dumps(_live_doc(session_open=False)))   # settled Friday; no quote since Monday's open
    monday = render_live(path, now=datetime(2026, 9, 28, 15, 0, tzinfo=UTC))
    assert "market open" in monday and "stale" in monday


def test_page_refresh_follows_the_session(site, monkeypatch):
    assert b'http-equiv="refresh"' not in site.request("/")[2]        # no live loop, no refresh
    _write(site.path("output", "live.json"), json.dumps(_live_doc()))
    monkeypatch.setattr(serve, "_session_hours", lambda now: False)
    assert b'content="300"' in site.request("/")[2]
    monkeypatch.setattr(serve, "_session_hours", lambda now: True)
    assert b'content="60"' in site.request("/")[2]


def test_ordinals_match_the_report():
    from spxlcast.report import ordinal
    assert [serve._ordinal(n) for n in range(0, 124)] == [ordinal(n) for n in range(0, 124)]
    cells = [serve._check_cell({"price": 90.0, "percentile": p, "p_touch_below": 0.3, "p_touch_above": 1.0}, 100.0)
             for p in (1, 2, 3, 11, 22.4)]
    assert [c.split(" /")[0][4:] for c in cells] == ["1st", "2nd", "3rd", "11th", "22nd"]


def test_level_above_the_quote_shows_the_chance_to_rise(tmp_path):
    doc = reprice(_base(levels=(90.0, 110.0)), 100.0, now=FRI_1559)
    path = str(tmp_path / "live.json")
    _write(path, json.dumps(doc))
    block = render_live(path, now=FRI_1559)
    h = doc["horizons"]["126"]["price_lookup"]
    below, above = h[0], h[1]
    assert f"dips {below['p_touch_below']:.0%}" in block and f"rises {above['p_touch_above']:.0%}" in block
    assert 0.0 < above["p_touch_above"] < 1.0 and "P(dips)" not in block


def test_interval_is_validated(monkeypatch):
    monkeypatch.setattr(serve, "ThreadingHTTPServer", None)     # never start a server here
    for bad in ("0", "-5", "4.9", "nan", "inf", "7200", "x"):
        with pytest.raises(SystemExit):
            serve.main(["--interval", bad])
    assert serve._interval("60") == 60.0


# ---------------------------------------------------------------------------------------
# live.py
# ---------------------------------------------------------------------------------------
def test_a_level_at_the_quote_counts_as_touched():
    base = _base(spot=294.17, levels=(200.0, 250.0))
    assert 294.17 * (250.0 / 294.17) > 250.0 and 294.17 * (200.0 / 294.17) < 200.0   # the float edge
    for spot in (250.0, 200.0):
        out = reprice(base, spot, now=FRI_1559)
        for h in out["horizons"].values():
            row = next(r for r in h["price_lookup"] if r["price"] == spot)
            assert row["p_touch_below"] == 1.0 and row["p_touch_above"] == 1.0, (spot, h["horizon"])


def test_base_run_at_matches_the_track_record_format(tmp_path):
    assert reprice(_base(), 101.0)["base_run_at"] == "2026-09-25T17:40:00Z"
    assert reprice(_base(run_at="2026-09-22T21:40:21.999999+00:00"), 101.0)["base_run_at"] == "2026-09-22T21:40:21Z"
    assert reprice(_base(run_at="2026-09-22T21:40:21Z"), 101.0)["base_run_at"] == "2026-09-22T21:40:21Z"
    _write(str(tmp_path / "output" / "forecast.json"), json.dumps(_base()))
    refresh_once(str(tmp_path), fetch=lambda t: 101.0, now=FRI_1559)
    rows = list(csv.DictReader(open(tmp_path / "logs" / "spot_log.csv", encoding="utf-8")))
    assert rows[0]["base_run_at"] == "2026-09-25T17:40:00Z"


def test_settle_marks_the_close_and_rebases_on_a_new_run(tmp_path):
    root = str(tmp_path)
    fc = tmp_path / "output" / "forecast.json"
    lj = tmp_path / "output" / "live.json"
    _write(str(fc), json.dumps(_base()))
    tick = refresh_once(root, fetch=lambda t: 103.0, now=FRI_1559)
    assert tick["session_open"] is True

    closed = settle_closed(root)                          # after the bell: same quote and time, now closed
    assert closed["session_open"] is False and closed["spot"] == 103.0 and closed["asof"] == tick["asof"]
    assert settle_closed(root) is None                    # settled: nothing rewritten
    before = lj.read_bytes()
    assert settle_closed(root) is None and lj.read_bytes() == before

    # the after-close run lands: restated at its own spot and time, with its rating
    _write(str(fc), json.dumps(_base(spot=105.0, run_at="2026-09-25T21:40:21.5+00:00", label="BUY")))
    assert settle_closed(root) is not None
    rebased = json.loads(lj.read_text(encoding="utf-8"))
    assert rebased["base_run_at"] == "2026-09-25T21:40:21Z" and rebased["spot"] == 105.0
    assert rebased["rating"]["label"] == "BUY" and rebased["session_open"] is False
    assert rebased["asof"] == "2026-09-25T21:40:21+00:00" and rebased["change_vs_base"] == 0.0
    block = render_live(str(lj), now=SAT_NOON)
    assert "market closed" in block and ">BUY<" in block and "+0.00% since the full run at 105.00" in block
    rows = list(csv.DictReader(open(tmp_path / "logs" / "spot_log.csv", encoding="utf-8")))
    assert len(rows) == 1                                 # settling never logs a price row


def test_settle_without_live_json_or_forecast(tmp_path):
    assert settle_closed(str(tmp_path)) is None and not (tmp_path / "output" / "live.json").exists()
    _write(str(tmp_path / "output" / "forecast.json"), json.dumps(_base()))
    _write(str(tmp_path / "output" / "live.json"), "{not json")
    out = settle_closed(str(tmp_path))
    assert out["spot"] == 100.0 and out["session_open"] is False


def test_closed_loop_settles_without_quotes(tmp_path, monkeypatch):
    root = str(tmp_path)
    _write(str(tmp_path / "output" / "forecast.json"), json.dumps(_base()))
    refresh_once(root, fetch=lambda t: 103.0, now=FRI_1559)
    monkeypatch.setattr(live, "ny_now", lambda: pd.Timestamp("2026-09-26 12:00", tz="America/New_York"))
    calls, stop = [], threading.Event()
    t = threading.Thread(target=run_loop, args=(root, 0.01, stop, lambda s: calls.append(s), 0.01), daemon=True)
    t.start()
    t.join(0.2)
    stop.set()
    t.join(1.0)
    doc = json.loads((tmp_path / "output" / "live.json").read_text(encoding="utf-8"))
    assert calls == [] and not t.is_alive() and doc["session_open"] is False and doc["spot"] == 103.0


def test_the_loop_wakes_at_the_open(tmp_path, monkeypatch):
    ny = lambda s: pd.Timestamp(s, tz="America/New_York")   # noqa: E731
    assert live._seconds_to_open(ny("2026-09-28 09:29:30")) == 30.0
    assert live._seconds_to_open(ny("2026-09-25 16:00")) == 17.5 * 3600
    assert live._seconds_to_open(ny("2026-10-31 10:00")) == 24.5 * 3600      # across the fall-back night
    assert live._seconds_to_open(ny("2026-09-28 09:30")) == 24 * 3600

    class OneWait:
        def __init__(self):
            self.waits = []

        def is_set(self):
            return bool(self.waits)

        def wait(self, t):
            self.waits.append(t)
    monkeypatch.setattr(live, "ny_now", lambda: ny("2026-09-28 09:29:30"))
    stop = OneWait()
    run_loop(str(tmp_path), 60.0, stop, lambda s: None, 300.0)
    assert 30.0 < stop.waits[0] <= 31.0                     # not whatever is left of a 5-minute nap


def test_sigterm_stops_the_live_loop_and_the_server(tmp_path, monkeypatch):
    import signal
    handlers, stops = {}, []
    monkeypatch.setattr(serve.signal, "signal", lambda sig, fn: handlers.__setitem__(sig, fn))
    monkeypatch.setattr(live, "start_background", lambda root, interval: stops.append(threading.Event()) or stops[-1])

    class FakeServer:
        def __init__(self, addr, handler):
            self.done = threading.Event()

        def serve_forever(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            assert self.done.wait(2)

        def shutdown(self):
            self.done.set()
    monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeServer)
    assert serve.main(["--root", str(tmp_path), "--live", "--port", "0"]) == 0 and stops[0].is_set()


def test_write_live_uses_its_own_temp_file_and_retries(tmp_path, monkeypatch):
    root = str(tmp_path)
    docs = [_live_doc(spot=100.0 + i) for i in range(2)]
    errors = []

    def writer(doc):
        try:
            for _ in range(40):
                write_live(root, doc)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=writer, args=(d,)) for d in docs]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert json.loads((tmp_path / "output" / "live.json").read_text(encoding="utf-8"))["spot"] in (100.0, 101.0)
    assert glob.glob(str(tmp_path / "output" / "*.tmp")) == []

    real = os.replace
    fails = []

    def flaky(src, dst):
        if len(fails) < 2:
            fails.append(src)
            raise PermissionError(13, "in use")
        real(src, dst)
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    monkeypatch.setattr(os, "replace", flaky)
    write_live(root, docs[0])
    assert len(fails) == 2
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError(13, "in use")))
    with pytest.raises(PermissionError):
        write_live(root, docs[1])
    assert glob.glob(str(tmp_path / "output" / "*.tmp")) == []


def test_a_failed_page_write_still_logs_the_price(tmp_path, monkeypatch):
    _write(str(tmp_path / "output" / "forecast.json"), json.dumps(_base()))

    def fail(root, doc):
        raise PermissionError(13, "in use")
    monkeypatch.setattr(live, "write_live", fail)
    with pytest.raises(PermissionError):
        refresh_once(str(tmp_path), fetch=lambda t: 104.0, now=FRI_1559)
    rows = list(csv.DictReader(open(tmp_path / "logs" / "spot_log.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["spot"] == "104.0000"


def test_spot_log_survives_a_torn_last_row(tmp_path):
    root = str(tmp_path)
    append_spot_log(root, _live_doc(spot=103.0))
    append_spot_log(root, _live_doc(spot=104.0))
    path = tmp_path / "logs" / "spot_log.csv"
    data = path.read_bytes()
    path.write_bytes(data[:-40])                          # the last row cut mid-way
    append_spot_log(root, _live_doc(spot=105.0))
    df = pd.read_csv(path)
    assert df["spot"].dtype == float and df["spot"].tolist()[0] == 103.0 and df["spot"].tolist()[-1] == 105.0


def test_spot_log_header_once_when_emptied_or_created_together(tmp_path):
    _write(str(tmp_path / "a" / "logs" / "spot_log.csv"), "")          # emptied by hand
    append_spot_log(str(tmp_path / "a"), _live_doc())
    assert pd.read_csv(tmp_path / "a" / "logs" / "spot_log.csv")["spot"].tolist() == [103.0]

    for k in range(10):                    # several loops find the file absent, or emptied, at once
        root = str(tmp_path / f"r{k}")
        if k % 2:
            _write(os.path.join(root, "logs", "spot_log.csv"), "")
        barrier = threading.Barrier(6)

        def go():
            barrier.wait()
            append_spot_log(root, _live_doc())
        threads = [threading.Thread(target=go) for _ in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        text = open(os.path.join(root, "logs", "spot_log.csv"), encoding="utf-8").read()
        assert text.count("ts_utc") == 1 and text.startswith("ts_utc")
        df = pd.read_csv(os.path.join(root, "logs", "spot_log.csv"))
        assert df["spot"].dtype == float
        # Windows emulates append mode with a seek and a write, so threads can overwrite each other's
        # row there; appends are atomic on Linux, where the loop runs
        assert len(df) == 6 if sys.platform != "win32" else 1 <= len(df) <= 6
