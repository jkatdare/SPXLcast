import http.client
import threading
from http.server import ThreadingHTTPServer

from spxlcast.serve import make_handler


def test_status_page_and_downloads(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "output" / "report.txt").write_text("RATING HOLD <b>&", encoding="utf-8")
    (tmp_path / "output" / "score.txt").write_text("nothing to score yet", encoding="utf-8")
    (tmp_path / "logs" / "forecast_log.csv").write_text("run_at,spot\n2026-09-16,277.75\n", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(tmp_path)))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def get(path):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)   # the server closes after each reply
        conn.request("GET", path)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    try:
        status, body = get("/")
        text = body.decode("utf-8")
        assert status == 200 and "RATING HOLD &lt;b&gt;&amp;" in text and "nothing to score yet" in text
        status, body = get("/logs/forecast_log.csv")
        assert status == 200 and b"277.75" in body
        assert get("/healthz")[1] == b"ok"
        assert get("/etc/passwd")[0] == 404
    finally:
        server.shutdown()
