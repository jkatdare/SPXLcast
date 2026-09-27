"""Regression tests for the track-record scorer fixes (tracklog, evaluation, report)."""
import csv
import gzip
import io
import json
import math
import os
import warnings
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from scipy.stats import norm

from spxlcast import tracklog
from spxlcast.config import Config
from spxlcast.report import render_news, render_score
from spxlcast.tracklog import append_log, load_log, score_log


class _Sim:
    """Lognormal terminal prices with a zero median log return, in closed form."""

    def __init__(self, spot, horizons, sd=0.02):
        self.spot, self.horizons, self.sd = spot, list(horizons), sd
        self.rf_growth = {h: 1.0 + 0.04 * h / 252 for h in self.horizons}

    def quantiles(self, h, levels):
        return {float(l): self.spot * math.exp(self.sd * math.sqrt(h) * norm.ppf(l / 100.0)) for l in levels}

    def summary(self, h):
        return {"p_positive": 0.5, "p_beat_rf": 0.48, "p_drawdown_20": 0.01, "p_up_20": 0.01}


def _fc(spot, spot_date, asof, status="close", horizons=(5, 21), vix=17.0, sentiment=0.1):
    return SimpleNamespace(
        snap=SimpleNamespace(asof=asof), spot_date=str(spot_date), spot_status=status, spot=float(spot),
        rating=SimpleNamespace(label="HOLD", score=0.0, score_se=0.01, conviction="Low"),
        expected=SimpleNamespace(final=0.06), macro=SimpleNamespace(rf_3m=0.04, vix=vix),
        etf=SimpleNamespace(annual_cost=0.10), sentiment=SimpleNamespace(score=sentiment, n_used=50),
        sim=_Sim(spot, horizons), vol=SimpleNamespace(total_vol=lambda h: 0.16),
        data_quality=lambda: {"flags": [], "fred_series": 9, "news_fetched": 100},
    )


def _closes(n=160, seed=0, start="2026-01-02"):
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(200.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n))), index=dates)


def _after_close(day):
    return datetime(day.year, day.month, day.day, 21, 40, tzinfo=timezone.utc)


def _daily_log(path, closes, n, **kw):
    for d in closes.index[:n]:
        append_log(_fc(closes[d], d.date(), _after_close(d), **kw), path)


def _render(rep, path="log.csv"):
    buf = io.StringIO()
    render_score(rep, Console(file=buf, width=250, color_system=None), path)
    return buf.getvalue()


# ---- R1-25: one price basis --------------------------------------------------------------
def test_a_later_distribution_or_split_does_not_move_closed_scores(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 80)                               # every 1W/1M window closes by session 101
    ex = raw.index[130]
    dividend = raw.copy()
    dividend[dividend.index < ex] *= 1 - 0.30 / raw[ex]      # Yahoo's back-adjustment for a distribution
    split = raw / 2.0                                       # and for a 2-for-1 split
    base = {h.horizon: h for h in score_log(path, closes=raw).horizons}
    for series in (dividend, split):
        rep = {h.horizon: h for h in score_log(path, closes=series).horizons}
        for h in (5, 21):
            assert rep[h].n == base[h].n == 80
            assert rep[h].mean_pit == pytest.approx(base[h].mean_pit)
            assert rep[h].cov_5_95 == pytest.approx(base[h].cov_5_95)
            assert rep[h].mean_realised_return == pytest.approx(base[h].mean_realised_return)
            assert rep[h].real_dd20 == base[h].real_dd20 and rep[h].crps_model == pytest.approx(base[h].crps_model)


def test_intraday_row_is_scored_from_its_own_quote_in_the_right_units(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    d0, d12 = raw.index[0], raw.index[12]
    quote = raw[d12] * 1.01                                 # live quote, 1% above that day's close
    append_log(_fc(raw[d0], d0.date(), _after_close(d0), horizons=(5,)), path)
    append_log(_fc(quote, d12.date(), datetime(d12.year, d12.month, d12.day, 15, 40, tzinfo=timezone.utc),
                   status="intraday", horizons=(5,)), path)
    adjusted = raw.copy()
    adjusted[adjusted.index < raw.index[40]] *= 0.99         # a distribution after both windows
    expected = (raw.iloc[5] / raw[d0] + raw.iloc[17] / quote) / 2 - 1.0
    for series in (raw, adjusted):
        hs = score_log(path, closes=series).horizons[0]
        assert hs.n == 2 and hs.mean_realised_return == pytest.approx(expected, rel=1e-4)   # spot logged to 4 dp


def test_intraday_row_that_cannot_be_placed_after_a_split_is_not_scored(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    d = raw.index[3]
    append_log(_fc(raw[d] * 1.01, d.date(), datetime(d.year, d.month, d.day, 15, 40, tzinfo=timezone.utc),
                   status="intraday", horizons=(5,)), path)
    assert score_log(path, closes=raw).horizons[0].n == 1          # no close row: raw prices, as before
    rep = score_log(path, closes=raw / 3.0)                           # a 3-for-1 split since
    assert rep.horizons == [] and any("intraday row(s) not scored" in n for n in rep.notes)


# ---- R1-26: which run stands for a spot date --------------------------------------------
def test_next_morning_and_holiday_runs_do_not_replace_the_run_made_on_the_day(tmp_path):
    path = str(tmp_path / "log.csv")
    utc = timezone.utc
    # winter: 21:40Z is 16:40 ET on the day; the next 13:40Z run is 08:40 ET, before the open
    append_log(_fc(100, "2026-11-03", datetime(2026, 11, 3, 18, 40, tzinfo=utc), status="intraday", sentiment=0.5),
               path)
    append_log(_fc(101, "2026-11-03", datetime(2026, 11, 3, 21, 40, tzinfo=utc), sentiment=0.6), path)
    append_log(_fc(101, "2026-11-03", datetime(2026, 11, 4, 13, 40, tzinfo=utc), sentiment=-0.2), path)
    # nothing ran on 11-25 itself: the earliest later run stands in (Thanksgiving, then Friday)
    append_log(_fc(102, "2026-11-25", datetime(2026, 11, 26, 14, 40, tzinfo=utc), sentiment=0.3), path)
    append_log(_fc(102, "2026-11-25", datetime(2026, 11, 26, 21, 40, tzinfo=utc), sentiment=0.2), path)
    append_log(_fc(102, "2026-11-25", datetime(2026, 11, 27, 13, 40, tzinfo=utc), sentiment=-0.4), path)
    df = load_log(path).set_index("spot_date")
    assert df.loc["2026-11-03", "run_at"] == pd.Timestamp("2026-11-03 21:40", tz="UTC")
    assert df.loc["2026-11-03", "sentiment_score"] == pytest.approx(0.6)
    assert df.loc["2026-11-25", "run_at"] == pd.Timestamp("2026-11-26 14:40", tz="UTC")


# ---- R1-27: today's live bar is not a close ----------------------------------------------
def test_todays_partial_bar_is_not_scored_while_the_session_is_open(tmp_path, monkeypatch):
    import spxlcast.data as data
    path = str(tmp_path / "log.csv")
    raw = _closes(n=30, start="2026-03-02")
    today = raw.index[-1]
    d = raw.index[-6]                                        # its 1W window ends today
    append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(5,)), path)
    monkeypatch.setattr(data, "fetch_prices", lambda *a, **k: ({"SPXL": pd.DataFrame({"Close": raw})}, None))
    cfg = Config(cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(data, "session_state", lambda *a: (str(today.date()), True))
    rep = score_log(path, cfg=cfg)
    assert rep.n_scoreable == 0 and any("resolves at the next close" in n for n in rep.notes)
    monkeypatch.setattr(data, "session_state", lambda *a: (str(today.date()), False))
    assert score_log(path, cfg=cfg).n_scoreable == 1
    assert score_log(path, closes=raw).n_scoreable == 1                # a series passed in is left alone


# ---- R1-31 / R1-32: intervals and the CRPS columns ------------------------------------------
def test_intervals_are_never_zero_width_and_crps_columns_state_their_rows(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    for i, d in enumerate(raw.index[:80]):
        append_log(_fc(raw[d], d.date(), _after_close(d), vix=17.0 if i % 2 else None), path)
    by_h = {h.horizon: h for h in score_log(path, closes=raw).horizons}
    hs = by_h[5]
    lo, hi = hs.cov_5_95_ci
    assert hs.n_eff >= tracklog.MIN_INDEPENDENT and lo < hi and lo <= hs.cov_5_95 <= hi
    assert 0.0 <= hs.mean_pit_ci[0] < hs.mean_pit < hs.mean_pit_ci[1] <= 1.0
    assert hs.n == 80 and hs.n_crps == 40 and np.isfinite(hs.crps_skill) and all(np.isfinite(hs.crps_skill_ci))
    out = _render(score_log(path, closes=raw))
    assert "1W 40 of 80" in out
    # no row with the benchmark's inputs: the model's CRPS over all rows, the rest n/a (not "nan")
    path2 = str(tmp_path / "novix.csv")
    _daily_log(path2, raw, 40, vix=None)
    hs = score_log(path2, closes=raw).horizons[0]
    assert hs.n_crps == 0 and np.isfinite(hs.crps_model) and math.isnan(hs.crps_naive)
    assert "nan" not in _render(score_log(path2, closes=raw))


# ---- R1-30 / R2-08: the header and notes say what happened ----------------------------------
def test_model_version_filter_on_a_log_without_the_column(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 20)
    pd.read_csv(path, dtype=str).drop(columns="model_version").to_csv(path, index=False)
    rep = score_log(path, closes=raw, model_version="9.9.9")
    assert rep.n_scoreable == 0
    assert any("no rows from model version 9.9.9 (the log has unknown)" in n for n in rep.notes)
    assert score_log(path, closes=raw, model_version="unknown").n_scoreable == 20


def test_nothing_to_score_note_names_the_first_forecast_to_resolve_and_keeps_the_news_line(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes(n=40)
    for d in raw.index[:10]:                                 # old rows: 1M only, not due yet
        append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(21,)), path)
    for d in raw.index[36:40]:                               # new rows add 1W
        append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(5, 21)), path)
    rep = score_log(path, closes=raw.iloc[:21])
    assert rep.n_scoreable == 0 and rep.sentiment_n == 10
    note = next(n for n in rep.notes if n.startswith("nothing to score yet"))
    assert "the 1M forecast from 2026-01-02, resolves at the next close" in note
    out = _render(rep)
    assert "News score vs next-10-session SPXL return" in out


# ---- R1-16: a damaged archive run falls back to the logged quantiles ------------------------
def test_damaged_archive_runs_fall_back_to_the_logged_quantiles(tmp_path):
    path, archive = str(tmp_path / "log.csv"), str(tmp_path / "archive")
    raw = _closes()
    _daily_log(path, raw, 12, horizons=(5,))
    levels = list(np.linspace(0.5, 99.5, 103))
    files = []
    for d in raw.index[:12]:
        t = _after_close(d)
        folder = os.path.join(archive, "runs", t.strftime("%Y-%m-%d"))
        os.makedirs(folder, exist_ok=True)
        f = os.path.join(folder, t.strftime("%H%M%SZ") + ".json.gz")
        grid = {"percentiles": levels, "terminal": [_Sim(raw[d], [5]).quantiles(5, [l])[float(l)] for l in levels]}
        with gzip.open(f, "wt", encoding="utf-8") as fh:
            json.dump({"forecast": {"forecast": {"5": {"grid": grid}}}}, fh)
        files.append(f)
    assert score_log(path, closes=raw, archive_dir=archive).horizons[0].n_fine == 12
    data = open(files[0], "rb").read()
    open(files[0], "wb").write(data[: len(data) // 2])                  # truncated gzip (EOFError)
    open(files[1], "wb").write(gzip.compress(b"[1, 2, 3]"))            # not a record
    open(files[2], "wb").write(gzip.compress(json.dumps(
        {"forecast": {"forecast": {"5": {"grid": {"percentiles": [1, 2], "terminal": [1, 2, 3]}}}}}).encode()))
    open(files[3], "wb").write(data[:20] + b"\x00" * (len(data) - 20))  # zero-filled tail
    hs = score_log(path, closes=raw, archive_dir=archive).horizons[0]
    assert hs.n == 12 and hs.n_fine == 8


# ---- R1-22 / R1-23 / R1-24 / R1-28: the log file itself -------------------------------------
def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def test_a_bom_is_not_part_of_the_first_column(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 3)
    data = open(path, "rb").read()
    open(path, "wb").write(b"\xef\xbb\xbf" + data)                    # re-saved by Excel / PowerShell 5.1
    append_log(_fc(raw.iloc[3], raw.index[3].date(), _after_close(raw.index[3])), path)
    header = _rows(path)[0]
    assert header.count("run_at") == 1 and header[0] == "run_at"
    df = load_log(path)
    assert len(df) == 4 and df["run_at"].notna().all()
    assert score_log(path, closes=raw, archive_dir=str(tmp_path / "none")).n_scoreable == 4


def test_a_torn_last_row_is_set_aside_and_the_next_row_starts_on_its_own_line(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 3)
    data = open(path, "rb").read()
    open(path, "wb").write(data[:-200])                                 # a write cut short
    append_log(_fc(raw.iloc[3], raw.index[3].date(), _after_close(raw.index[3])), path)
    rows = _rows(path)
    assert len(rows) == 4 and len({len(r) for r in rows}) == 1          # header, 2 whole rows, the new one
    assert os.path.exists(path + ".partial") and len(load_log(path)) == 3
    # a whole row that only lost its line end keeps its place
    data = open(path, "rb").read()
    open(path, "wb").write(data.rstrip(b"\r\n"))
    append_log(_fc(raw.iloc[4], raw.index[4].date(), _after_close(raw.index[4])), path)
    assert len(load_log(path)) == 4
    # a file holding only a newline, and an empty one
    nl = str(tmp_path / "nl.csv")
    open(nl, "w").write("\n")
    append_log(_fc(200.0, "2026-01-02", _after_close(raw.index[0])), nl)
    assert len(load_log(nl)) == 1
    empty = str(tmp_path / "empty.csv")
    open(empty, "w").close()
    assert load_log(empty).empty and any("no rows" in n for n in score_log(empty, closes=raw).notes)


def test_damaged_lines_and_cells_do_not_stop_the_score_or_the_next_append(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 6, horizons=(5,))
    rows = _rows(path)
    rows[2] = rows[2] + rows[3]                                         # a line with a row glued onto it
    rows[4][rows[0].index("rf_3m")] = "4.1%"                            # malformed cells
    rows[5][rows[0].index("h5_q50")] = "1,234.5"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    with pytest.warns(pd.errors.ParserWarning):
        rep = score_log(path, closes=raw)
    assert rep.n_rows == 5 and rep.n_scoreable == 4                   # glued line skipped, bad quantile row not scored
    assert rep.horizons[0].n_crps == 3                                  # the bad rf_3m row has no benchmark
    d = raw.index[6]
    append_log(_fc(raw[d], d.date(), _after_close(d), horizons=(5, 10)), path)          # a build with new columns
    rows = _rows(path)
    assert len({len(r) for r in rows}) == 1 and len(rows) == 7         # the glued line moved aside, not misaligned
    assert "h10_q50" in rows[0] and os.path.exists(path + ".partial")


def test_an_interrupted_column_rewrite_leaves_the_log_intact(tmp_path, monkeypatch):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 30, horizons=(21,))
    before = open(path, "rb").read()

    def fail(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(tracklog.os, "replace", fail)
    with pytest.raises(OSError):
        append_log(_fc(raw.iloc[30], raw.index[30].date(), _after_close(raw.index[30]), horizons=(5, 21)), path)
    monkeypatch.undo()
    assert open(path, "rb").read() == before
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []
    append_log(_fc(raw.iloc[30], raw.index[30].date(), _after_close(raw.index[30]), horizons=(5, 21)), path)
    assert len(load_log(path)) == 31


def test_loading_a_wide_log_does_not_warn(tmp_path):
    path = str(tmp_path / "log.csv")
    raw = _closes()
    _daily_log(path, raw, 3, horizons=(5, 10, 21, 63, 126, 252))      # 110 columns
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # pandas 3 would warn "highly fragmented"
        assert len(load_log(path)) == 3


# ---- R1-39: outside text is printed, never parsed as markup ---------------------------------
def test_headlines_feeds_and_notes_are_printed_literally():
    item = SimpleNamespace(title="Stocks rally as S&P 500 hits record [/] high [red]x[/red]",
                           published=datetime(2026, 9, 26, 13, 0, tzinfo=timezone.utc))
    story = SimpleNamespace(score=0.5, weight=0.9, feeds=["[/x]"], item=item)
    s = SimpleNamespace(score=0.3, label="Positive", n_used=1, n_articles=1, drift_adjustment=0.01,
                        notes=["feed [/broken] skipped"], by_ticker={"[b]": 0.2},
                        top_positive=[story], top_negative=[])
    buf = io.StringIO()
    render_news(SimpleNamespace(sentiment=s), Console(file=buf, width=250, color_system=None))
    out = buf.getvalue()
    assert "record [/] high [red]x[/red]" in out and "[/x]" in out and "[/broken]" in out and "[b]" in out
    rep = tracklog.ScoreReport(n_rows=0, n_scoreable=0, first_date=None, last_date=None,
                               notes=["no rows from model version [/bold] (the log has [/])"])
    assert "[/bold]" in _render(rep, "logs/[red]odd[/].csv")
