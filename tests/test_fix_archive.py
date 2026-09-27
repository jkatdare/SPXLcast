import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from spxlcast.archive import _keys, append_news, load_run, news_records, story_key, write_run
from spxlcast.config import Config
from spxlcast.data import NewsItem
from spxlcast.sentiment import analyze_news

NOW = datetime(2026, 9, 28, 14, 40, 5, tzinfo=timezone.utc)


def rec(key, title=None, seen=NOW):
    return {"key": key, "first_seen": seen.isoformat(), "title": title or key}


def readable(path):
    out = []
    for line in path.read_bytes().split(b"\n"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def test_lone_surrogate_headline_is_archived_with_the_rest(tmp_path):
    cut = json.loads('"Stocks rise as markets cheer \\ud83d"')     # a teaser cut in the middle of an emoji
    heads = [("Stocks climb on strong jobs report", ""), (cut, "teaser cut \ud83d"),
             ("Treasury yields ease as inflation cools", ""), ("Dow gains as oil prices drop", "")]
    items = [NewsItem("SPY", t, s, NOW - timedelta(hours=k), "R", f"https://x.test/{k}")
             for k, (t, s) in enumerate(heads, 1)]
    recs = news_records(analyze_news(items, Config(), now=NOW), NOW.isoformat())
    assert len(recs) == 4
    assert append_news(str(tmp_path), recs) == 4
    data = (tmp_path / "news" / "2026-09.jsonl").read_bytes()
    assert data.isascii() and cut in [json.loads(x)["title"] for x in data.splitlines()]
    assert append_news(str(tmp_path), recs) == 0            # the next run does not retry it


def test_line_torn_inside_a_character_does_not_block_archiving(tmp_path):
    news = tmp_path / "news"
    news.mkdir()
    old = json.dumps(rec("a", "Café"), ensure_ascii=False).encode()   # as the previous version wrote it
    torn = json.dumps({"key": "torn", "title": "Wall Street’s rally stalls"}, ensure_ascii=False).encode()
    (news / "2026-09.jsonl").write_bytes(old + b"\n" + torn[: torn.index("’".encode()) + 1])
    assert append_news(str(tmp_path), [rec("a", "Café"), rec("b"), rec("c")]) == 2
    assert [r["key"] for r in readable(news / "2026-09.jsonl")] == ["a", "b", "c"]
    october = datetime(2026, 10, 5, 14, 40, tzinfo=timezone.utc)      # last month's file is still read
    assert append_news(str(tmp_path), [rec("b", seen=october), rec("d", seen=october)]) == 1


def test_hand_edited_month_file_keeps_its_keys(tmp_path):
    news = tmp_path / "news"
    news.mkdir()
    (news / "2026-09.jsonl").write_bytes(b"\xef\xbb\xbf" + json.dumps(rec("first")).encode() + b"\n"
                                         + b'{"key": "b", "title": "caf\xe9"}\n')     # BOM, then a cp1252 byte
    assert _keys(str(news / "2026-09.jsonl")) == {("first", "first"), ("b", "caf")}
    assert append_news(str(tmp_path), [rec("first"), rec("b", "café"), rec("new")]) == 1


def test_torn_last_line_does_not_swallow_the_next_record(tmp_path):
    path = tmp_path / "news" / "2026-09.jsonl"
    append_news(str(tmp_path), [rec("a"), rec("b")])
    with open(path, "ab") as fh:
        fh.write(b'{"key": "c", "tit')
    assert append_news(str(tmp_path), [rec("d"), rec("e")]) == 2
    assert [r["key"] for r in readable(path)] == ["a", "b", "d", "e"]
    assert append_news(str(tmp_path), [rec("d"), rec("e")]) == 0


def test_two_runs_in_the_same_second_keep_both_records(tmp_path):
    first = {"run_at": "2026-09-25T20:40:07.100000+00:00", "n_paths": 2000}
    second = {"run_at": "2026-09-25T20:40:07.900000+00:00", "n_paths": 3000}
    p1, p2 = write_run(str(tmp_path), first), write_run(str(tmp_path), second)
    assert p1.endswith("204007Z.json.gz") and p2.endswith("204007Z-1.json.gz")
    assert load_run(p1) == first and load_run(p2) == second
    assert sorted(os.listdir(os.path.dirname(p1))) == ["204007Z-1.json.gz", "204007Z.json.gz"]


def test_failed_run_write_leaves_no_file(tmp_path):
    with pytest.raises(ValueError):
        write_run(str(tmp_path), {"run_at": NOW.isoformat(), "bad": float("nan")})
    assert os.listdir(tmp_path / "runs" / "2026-09-28") == []


def test_headline_rewritten_under_the_same_url_gets_a_new_line(tmp_path):
    url = "https://finance.yahoo.com/markets/live/stock-market-today-333.html"
    heads = ["Stock market today: Dow, S&P 500 rally as Fed signals rate cuts",
             "Stock market today: Stocks plunge as recession fears return"]
    added = []
    for i, head in enumerate(heads + heads[1:]):              # the second headline is seen twice
        t = NOW + timedelta(hours=i)
        item = NewsItem("SPY", head, "", NOW - timedelta(minutes=5), "Yahoo Finance", url)
        added.append(append_news(str(tmp_path), news_records(analyze_news([item], Config(), now=t), t.isoformat())))
    assert added == [1, 1, 0]
    lines = readable(tmp_path / "news" / "2026-09.jsonl")
    assert [x["key"] for x in lines] == [url, url] and [x["title"] for x in lines] == heads
    assert lines[0]["score"] > 0 > lines[1]["score"]
    assert append_news(str(tmp_path), [rec(url, heads[1].upper() + "!")]) == 0   # case/punctuation only


def test_fragment_only_url_falls_back_to_the_title_key():
    assert story_key("#top", "Dow slips", "2026-09-25T10:00:00+00:00") == "dow slips|2026-09-25"
    assert story_key("https://x.test/a#top", "Dow slips", "2026-09-25") == "https://x.test/a"
